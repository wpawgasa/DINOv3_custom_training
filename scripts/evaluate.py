#!/usr/bin/env python3
"""
DINOv3 Evaluation Script

Comprehensive evaluation script for DINOv3 models with support for multiple
evaluation modes, benchmarking, and results export.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import create_transforms
from data.dataset import create_dataset
from evaluation.benchmarks import BenchmarkRunner, RobustnessEvaluator
from evaluation.evaluators import create_evaluator
from evaluation.visualization import create_visualizer
from models.model_factory import create_model
from utils.config import load_hierarchical_config
from utils.device import get_device_manager
from utils.logging import setup_logging
from utils.reproducibility import set_seed
from utils.schemas import ExperimentConfig


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv3 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data
    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (optional, will try to load from checkpoint)",
    )
    parser.add_argument(
        "--data-path", "-d", type=str, required=True, help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--annotation-format",
        type=str,
        default="imagefolder",
        choices=["imagefolder", "coco", "csv", "json"],
        help="Dataset annotation format",
    )

    # Evaluation modes
    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        choices=["standard", "benchmark", "robustness", "feature_analysis"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="*",
        default=[],
        choices=["imagenet", "cifar10", "cifar100", "medical", "satellite"],
        help="Benchmark datasets to evaluate on",
    )

    # Output and reporting
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        nargs="*",
        default=["json"],
        choices=["json", "csv", "html", "pdf"],
        help="Export formats for results",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", help="Save individual predictions"
    )
    parser.add_argument(
        "--save-features", action="store_true", help="Save extracted features"
    )

    # Visualization
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization plots"
    )
    parser.add_argument(
        "--attention-maps",
        action="store_true",
        help="Generate attention visualization maps",
    )
    parser.add_argument(
        "--num-vis-samples",
        type=int,
        default=100,
        help="Number of samples for visualization",
    )

    # Inference settings
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for evaluation",
    )

    # Analysis options
    parser.add_argument(
        "--per-class-metrics", action="store_true", help="Compute per-class metrics"
    )
    parser.add_argument(
        "--confidence-analysis",
        action="store_true",
        help="Analyze prediction confidence",
    )
    parser.add_argument(
        "--error-analysis", action="store_true", help="Perform detailed error analysis"
    )

    # Robustness evaluation
    parser.add_argument(
        "--robustness-tests",
        type=str,
        nargs="*",
        default=[],
        choices=["noise", "blur", "brightness", "contrast", "adversarial"],
        help="Robustness tests to perform",
    )
    parser.add_argument(
        "--corruption-severity",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="Corruption severity levels",
    )

    # Configuration overrides
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration values (format: key=value)",
    )

    # Debug and development
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--limit-samples", type=int, help="Limit number of samples for testing"
    )

    return parser.parse_args()


def load_model_and_config(
    model_path: str, config_path: Optional[str] = None, overrides: List[str] = None
) -> tuple:
    """Load model and configuration from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Try to get config from checkpoint
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        print("Using configuration from checkpoint")
    elif config_path:
        config_dict = load_hierarchical_config(
            base_config=config_path, overrides=overrides or []
        )
        print(f"Using configuration from: {config_path}")
    else:
        raise ValueError(
            "No configuration found in checkpoint and no config file provided"
        )

    # Create experiment config
    config = ExperimentConfig(**config_dict)

    # Create model
    model = create_model(config.model)

    # Load model state
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel/DistributedDataParallel prefixes
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # Get additional checkpoint info
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "best_metric": checkpoint.get("best_metric", "unknown"),
        "training_time": checkpoint.get("training_time", "unknown"),
    }

    print(f"Model loaded successfully:")
    print(f"  - Epoch: {checkpoint_info['epoch']}")
    print(f"  - Best metric: {checkpoint_info['best_metric']}")

    return model, config, checkpoint_info


def create_evaluation_dataset(
    data_path: str,
    annotation_format: str,
    config: ExperimentConfig,
    limit_samples: Optional[int] = None,
) -> tuple:
    """Create evaluation dataset and dataloader."""
    print(f"Creating evaluation dataset from: {data_path}")

    # Create transforms
    transform_manager = create_transforms(
        domain=config.augmentation.get("domain", "natural"),
        image_size=config.data.image_size,
        train_kwargs={},
        val_kwargs={},
    )

    # Create dataset
    dataset = create_dataset(
        data_path=data_path,
        annotation_format=annotation_format,
        transform=transform_manager.get_val_transform(),
        cache_images=False,
    )

    # Limit samples if requested
    if limit_samples and limit_samples < len(dataset):
        indices = np.random.choice(len(dataset), limit_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Limited dataset to {limit_samples} samples")

    print(f"Evaluation dataset: {len(dataset)} samples")
    if hasattr(dataset, "classes"):
        print(f"Number of classes: {len(dataset.classes)}")

    return dataset, transform_manager


def run_standard_evaluation(
    model: torch.nn.Module,
    dataset,
    config: ExperimentConfig,
    args: argparse.Namespace,
    device_manager,
) -> Dict[str, Any]:
    """Run standard evaluation on the dataset."""
    print("Running standard evaluation...")

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device_manager.device.type == "cuda",
        drop_last=False,
    )

    # Create evaluator
    evaluator = create_evaluator(
        task_type=config.model.task_type,
        num_classes=config.model.num_classes,
        device=device_manager.device,
    )

    # Set model to evaluation mode
    model.eval()
    model = model.to(device_manager.device)

    # Storage for predictions and features
    all_predictions = []
    all_targets = []
    all_features = []
    all_confidences = []

    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch

            inputs = inputs.to(device_manager.device)
            targets = targets.to(device_manager.device)

            # Forward pass
            outputs = model(inputs)

            # Get predictions and features
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                features = outputs.get("features", None)
            else:
                logits = outputs
                features = None

            predictions = torch.argmax(logits, dim=1)
            confidences = F.softmax(logits, dim=1).max(dim=1)[0]

            # Store results
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_confidences.append(confidences.cpu())

            if features is not None and args.save_features:
                all_features.append(features.cpu())

            # Update evaluator
            evaluator.update(logits.cpu(), targets.cpu())

    # Compute metrics
    metrics = evaluator.compute()

    # Concatenate all results
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    all_confidences = torch.cat(all_confidences)

    if all_features:
        all_features = torch.cat(all_features)

    # Additional analysis
    results = {
        "metrics": metrics,
        "predictions": all_predictions.numpy() if args.save_predictions else None,
        "targets": all_targets.numpy(),
        "confidences": all_confidences.numpy(),
        "features": all_features.numpy() if all_features else None,
    }

    # Per-class metrics
    if args.per_class_metrics:
        results["per_class_metrics"] = evaluator.compute_per_class_metrics()

    # Confidence analysis
    if args.confidence_analysis:
        results["confidence_analysis"] = analyze_confidence(
            all_confidences.numpy(), all_predictions.numpy(), all_targets.numpy()
        )

    return results


def analyze_confidence(
    confidences: np.ndarray, predictions: np.ndarray, targets: np.ndarray
) -> Dict:
    """Analyze prediction confidence statistics."""
    correct = predictions == targets

    return {
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_correct": float(np.mean(confidences[correct])),
        "mean_confidence_incorrect": float(np.mean(confidences[~correct])),
        "confidence_accuracy_correlation": float(
            np.corrcoef(confidences, correct)[0, 1]
        ),
        "high_confidence_accuracy": float(np.mean(correct[confidences > 0.9]))
        if np.any(confidences > 0.9)
        else 0.0,
        "low_confidence_accuracy": float(np.mean(correct[confidences < 0.5]))
        if np.any(confidences < 0.5)
        else 0.0,
    }


def run_benchmark_evaluation(
    model: torch.nn.Module,
    config: ExperimentConfig,
    benchmark_names: List[str],
    args: argparse.Namespace,
    device_manager,
) -> Dict[str, Any]:
    """Run benchmark evaluation on standard datasets."""
    print(f"Running benchmark evaluation on: {benchmark_names}")

    benchmark_runner = BenchmarkRunner(
        model=model,
        device=device_manager.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = {}
    for benchmark_name in benchmark_names:
        print(f"Evaluating on {benchmark_name}...")
        benchmark_results = benchmark_runner.run_benchmark(
            benchmark_name=benchmark_name, model_config=config.model
        )
        results[benchmark_name] = benchmark_results

    return results


def run_robustness_evaluation(
    model: torch.nn.Module,
    dataset,
    robustness_tests: List[str],
    corruption_severity: List[int],
    args: argparse.Namespace,
    device_manager,
) -> Dict[str, Any]:
    """Run robustness evaluation with various corruptions."""
    print(f"Running robustness evaluation: {robustness_tests}")

    robustness_evaluator = RobustnessEvaluator(
        model=model,
        device=device_manager.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = {}
    for test_type in robustness_tests:
        print(f"Running {test_type} robustness test...")
        test_results = robustness_evaluator.evaluate_robustness(
            dataset=dataset,
            corruption_type=test_type,
            severity_levels=corruption_severity,
        )
        results[test_type] = test_results

    return results


def generate_visualizations(
    model: torch.nn.Module,
    dataset,
    results: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device_manager,
):
    """Generate evaluation visualizations."""
    print("Generating visualizations...")

    visualizer = create_visualizer(model=model, device=device_manager.device)

    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Sample data for visualization
    dataloader = DataLoader(
        dataset,
        batch_size=min(args.num_vis_samples, args.batch_size),
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Get sample batch
    sample_batch = next(iter(dataloader))
    if len(sample_batch) == 2:
        sample_inputs, sample_targets = sample_batch
    else:
        sample_inputs, sample_targets, _ = sample_batch

    # Generate confusion matrix
    if "metrics" in results and "confusion_matrix" in results["metrics"]:
        visualizer.plot_confusion_matrix(
            results["metrics"]["confusion_matrix"],
            class_names=getattr(dataset, "classes", None),
            save_path=viz_dir / "confusion_matrix.png",
        )

    # Generate ROC curves
    if "targets" in results and "confidences" in results:
        visualizer.plot_roc_curves(
            results["targets"],
            results["confidences"],
            save_path=viz_dir / "roc_curves.png",
        )

    # Generate attention maps if requested
    if args.attention_maps and hasattr(model, "get_attention_maps"):
        attention_maps = visualizer.generate_attention_maps(
            sample_inputs[: min(10, len(sample_inputs))],
            save_dir=viz_dir / "attention_maps",
        )


def export_results(
    results: Dict[str, Any],
    export_formats: List[str],
    output_dir: Path,
    experiment_name: str,
):
    """Export evaluation results in specified formats."""
    print(f"Exporting results in formats: {export_formats}")

    for format_type in export_formats:
        if format_type == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif isinstance(value, dict):
                    json_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value

            with open(output_dir / f"{experiment_name}_results.json", "w") as f:
                json.dump(json_results, f, indent=2, default=str)

        elif format_type == "csv" and "metrics" in results:
            import pandas as pd

            metrics_df = pd.DataFrame([results["metrics"]])
            metrics_df.to_csv(
                output_dir / f"{experiment_name}_metrics.csv", index=False
            )

        # TODO: Implement HTML and PDF export
        elif format_type in ["html", "pdf"]:
            print(f"Export format '{format_type}' not yet implemented")


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(
        name="dinov3_evaluation",
        level=log_level,
        log_dir=Path(args.output_dir) / "logs",
    )

    try:
        # Setup device
        device_manager = get_device_manager()
        if args.device != "auto":
            device_manager.set_device(args.device)

        print(f"Using device: {device_manager.device}")

        # Load model and configuration
        model, config, checkpoint_info = load_model_and_config(
            args.model_path, args.config, args.override
        )

        # Set random seed
        set_seed(config.seed, strict=True)

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store checkpoint info
        with open(output_dir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2, default=str)

        experiment_name = f"eval_{config.name}_{args.mode}"

        # Initialize results dictionary
        all_results = {
            "experiment_name": experiment_name,
            "model_path": args.model_path,
            "config": OmegaConf.to_container(config, resolve=True),
            "checkpoint_info": checkpoint_info,
            "evaluation_args": vars(args),
        }

        # Run evaluation based on mode
        if args.mode == "standard":
            # Create evaluation dataset
            dataset, transform_manager = create_evaluation_dataset(
                args.data_path, args.annotation_format, config, args.limit_samples
            )

            # Run standard evaluation
            eval_results = run_standard_evaluation(
                model, dataset, config, args, device_manager
            )
            all_results.update(eval_results)

            # Generate visualizations if requested
            if args.visualize:
                generate_visualizations(
                    model, dataset, all_results, args, output_dir, device_manager
                )

        elif args.mode == "benchmark":
            if not args.benchmark:
                raise ValueError("Benchmark mode requires --benchmark argument")

            benchmark_results = run_benchmark_evaluation(
                model, config, args.benchmark, args, device_manager
            )
            all_results["benchmark_results"] = benchmark_results

        elif args.mode == "robustness":
            if not args.robustness_tests:
                raise ValueError("Robustness mode requires --robustness-tests argument")

            dataset, _ = create_evaluation_dataset(
                args.data_path, args.annotation_format, config, args.limit_samples
            )

            robustness_results = run_robustness_evaluation(
                model,
                dataset,
                args.robustness_tests,
                args.corruption_severity,
                args,
                device_manager,
            )
            all_results["robustness_results"] = robustness_results

        # Export results
        export_results(all_results, args.export_format, output_dir, experiment_name)

        # Print summary
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

        if "metrics" in all_results:
            metrics = all_results["metrics"]
            logger.info("Key metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

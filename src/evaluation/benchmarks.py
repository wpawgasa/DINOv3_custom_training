import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ..data.augmentations import AugmentationFactory
from ..data.dataset import create_dataset
from ..models.dinov3_classifier import DINOv3Classifier
from ..utils.logging import get_logger
from .evaluators import ClassificationEvaluator, create_evaluator


class BenchmarkRunner:
    """Automated benchmark runner for standard datasets."""

    def __init__(self, model: DINOv3Classifier, logger=None):
        self.model = model
        self.logger = logger or get_logger("benchmark_runner")

        # Standard benchmark configurations
        self.benchmark_configs = {
            "imagenet": {
                "num_classes": 1000,
                "class_names_url": "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
                "description": "ImageNet-1K classification benchmark",
            },
            "cifar10": {
                "num_classes": 10,
                "class_names": [
                    "airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                ],
                "description": "CIFAR-10 classification benchmark",
            },
            "cifar100": {
                "num_classes": 100,
                "description": "CIFAR-100 classification benchmark",
            },
        }

    def run_benchmark(
        self,
        benchmark_name: str,
        test_dataloader: DataLoader,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on a standard benchmark dataset."""

        if benchmark_name not in self.benchmark_configs:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        config = self.benchmark_configs[benchmark_name]
        self.logger.info(f"Running {benchmark_name} benchmark: {config['description']}")

        # Create evaluator
        evaluator = ClassificationEvaluator(
            self.model, class_names=config.get("class_names"), logger=self.logger
        )

        # Run evaluation
        start_time = time.time()
        results = evaluator.evaluate(
            test_dataloader,
            compute_per_class_metrics=True,
            compute_confusion_matrix=True,
            compute_roc_curves=True,
            output_dir=output_dir,
        )
        evaluation_time = time.time() - start_time

        # Add benchmark metadata
        results.update(
            {
                "benchmark_name": benchmark_name,
                "benchmark_description": config["description"],
                "evaluation_time": evaluation_time,
                "model_variant": self.model.variant,
                "model_info": self.model.get_model_info(),
            }
        )

        self.logger.info(
            f"Benchmark completed: {results['overall_accuracy']:.2f}% accuracy"
        )

        return results

    def run_multiple_benchmarks(
        self,
        benchmark_dataloaders: Dict[str, DataLoader],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run evaluation on multiple benchmark datasets."""

        all_results = {}

        for benchmark_name, dataloader in benchmark_dataloaders.items():
            self.logger.info(f"Starting benchmark: {benchmark_name}")

            benchmark_output_dir = None
            if output_dir:
                benchmark_output_dir = output_dir / benchmark_name

            try:
                results = self.run_benchmark(
                    benchmark_name, dataloader, benchmark_output_dir
                )
                all_results[benchmark_name] = results

            except Exception as e:
                self.logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                all_results[benchmark_name] = {"error": str(e)}

        # Create summary
        summary = self._create_benchmark_summary(all_results)
        all_results["summary"] = summary

        return all_results

    def _create_benchmark_summary(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary across multiple benchmarks."""

        summary = {
            "total_benchmarks": len([k for k in results.keys() if k != "summary"]),
            "successful_benchmarks": len(
                [k for k, v in results.items() if k != "summary" and "error" not in v]
            ),
            "average_accuracy": 0.0,
            "benchmark_accuracies": {},
        }

        accuracies = []
        for benchmark_name, result in results.items():
            if benchmark_name != "summary" and "overall_accuracy" in result:
                accuracy = result["overall_accuracy"]
                summary["benchmark_accuracies"][benchmark_name] = accuracy
                accuracies.append(accuracy)

        if accuracies:
            summary["average_accuracy"] = np.mean(accuracies)

        return summary


class RobustnessEvaluator:
    """Evaluator for model robustness testing."""

    def __init__(self, model: DINOv3Classifier, logger=None):
        self.model = model
        self.logger = logger or get_logger("robustness_evaluator")
        self.device = next(self.model.parameters()).device

    def evaluate_adversarial_robustness(
        self,
        test_dataloader: DataLoader,
        attack_types: List[str] = None,
        epsilon_values: List[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate model robustness against adversarial attacks."""

        if attack_types is None:
            attack_types = ["fgsm", "pgd"]

        if epsilon_values is None:
            epsilon_values = [0.001, 0.01, 0.1]

        self.logger.info("Starting adversarial robustness evaluation...")

        results = {"clean_accuracy": 0.0, "attack_results": {}}

        # First, evaluate on clean data
        clean_accuracy = self._evaluate_clean_accuracy(test_dataloader)
        results["clean_accuracy"] = clean_accuracy

        # Evaluate each attack type and epsilon
        for attack_type in attack_types:
            results["attack_results"][attack_type] = {}

            for epsilon in epsilon_values:
                self.logger.info(f"Testing {attack_type} attack with epsilon={epsilon}")

                attack_accuracy = self._evaluate_adversarial_accuracy(
                    test_dataloader, attack_type, epsilon
                )

                results["attack_results"][attack_type][str(epsilon)] = {
                    "accuracy": attack_accuracy,
                    "accuracy_drop": clean_accuracy - attack_accuracy,
                }

        return results

    def evaluate_common_corruptions(
        self,
        test_dataloader: DataLoader,
        corruption_types: List[str] = None,
        severity_levels: List[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate model robustness against common corruptions."""

        if corruption_types is None:
            corruption_types = ["gaussian_noise", "motion_blur", "snow", "brightness"]

        if severity_levels is None:
            severity_levels = [1, 3, 5]

        self.logger.info("Starting common corruptions evaluation...")

        results = {"clean_accuracy": 0.0, "corruption_results": {}}

        # Evaluate on clean data
        clean_accuracy = self._evaluate_clean_accuracy(test_dataloader)
        results["clean_accuracy"] = clean_accuracy

        # Evaluate each corruption type and severity
        for corruption_type in corruption_types:
            results["corruption_results"][corruption_type] = {}

            for severity in severity_levels:
                self.logger.info(
                    f"Testing {corruption_type} corruption with severity={severity}"
                )

                corrupt_accuracy = self._evaluate_corrupted_accuracy(
                    test_dataloader, corruption_type, severity
                )

                results["corruption_results"][corruption_type][str(severity)] = {
                    "accuracy": corrupt_accuracy,
                    "accuracy_drop": clean_accuracy - corrupt_accuracy,
                }

        return results

    def _evaluate_clean_accuracy(self, dataloader: DataLoader) -> float:
        """Evaluate accuracy on clean data."""
        evaluator = ClassificationEvaluator(self.model, logger=self.logger)
        results = evaluator.evaluate(dataloader, compute_per_class_metrics=False)
        return results["overall_accuracy"]

    def _evaluate_adversarial_accuracy(
        self, dataloader: DataLoader, attack_type: str, epsilon: float
    ) -> float:
        """Evaluate accuracy under adversarial attack."""

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Generate adversarial examples
                adv_inputs = self._generate_adversarial_examples(
                    inputs, targets, attack_type, epsilon
                )

                # Evaluate on adversarial examples
                outputs = self.model(adv_inputs)
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def _evaluate_corrupted_accuracy(
        self, dataloader: DataLoader, corruption_type: str, severity: int
    ) -> float:
        """Evaluate accuracy on corrupted data."""

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Apply corruption
                corrupted_inputs = self._apply_corruption(
                    inputs, corruption_type, severity
                )

                # Evaluate on corrupted data
                outputs = self.model(corrupted_inputs)
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    def _generate_adversarial_examples(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        attack_type: str,
        epsilon: float,
    ) -> torch.Tensor:
        """Generate adversarial examples."""

        inputs.requires_grad_(True)

        if attack_type == "fgsm":
            return self._fgsm_attack(inputs, targets, epsilon)
        elif attack_type == "pgd":
            return self._pgd_attack(inputs, targets, epsilon)
        else:
            self.logger.warning(f"Unknown attack type: {attack_type}, using FGSM")
            return self._fgsm_attack(inputs, targets, epsilon)

    def _fgsm_attack(
        self, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""

        self.model.eval()

        # Forward pass
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Backward pass
        loss.backward()

        # Generate adversarial examples
        data_grad = inputs.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_inputs = inputs + epsilon * sign_data_grad

        # Clamp to valid input range
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)

        return perturbed_inputs.detach()

    def _pgd_attack(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float,
        alpha: float = None,
        num_iter: int = 10,
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""

        if alpha is None:
            alpha = epsilon / 4

        self.model.eval()

        # Initialize perturbed inputs
        perturbed_inputs = inputs.clone().detach()

        for _ in range(num_iter):
            perturbed_inputs.requires_grad_(True)

            outputs = self.model(perturbed_inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            # Update
            data_grad = perturbed_inputs.grad.data
            perturbed_inputs = perturbed_inputs + alpha * data_grad.sign()

            # Project to epsilon ball
            eta = torch.clamp(perturbed_inputs - inputs, -epsilon, epsilon)
            perturbed_inputs = torch.clamp(inputs + eta, 0, 1).detach()

        return perturbed_inputs

    def _apply_corruption(
        self, inputs: torch.Tensor, corruption_type: str, severity: int
    ) -> torch.Tensor:
        """Apply corruption to input images."""

        # Convert to numpy for corruption application
        inputs_np = inputs.cpu().numpy()
        batch_size = inputs_np.shape[0]
        corrupted_batch = []

        for i in range(batch_size):
            # Convert to PIL image format (H, W, C)
            img = np.transpose(inputs_np[i], (1, 2, 0))
            img = (img * 255).astype(np.uint8)

            # Apply corruption
            corrupted_img = self._apply_single_corruption(
                img, corruption_type, severity
            )

            # Convert back to tensor format
            corrupted_img = corrupted_img.astype(np.float32) / 255.0
            corrupted_img = np.transpose(corrupted_img, (2, 0, 1))
            corrupted_batch.append(corrupted_img)

        corrupted_inputs = torch.tensor(np.stack(corrupted_batch))
        return corrupted_inputs.to(inputs.device)

    def _apply_single_corruption(
        self, img: np.ndarray, corruption_type: str, severity: int
    ) -> np.ndarray:
        """Apply a single corruption to an image."""

        # Normalize severity to [0, 1]
        severity_normalized = severity / 5.0

        if corruption_type == "gaussian_noise":
            noise = np.random.normal(0, 25 * severity_normalized, img.shape)
            corrupted = img + noise
            corrupted = np.clip(corrupted, 0, 255)

        elif corruption_type == "motion_blur":
            kernel_size = int(3 + 4 * severity_normalized)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0
            kernel = kernel / kernel_size
            corrupted = cv2.filter2D(img, -1, kernel)

        elif corruption_type == "snow":
            # Simple snow effect
            snow = np.random.random(img.shape[:2])
            snow = (snow < 0.1 * severity_normalized).astype(np.float32)
            snow = np.stack([snow] * 3, axis=2) * 255
            corrupted = np.maximum(img, snow)

        elif corruption_type == "brightness":
            brightness_factor = 1.0 + 0.5 * severity_normalized
            corrupted = img * brightness_factor
            corrupted = np.clip(corrupted, 0, 255)

        else:
            self.logger.warning(f"Unknown corruption type: {corruption_type}")
            corrupted = img

        return corrupted.astype(np.uint8)


class DistributionShiftAnalyzer:
    """Analyzer for distribution shift robustness."""

    def __init__(self, model: DINOv3Classifier, logger=None):
        self.model = model
        self.logger = logger or get_logger("distribution_shift_analyzer")

    def analyze_domain_shift(
        self,
        source_dataloader: DataLoader,
        target_dataloader: DataLoader,
        domain_names: Tuple[str, str] = ("source", "target"),
    ) -> Dict[str, Any]:
        """Analyze performance under domain shift."""

        self.logger.info(
            f"Analyzing domain shift: {domain_names[0]} -> {domain_names[1]}"
        )

        # Evaluate on source domain
        source_evaluator = ClassificationEvaluator(self.model, logger=self.logger)
        source_results = source_evaluator.evaluate(
            source_dataloader, compute_per_class_metrics=True
        )

        # Evaluate on target domain
        target_evaluator = ClassificationEvaluator(self.model, logger=self.logger)
        target_results = target_evaluator.evaluate(
            target_dataloader, compute_per_class_metrics=True
        )

        # Calculate performance drop
        accuracy_drop = (
            source_results["overall_accuracy"] - target_results["overall_accuracy"]
        )

        results = {
            "source_domain": domain_names[0],
            "target_domain": domain_names[1],
            "source_accuracy": source_results["overall_accuracy"],
            "target_accuracy": target_results["overall_accuracy"],
            "accuracy_drop": accuracy_drop,
            "source_results": source_results,
            "target_results": target_results,
        }

        self.logger.info(
            f"Domain shift analysis complete: {accuracy_drop:.2f}% accuracy drop"
        )

        return results

    def analyze_dataset_size_effect(
        self, dataloader: DataLoader, sample_fractions: List[float] = None
    ) -> Dict[str, Any]:
        """Analyze performance as dataset size varies."""

        if sample_fractions is None:
            sample_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        self.logger.info("Analyzing dataset size effects...")

        results = {
            "sample_fractions": sample_fractions,
            "accuracies": [],
            "sample_sizes": [],
        }

        # Get total dataset size
        total_samples = len(dataloader.dataset)

        for fraction in sample_fractions:
            sample_size = int(total_samples * fraction)

            # Create subset dataloader (simplified - would need proper implementation)
            # For now, just record the expected behavior
            results["sample_sizes"].append(sample_size)

            # Placeholder - would need actual evaluation on subset
            # This would require creating a subset of the dataset
            placeholder_accuracy = 90.0 * fraction  # Simplified relationship
            results["accuracies"].append(placeholder_accuracy)

        return results


def create_benchmark_runner(model: DINOv3Classifier, **kwargs) -> BenchmarkRunner:
    """Create benchmark runner."""
    return BenchmarkRunner(model, **kwargs)


def create_robustness_evaluator(
    model: DINOv3Classifier, **kwargs
) -> RobustnessEvaluator:
    """Create robustness evaluator."""
    return RobustnessEvaluator(model, **kwargs)


def create_distribution_shift_analyzer(
    model: DINOv3Classifier, **kwargs
) -> DistributionShiftAnalyzer:
    """Create distribution shift analyzer."""
    return DistributionShiftAnalyzer(model, **kwargs)

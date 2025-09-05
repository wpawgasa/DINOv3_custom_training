#!/usr/bin/env python3
"""
DINOv3 Training Script

Main training entry point for DINOv3 models with comprehensive configuration support,
distributed training, and experiment tracking.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.augmentations import create_transforms
from data.dataset import create_dataset
from models.model_factory import create_model, get_model_factory
from training.experiment_tracking import create_experiment_manager
from training.trainer import DINOv3Trainer
from utils.config import load_hierarchical_config
from utils.device import (
    cleanup_distributed,
    get_device_manager,
    setup_distributed_training,
)
from utils.logging import setup_logging
from utils.reproducibility import set_seed
from utils.schemas import ExperimentConfig


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINOv3 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration files
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to main configuration file",
    )
    parser.add_argument(
        "--model-config", type=str, help="Path to model configuration file"
    )
    parser.add_argument(
        "--training-config", type=str, help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-config", type=str, help="Path to data configuration file"
    )

    # Output and logging
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./outputs",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment (default: auto-generated)",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    # Distributed training
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )

    # Experiment tracking
    parser.add_argument(
        "--use-wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb-project", type=str, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, help="W&B entity name")
    parser.add_argument(
        "--use-mlflow", action="store_true", help="Enable MLflow logging"
    )
    parser.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")

    # Configuration overrides
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override configuration values (format: key=value)",
    )

    # Debug and development
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run (validate configuration without training)",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")

    return parser.parse_args()


def setup_distributed_training_env(args: argparse.Namespace) -> tuple:
    """Setup distributed training environment."""
    local_rank = 0
    world_size = 1
    rank = 0

    if args.distributed or "LOCAL_RANK" in os.environ:
        try:
            local_rank, world_size, rank = setup_distributed_training()
            print(
                f"Distributed training: rank {rank}, local_rank {local_rank}, world_size {world_size}"
            )
        except Exception as e:
            print(f"Failed to setup distributed training: {e}")
            print("Falling back to single GPU training")
            args.distributed = False

    return local_rank, world_size, rank


def load_and_validate_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load and validate configuration."""
    print("Loading configuration...")

    # Load hierarchical configuration
    config = load_hierarchical_config(
        base_config=args.config,
        model_config=args.model_config,
        training_config=args.training_config,
        dataset_config=args.data_config,
        overrides=args.override,
    )

    # Convert to ExperimentConfig for validation
    try:
        experiment_config = ExperimentConfig(**config)
        print(f"Configuration loaded successfully: {experiment_config.name}")
        return experiment_config
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        raise


def create_datasets_and_loaders(config: ExperimentConfig, transform_manager) -> tuple:
    """Create datasets and data loaders."""
    print("Creating datasets...")

    # Training dataset
    train_dataset = create_dataset(
        data_path=config.data.train_data_path,
        annotation_format="imagefolder",  # TODO: Make configurable
        transform=transform_manager.get_train_transform(),
        cache_images=False,  # TODO: Make configurable
    )

    # Validation dataset (optional)
    val_dataset = None
    if config.data.val_data_path:
        val_dataset = create_dataset(
            data_path=config.data.val_data_path,
            annotation_format="imagefolder",
            transform=transform_manager.get_val_transform(),
            cache_images=False,
        )

    print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,  # For distributed training
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=False,
        )

    return train_dataset, val_dataset, train_loader, val_loader


def create_model_from_config(config: ExperimentConfig) -> torch.nn.Module:
    """Create model from configuration."""
    print(f"Creating model: {config.model.variant}")

    model = create_model(config.model)

    # Print model information
    model_info = model.get_model_info()
    print(f"Model created: {model_info['variant']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.1f} MB")

    return model


def setup_experiment_tracking(
    args: argparse.Namespace, config: ExperimentConfig
) -> Optional[Any]:
    """Setup experiment tracking."""

    experiment_name = args.experiment_name or config.name
    output_dir = Path(args.output_dir) / experiment_name

    # Create experiment manager
    experiment_manager = create_experiment_manager(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=OmegaConf.to_container(config, resolve=True),
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow,
        use_tensorboard=config.logging.use_tensorboard,
        wandb_project=args.wandb_project or config.logging.wandb_project,
        wandb_entity=args.wandb_entity or config.logging.wandb_entity,
        mlflow_tracking_uri=args.mlflow_uri or config.logging.mlflow_tracking_uri,
    )

    print(f"Experiment tracking setup: {experiment_name}")
    return experiment_manager


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(
        name="dinov3_training",
        level=log_level,
        log_dir=Path(args.output_dir) / "logs" if args.output_dir else None,
    )

    try:
        # Setup distributed training
        local_rank, world_size, rank = setup_distributed_training_env(args)

        # Load and validate configuration
        config = load_and_validate_config(args)

        # Set random seed for reproducibility
        set_seed(config.seed, strict=True)

        # Create experiment tracking
        experiment_manager = None
        if rank == 0:  # Only setup tracking on main process
            experiment_manager = setup_experiment_tracking(args, config)

        # Create transforms
        transform_manager = create_transforms(
            domain=config.augmentation.get("domain", "natural"),
            image_size=config.data.image_size,
            train_kwargs={},
            val_kwargs={},
        )

        # Create datasets and data loaders
        (
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
        ) = create_datasets_and_loaders(config, transform_manager)

        # Create model
        model = create_model_from_config(config)

        # Set training mode based on config
        model.set_training_mode(config.training.mode)

        # Setup distributed model if needed
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully!")
            return

        # Create trainer
        trainer = DINOv3Trainer(
            model=model,
            config=config.training,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            logger=logger,
            experiment_tracker=experiment_manager,
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training
        output_dir = Path(args.output_dir) / (args.experiment_name or config.name)

        if args.profile:
            # TODO: Add profiling support
            logger.warning("Profiling not yet implemented")

        logger.info("Starting training...")
        training_results = trainer.train(output_dir)

        # Log final results
        logger.info("Training completed!")
        logger.info(f"Training time: {training_results['training_time']:.2f}s")
        logger.info(f"Final epoch: {training_results['final_epoch']}")
        if training_results["best_metric"]:
            logger.info(f"Best metric: {training_results['best_metric']:.4f}")

        # Finish experiment tracking
        if experiment_manager:
            experiment_manager.finish()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up distributed training
        if args.distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()

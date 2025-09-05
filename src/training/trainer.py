import math
import os
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..models.dinov3_classifier import DINOv3Classifier
from ..utils.device import get_device_manager
from ..utils.logging import get_logger, with_context
from ..utils.reproducibility import ReproducibleTraining
from ..utils.schemas import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingMode,
)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None

        self.monitor_op = torch.lt if mode == "min" else torch.gt
        self.min_delta *= 1 if mode == "min" else -1

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(score, self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False


class MetricsTracker:
    """Utility for tracking and smoothing training metrics."""

    def __init__(self, smoothing_window: int = 100):
        self.smoothing_window = smoothing_window
        self.metrics = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.epoch_metrics = defaultdict(list)

    def update(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        for name, value in metrics.items():
            self.metrics[name].append(value)
            if epoch is not None:
                if len(self.epoch_metrics[name]) <= epoch:
                    self.epoch_metrics[name].extend(
                        [None] * (epoch + 1 - len(self.epoch_metrics[name]))
                    )
                self.epoch_metrics[name][epoch] = value

    def get_smooth_value(self, metric_name: str) -> Optional[float]:
        if metric_name in self.metrics and self.metrics[metric_name]:
            return np.mean(self.metrics[metric_name])
        return None

    def get_latest_value(self, metric_name: str) -> Optional[float]:
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None

    def get_epoch_metrics(self, epoch: int) -> Dict[str, float]:
        metrics = {}
        for name, values in self.epoch_metrics.items():
            if epoch < len(values) and values[epoch] is not None:
                metrics[name] = values[epoch]
        return metrics


class OptimizerFactory:
    """Factory for creating optimizers with different configurations."""

    @staticmethod
    def create_optimizer(
        model: nn.Module,
        config: OptimizerConfig,
        use_parameter_groups: bool = False,
        backbone_lr: Optional[float] = None,
        head_lr: Optional[float] = None,
    ) -> optim.Optimizer:
        if use_parameter_groups and hasattr(model, "get_parameter_groups"):
            if backbone_lr is None:
                backbone_lr = config.backbone_lr or config.learning_rate
            if head_lr is None:
                head_lr = config.head_lr or config.learning_rate

            param_groups = model.get_parameter_groups(backbone_lr, head_lr)
        else:
            param_groups = model.parameters()
            if backbone_lr is not None or head_lr is not None:
                warnings.warn(
                    "backbone_lr/head_lr specified but model doesn't support parameter groups"
                )

        if config.type == "adamw":
            return optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        elif config.type == "sgd":
            return optim.SGD(
                param_groups,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        elif config.type == "adam":
            return optim.Adam(
                param_groups,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                eps=config.eps,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {config.type}")


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""

    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        config: SchedulerConfig,
        steps_per_epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        if config is None:
            return None

        if config.type == "cosine":
            if config.max_steps:
                T_max = config.max_steps
            elif total_epochs and steps_per_epoch:
                T_max = total_epochs * steps_per_epoch
            else:
                T_max = 100  # Default fallback

            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=config.min_lr
            )

        elif config.type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.gamma
            )

        elif config.type == "exponential":
            return optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)

        elif config.type == "warmup_cosine":
            # Custom warmup cosine scheduler
            return WarmupCosineScheduler(
                optimizer,
                warmup_steps=config.warmup_steps or 0,
                total_steps=config.max_steps
                or (
                    total_epochs * steps_per_epoch
                    if total_epochs and steps_per_epoch
                    else 1000
                ),
                min_lr=config.min_lr,
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {config.type}")


class WarmupCosineScheduler:
    """Custom warmup + cosine annealing scheduler."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.step_count / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count <= self.warmup_steps:
                param_group["lr"] = base_lr * lr_scale
            else:
                param_group["lr"] = self.min_lr + (base_lr - self.min_lr) * lr_scale

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.base_lrs = state_dict["base_lrs"]


class DINOv3Trainer:
    """Comprehensive training framework for DINOv3 models."""

    def __init__(
        self,
        model: DINOv3Classifier,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        logger=None,
        experiment_tracker=None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger or get_logger("dinov3_trainer")
        self.experiment_tracker = experiment_tracker

        # Initialize device management
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        self.model = self.model.to(self.device)

        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.is_best_epoch = False

        # Setup reproducibility
        self.repro_training = ReproducibleTraining()

        # Initialize training components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_mixed_precision()
        self._setup_early_stopping()

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()

        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        self.logger.info("DINOv3Trainer initialized successfully")
        self.logger.info(f"Training mode: {self.config.mode}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.mixed_precision}")

    def _setup_optimizer(self):
        """Setup optimizer with differential learning rates if needed."""
        use_differential_lr = (
            self.config.mode == TrainingMode.FULL_FINE_TUNE
            and self.config.optimizer.backbone_lr is not None
            and self.config.optimizer.head_lr is not None
        )

        self.optimizer = OptimizerFactory.create_optimizer(
            self.model,
            self.config.optimizer,
            use_parameter_groups=use_differential_lr,
            backbone_lr=self.config.optimizer.backbone_lr,
            head_lr=self.config.optimizer.head_lr,
        )

        if use_differential_lr:
            self.logger.info(
                f"Using differential learning rates: "
                f"backbone={self.config.optimizer.backbone_lr}, "
                f"head={self.config.optimizer.head_lr}"
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        steps_per_epoch = len(self.train_dataloader)

        self.scheduler = SchedulerFactory.create_scheduler(
            self.optimizer,
            self.config.scheduler,
            steps_per_epoch=steps_per_epoch,
            total_epochs=self.config.max_epochs,
        )

    def _setup_loss_function(self):
        """Setup loss function based on task type."""
        if self.model.task_type == "classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.model.task_type == "segmentation":
            self.criterion = (
                nn.CrossEntropyLoss()
            )  # Can be extended for other seg losses
        else:
            self.criterion = nn.CrossEntropyLoss()  # Default fallback

    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.use_amp = self.config.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")

    def _setup_early_stopping(self):
        """Setup early stopping."""
        self.early_stopping = None
        if self.config.early_stopping_patience:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode="min",  # Assuming we want to minimize validation loss
            )
            self.logger.info(
                f"Early stopping enabled with patience {self.config.early_stopping_patience}"
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Set epoch seed for reproducibility
        self.repro_training.set_epoch_seed(self.current_epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = len(self.train_dataloader)

        # Progress tracking
        log_interval = max(1, num_batches // 10)

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            batch_start_time = time.time()

            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.max_grad_norm:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Scheduler step (if step-based)
                if (
                    self.scheduler
                    and hasattr(self.scheduler, "step")
                    and not hasattr(self.scheduler, "step_size")
                ):
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Metrics calculation
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            if self.model.task_type == "classification":
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)

            # Logging
            if batch_idx % log_interval == 0:
                batch_time = time.time() - batch_start_time
                current_lr = self.optimizer.param_groups[0]["lr"]

                self.logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}, "
                    f"Time: {batch_time:.2f}s"
                )

                # Track metrics
                batch_metrics = {
                    "batch_loss": loss.item(),
                    "learning_rate": current_lr,
                    "batch_time": batch_time,
                }
                self.metrics_tracker.update(batch_metrics)

                # Log to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(
                        batch_metrics, step=self.global_step
                    )

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        train_metrics = {"train_loss": avg_loss}

        if self.model.task_type == "classification" and total_samples > 0:
            train_acc = 100.0 * total_correct / total_samples
            train_metrics["train_acc"] = train_acc

        # Step scheduler (if epoch-based)
        if self.scheduler and hasattr(self.scheduler, "step_size"):
            self.scheduler.step()

        return train_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                if self.model.task_type == "classification":
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()
                    total_samples += targets.size(0)

        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_dataloader)
        val_metrics = {"val_loss": avg_loss}

        if self.model.task_type == "classification" and total_samples > 0:
            val_acc = 100.0 * total_correct / total_samples
            val_metrics["val_acc"] = val_acc

        return val_metrics

    def train(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """Main training loop."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Starting training for {self.config.max_epochs} epochs")
        self.logger.info(f"Output directory: {output_dir}")

        training_start_time = time.time()

        try:
            for epoch in range(self.config.max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                # Training phase
                train_metrics = self.train_epoch()

                # Validation phase
                val_metrics = {}
                if self.current_epoch % self.config.validation_frequency == 0:
                    val_metrics = self.validate_epoch()

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}
                epoch_time = time.time() - epoch_start_time
                epoch_metrics["epoch_time"] = epoch_time

                # Update training history
                for key, value in epoch_metrics.items():
                    if key in self.training_history:
                        self.training_history[key].append(value)

                # Track metrics
                self.metrics_tracker.update(epoch_metrics, epoch)

                # Check for best model
                if val_metrics and "val_loss" in val_metrics:
                    current_val_loss = val_metrics["val_loss"]
                    if self.best_metric is None or current_val_loss < self.best_metric:
                        self.best_metric = current_val_loss
                        self.is_best_epoch = True
                    else:
                        self.is_best_epoch = False

                # Logging
                metrics_str = ", ".join(
                    [
                        f"{k}: {v:.4f}"
                        for k, v in epoch_metrics.items()
                        if k != "epoch_time"
                    ]
                )
                self.logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.2f}s - {metrics_str}"
                )

                # Save checkpoint
                if (
                    epoch + 1
                ) % self.config.save_every_n_epochs == 0 or self.is_best_epoch:
                    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
                    self.save_checkpoint(checkpoint_path, epoch_metrics)

                    if self.is_best_epoch:
                        best_path = output_dir / "best_model.pth"
                        self.save_checkpoint(best_path, epoch_metrics)

                # Early stopping check
                if self.early_stopping and val_metrics:
                    if self.early_stopping(
                        val_metrics.get("val_loss", float("inf")), self.model
                    ):
                        self.logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

                # Log to experiment tracker
                if self.experiment_tracker:
                    self.experiment_tracker.log_metrics(epoch_metrics, step=epoch)

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {training_time:.2f}s")

        # Save final model
        final_path = output_dir / "final_model.pth"
        self.save_checkpoint(final_path, {"final_epoch": self.current_epoch})

        return {
            "training_time": training_time,
            "final_epoch": self.current_epoch,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
        }

    def save_checkpoint(
        self, checkpoint_path: Union[str, Path], metrics: Optional[Dict] = None
    ):
        """Save training checkpoint."""
        checkpoint_data = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config,
        }

        if self.scheduler:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.use_amp:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()

        if metrics:
            checkpoint_data["metrics"] = metrics

        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state
            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Load scaler state
            if "scaler_state_dict" in checkpoint and self.use_amp:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            # Load training state
            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.best_metric = checkpoint.get("best_metric")
            self.training_history = checkpoint.get("training_history", {})

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"Resuming from epoch {self.current_epoch}")

            return checkpoint

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

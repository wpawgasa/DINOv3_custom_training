import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ..utils.logging import get_logger


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics."""
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file"):
        """Log artifact (file, model, etc.)."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish the experiment."""
        pass


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        logger=None
    ):
        self.logger = logger or get_logger("wandb_tracker")
        
        try:
            import wandb
            self.wandb = wandb
            self.run = None
            
            # Initialize run
            self.run = wandb.init(
                project=project,
                name=name,
                entity=entity,
                config=config,
                tags=tags,
                reinit=True
            )
            
            self.logger.info(f"WandB run initialized: {self.run.name}")
            
        except ImportError:
            self.logger.error("wandb not installed. Install with: pip install wandb")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to WandB."""
        if self.run:
            self.wandb.log(metrics, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to WandB."""
        if self.run:
            self.wandb.config.update(params)
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file"):
        """Log artifact to WandB."""
        if self.run:
            artifact = self.wandb.Artifact(
                name=Path(file_path).stem,
                type=artifact_type
            )
            artifact.add_file(str(file_path))
            self.run.log_artifact(artifact)
    
    def log_image(self, image: Union[np.ndarray, Image.Image, str, Path], caption: str = ""):
        """Log image to WandB."""
        if self.run:
            self.wandb.log({caption or "image": self.wandb.Image(image, caption=caption)})
    
    def log_table(self, data: List[List], columns: List[str], key: str = "table"):
        """Log table to WandB."""
        if self.run:
            table = self.wandb.Table(data=data, columns=columns)
            self.wandb.log({key: table})
    
    def finish(self):
        """Finish WandB run."""
        if self.run:
            self.run.finish()
            self.logger.info("WandB run finished")


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""
    
    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        logger=None
    ):
        self.logger = logger or get_logger("mlflow_tracker")
        
        try:
            import mlflow
            import mlflow.pytorch
            self.mlflow = mlflow
            
            # Set tracking URI if provided
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except:
                experiment_id = mlflow.create_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_id=experiment_id)
            
            # Start run
            self.run = mlflow.start_run(run_name=run_name)
            
            self.logger.info(f"MLflow run started: {self.run.info.run_name}")
            
        except ImportError:
            self.logger.error("mlflow not installed. Install with: pip install mlflow")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to MLflow."""
        for key, value in metrics.items():
            self.mlflow.log_metric(key, value, step=step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to MLflow."""
        # Convert complex objects to strings
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                clean_params[key] = value
            else:
                clean_params[key] = str(value)
        
        self.mlflow.log_params(clean_params)
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file"):
        """Log artifact to MLflow."""
        self.mlflow.log_artifact(str(file_path))
    
    def log_model(self, model, model_name: str = "model"):
        """Log PyTorch model to MLflow."""
        self.mlflow.pytorch.log_model(model, model_name)
    
    def finish(self):
        """Finish MLflow run."""
        self.mlflow.end_run()
        self.logger.info("MLflow run finished")


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard experiment tracker."""
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        logger=None
    ):
        self.logger = logger or get_logger("tensorboard_tracker")
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.logger.info(f"TensorBoard logging to: {log_dir}")
            
        except ImportError:
            self.logger.error("tensorboard not available. Install with: pip install tensorboard")
            raise
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # TensorBoard hyperparameters require metrics as well
        # For now, just log as text
        hparam_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        self.writer.add_text("hyperparameters", hparam_text)
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file"):
        """TensorBoard doesn't support general artifacts."""
        self.logger.warning("TensorBoard doesn't support general artifact logging")
    
    def log_image(self, image: np.ndarray, tag: str, step: Optional[int] = None):
        """Log image to TensorBoard."""
        self.writer.add_image(tag, image, step)
    
    def finish(self):
        """Close TensorBoard writer."""
        self.writer.close()
        self.logger.info("TensorBoard writer closed")


class CompositeTracker(ExperimentTracker):
    """Composite tracker that combines multiple trackers."""
    
    def __init__(self, trackers: List[ExperimentTracker]):
        self.trackers = trackers
        self.logger = get_logger("composite_tracker")
        self.logger.info(f"Initialized composite tracker with {len(trackers)} trackers")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {type(tracker).__name__}: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_hyperparameters(params)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to {type(tracker).__name__}: {e}")
    
    def log_artifact(self, file_path: Union[str, Path], artifact_type: str = "file"):
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(file_path, artifact_type)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to {type(tracker).__name__}: {e}")
    
    def finish(self):
        """Finish all trackers."""
        for tracker in self.trackers:
            try:
                tracker.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish {type(tracker).__name__}: {e}")


class ExperimentManager:
    """High-level experiment management with visualization and analysis."""
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        logger=None
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.logger = logger or get_logger("experiment_manager")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trackers list
        self.trackers: List[ExperimentTracker] = []
        
        # Metrics storage for analysis
        self.metrics_history = []
        self.step_counter = 0
    
    def add_wandb_tracking(
        self,
        project: str,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """Add Weights & Biases tracking."""
        try:
            tracker = WandbTracker(
                project=project,
                name=self.experiment_name,
                entity=entity,
                config=self.config,
                tags=tags,
                logger=self.logger
            )
            self.trackers.append(tracker)
            self.logger.info("Added WandB tracking")
        except Exception as e:
            self.logger.warning(f"Failed to add WandB tracking: {e}")
    
    def add_mlflow_tracking(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """Add MLflow tracking."""
        try:
            tracker = MLflowTracker(
                experiment_name=experiment_name or self.experiment_name,
                run_name=self.experiment_name,
                tracking_uri=tracking_uri,
                logger=self.logger
            )
            self.trackers.append(tracker)
            self.logger.info("Added MLflow tracking")
        except Exception as e:
            self.logger.warning(f"Failed to add MLflow tracking: {e}")
    
    def add_tensorboard_tracking(self):
        """Add TensorBoard tracking."""
        try:
            log_dir = self.output_dir / "tensorboard"
            tracker = TensorBoardTracker(log_dir=log_dir, logger=self.logger)
            self.trackers.append(tracker)
            self.logger.info("Added TensorBoard tracking")
        except Exception as e:
            self.logger.warning(f"Failed to add TensorBoard tracking: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log metrics to all trackers and store for analysis."""
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Store metrics for analysis
        metrics_with_step = {"step": step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
        # Log to all trackers
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_hyperparameters(params)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def create_training_plots(self, save_path: Optional[Path] = None) -> Path:
        """Create and save training plots."""
        if not self.metrics_history:
            self.logger.warning("No metrics history available for plotting")
            return None
        
        save_path = save_path or (self.output_dir / "training_plots.png")
        
        # Extract metrics
        steps = [m["step"] for m in self.metrics_history]
        
        # Determine available metrics
        available_metrics = set()
        for m in self.metrics_history:
            available_metrics.update(m.keys())
        available_metrics.discard("step")
        
        if not available_metrics:
            self.logger.warning("No metrics to plot")
            return None
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2,
            figsize=(15, 4 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics == 2:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(sorted(available_metrics)):
            values = []
            metric_steps = []
            
            for m in self.metrics_history:
                if metric in m:
                    values.append(m[metric])
                    metric_steps.append(m["step"])
            
            if values:
                axes[i].plot(metric_steps, values, linewidth=2)
                axes[i].set_title(metric.replace("_", " ").title())
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel("Value")
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {save_path}")
        return save_path
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate experiment summary statistics."""
        if not self.metrics_history:
            return {}
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.metrics_history),
            "config": self.config,
        }
        
        # Calculate summary statistics for each metric
        available_metrics = set()
        for m in self.metrics_history:
            available_metrics.update(m.keys())
        available_metrics.discard("step")
        
        for metric in available_metrics:
            values = [m[metric] for m in self.metrics_history if metric in m]
            if values:
                summary[f"{metric}_final"] = values[-1]
                summary[f"{metric}_best"] = min(values) if "loss" in metric else max(values)
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)
        
        return summary
    
    def save_experiment_summary(self, file_path: Optional[Path] = None):
        """Save experiment summary to file."""
        file_path = file_path or (self.output_dir / "experiment_summary.json")
        
        summary = self.generate_experiment_summary()
        
        import json
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Experiment summary saved to {file_path}")
    
    def finish(self):
        """Finish experiment tracking and generate final reports."""
        # Generate final plots and summary
        self.create_training_plots()
        self.save_experiment_summary()
        
        # Finish all trackers
        for tracker in self.trackers:
            try:
                tracker.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish tracker: {e}")
        
        self.logger.info(f"Experiment '{self.experiment_name}' completed")


def create_experiment_manager(
    experiment_name: str,
    output_dir: Union[str, Path],
    config: Dict[str, Any],
    use_wandb: bool = False,
    use_mlflow: bool = False,
    use_tensorboard: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    **kwargs
) -> ExperimentManager:
    """Create experiment manager with specified tracking backends."""
    
    manager = ExperimentManager(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config
    )
    
    if use_tensorboard:
        manager.add_tensorboard_tracking()
    
    if use_wandb and wandb_project:
        manager.add_wandb_tracking(
            project=wandb_project,
            entity=wandb_entity,
            **kwargs
        )
    
    if use_mlflow:
        manager.add_mlflow_tracking(
            tracking_uri=mlflow_tracking_uri,
            **kwargs
        )
    
    # Log initial hyperparameters
    manager.log_hyperparameters(config)
    
    return manager
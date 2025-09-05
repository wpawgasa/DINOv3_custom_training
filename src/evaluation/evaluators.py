import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from ..models.dinov3_classifier import DINOv3Classifier
from ..utils.device import get_device_manager
from ..utils.logging import get_logger


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, model: DINOv3Classifier, logger=None):
        self.model = model
        self.logger = logger or get_logger("evaluator")
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        self.model = self.model.to(self.device)

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, **kwargs) -> Dict[str, Any]:
        """Perform evaluation and return metrics."""
        pass

    def _prepare_model_for_evaluation(self):
        """Prepare model for evaluation."""
        self.model.eval()
        torch.set_grad_enabled(False)

    def _restore_model_state(self):
        """Restore model to training mode if needed."""
        torch.set_grad_enabled(True)


class ClassificationEvaluator(BaseEvaluator):
    """Comprehensive evaluator for classification tasks."""

    def __init__(
        self,
        model: DINOv3Classifier,
        class_names: Optional[List[str]] = None,
        logger=None,
    ):
        super().__init__(model, logger)
        self.class_names = class_names
        self.num_classes = model.num_classes

        if class_names and len(class_names) != self.num_classes:
            raise ValueError(
                f"Number of class names ({len(class_names)}) doesn't match model classes ({self.num_classes})"
            )

    def evaluate(
        self,
        dataloader: DataLoader,
        compute_per_class_metrics: bool = True,
        compute_confusion_matrix: bool = True,
        compute_roc_curves: bool = True,
        save_predictions: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive classification evaluation."""

        self.logger.info("Starting classification evaluation...")
        self._prepare_model_for_evaluation()

        # Storage for predictions and targets
        all_predictions = []
        all_targets = []
        all_probabilities = []
        prediction_times = []

        total_samples = 0
        correct_predictions = 0

        try:
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    batch_start_time = time.time()

                    # Move to device
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # Forward pass
                    outputs = self.model(inputs)

                    # Get predictions and probabilities
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                    # Store results
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())

                    # Calculate batch accuracy
                    batch_correct = predicted.eq(targets).sum().item()
                    correct_predictions += batch_correct
                    total_samples += targets.size(0)

                    # Track inference time
                    batch_time = time.time() - batch_start_time
                    prediction_times.append(
                        batch_time / targets.size(0)
                    )  # Per sample time

                    if batch_idx % 100 == 0:
                        batch_acc = 100.0 * batch_correct / targets.size(0)
                        self.logger.info(
                            f"Batch {batch_idx}: Accuracy = {batch_acc:.2f}%"
                        )

        finally:
            self._restore_model_state()

        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)

        # Calculate basic metrics
        overall_accuracy = 100.0 * correct_predictions / total_samples
        avg_inference_time = np.mean(prediction_times)

        self.logger.info(
            f"Evaluation completed: {overall_accuracy:.2f}% accuracy on {total_samples} samples"
        )

        # Compute comprehensive metrics
        results = {
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "avg_inference_time_per_sample": avg_inference_time,
            "predictions": y_pred.tolist() if save_predictions else None,
            "targets": y_true.tolist() if save_predictions else None,
        }

        # Per-class metrics
        if compute_per_class_metrics:
            per_class_metrics = self._compute_per_class_metrics(y_true, y_pred, y_proba)
            results.update(per_class_metrics)

        # Confusion matrix
        if compute_confusion_matrix:
            cm_results = self._compute_confusion_matrix(y_true, y_pred)
            results.update(cm_results)

        # ROC curves and AUC
        if compute_roc_curves and self.num_classes > 1:
            roc_results = self._compute_roc_metrics(y_true, y_proba)
            results.update(roc_results)

        # Top-k accuracy (if applicable)
        if self.num_classes > 5:
            top5_acc = self._compute_top_k_accuracy(y_true, y_proba, k=5)
            results["top5_accuracy"] = top5_acc

        # Save results if output directory specified
        if output_dir:
            self._save_evaluation_results(results, output_dir)

        return results

    def _compute_per_class_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute per-class precision, recall, and F1-score."""

        # Calculate precision, recall, F1 for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Calculate macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        (
            weighted_precision,
            weighted_recall,
            weighted_f1,
            _,
        ) = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        results = {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class_precision": precision.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "per_class_support": support.tolist(),
        }

        # Add class names if available
        if self.class_names:
            class_metrics = {}
            for i, class_name in enumerate(self.class_names):
                class_metrics[f"{class_name}_precision"] = precision[i]
                class_metrics[f"{class_name}_recall"] = recall[i]
                class_metrics[f"{class_name}_f1"] = f1[i]
                class_metrics[f"{class_name}_support"] = support[i]

            results["class_metrics"] = class_metrics

        return results

    def _compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute and analyze confusion matrix."""

        cm = confusion_matrix(y_true, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        results = {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_normalized.tolist(),
        }

        return results

    def _compute_roc_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute ROC curves and AUC metrics."""

        results = {}

        try:
            if self.num_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_true, y_proba[:, 1])
                fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])

                results.update(
                    {
                        "auc": auc,
                        "roc_curve": {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "thresholds": thresholds.tolist(),
                        },
                    }
                )

                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(
                    y_true, y_proba[:, 1]
                )
                avg_precision = average_precision_score(y_true, y_proba[:, 1])

                results.update(
                    {
                        "average_precision": avg_precision,
                        "pr_curve": {
                            "precision": precision.tolist(),
                            "recall": recall.tolist(),
                            "thresholds": pr_thresholds.tolist(),
                        },
                    }
                )

            else:
                # Multi-class classification
                # One-vs-rest AUC
                y_true_onehot = np.eye(self.num_classes)[y_true]

                # Macro average AUC
                auc_macro = roc_auc_score(
                    y_true_onehot, y_proba, average="macro", multi_class="ovr"
                )

                # Weighted average AUC
                auc_weighted = roc_auc_score(
                    y_true_onehot, y_proba, average="weighted", multi_class="ovr"
                )

                results.update(
                    {
                        "auc_macro": auc_macro,
                        "auc_weighted": auc_weighted,
                    }
                )

                # Per-class AUC
                per_class_auc = []
                for i in range(self.num_classes):
                    try:
                        auc_i = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
                        per_class_auc.append(auc_i)
                    except ValueError:
                        # Handle case where class is not present in test set
                        per_class_auc.append(0.0)

                results["per_class_auc"] = per_class_auc

        except ValueError as e:
            self.logger.warning(f"Could not compute ROC metrics: {e}")

        return results

    def _compute_top_k_accuracy(
        self, y_true: np.ndarray, y_proba: np.ndarray, k: int = 5
    ) -> float:
        """Compute top-k accuracy."""

        top_k_pred = np.argsort(y_proba, axis=1)[:, -k:]

        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top_k_pred[i]:
                correct += 1

        return 100.0 * correct / len(y_true)

    def _save_evaluation_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save numerical results as JSON
        import json

        results_path = output_dir / "evaluation_results.json"

        # Create a copy without large arrays for JSON saving
        json_results = {
            k: v
            for k, v in results.items()
            if k not in ["predictions", "targets", "roc_curve", "pr_curve"]
        }

        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2, default=str)

        self.logger.info(f"Evaluation results saved to {results_path}")


class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation tasks."""

    def __init__(self, model: DINOv3Classifier, num_classes: int, logger=None):
        super().__init__(model, logger)
        self.num_classes = num_classes

    def evaluate(self, dataloader: DataLoader, **kwargs) -> Dict[str, Any]:
        """Evaluate segmentation model."""
        self.logger.info("Starting segmentation evaluation...")
        self._prepare_model_for_evaluation()

        total_iou = 0.0
        total_pixel_accuracy = 0.0
        class_ious = np.zeros(self.num_classes)
        class_pixel_counts = np.zeros(self.num_classes)
        total_samples = 0

        try:
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs = self.model(inputs)
                    predictions = torch.argmax(outputs, dim=1)

                    # Calculate metrics
                    (
                        batch_iou,
                        batch_pixel_acc,
                        batch_class_ious,
                    ) = self._compute_segmentation_metrics(predictions, targets)

                    total_iou += batch_iou
                    total_pixel_accuracy += batch_pixel_acc
                    class_ious += batch_class_ious
                    total_samples += inputs.size(0)

        finally:
            self._restore_model_state()

        # Calculate final metrics
        mean_iou = total_iou / total_samples
        mean_pixel_accuracy = total_pixel_accuracy / total_samples
        mean_class_ious = class_ious / total_samples

        results = {
            "mean_iou": mean_iou,
            "pixel_accuracy": mean_pixel_accuracy,
            "per_class_iou": mean_class_ious.tolist(),
            "total_samples": total_samples,
        }

        return results

    def _compute_segmentation_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float, np.ndarray]:
        """Compute IoU and pixel accuracy for a batch."""

        # Flatten tensors
        pred_flat = predictions.view(-1).cpu().numpy()
        target_flat = targets.view(-1).cpu().numpy()

        # Pixel accuracy
        pixel_acc = np.mean(pred_flat == target_flat)

        # IoU calculation
        ious = []
        class_ious = np.zeros(self.num_classes)

        for class_id in range(self.num_classes):
            pred_class = pred_flat == class_id
            target_class = target_flat == class_id

            intersection = np.logical_and(pred_class, target_class).sum()
            union = np.logical_or(pred_class, target_class).sum()

            if union > 0:
                iou = intersection / union
                ious.append(iou)
                class_ious[class_id] = iou
            else:
                ious.append(0.0)

        mean_iou = np.mean(ious)

        return mean_iou, pixel_acc, class_ious


class MultiTaskEvaluator(BaseEvaluator):
    """Evaluator for multi-task models."""

    def __init__(
        self,
        model: DINOv3Classifier,
        task_configs: Dict[str, Dict[str, Any]],
        logger=None,
    ):
        super().__init__(model, logger)
        self.task_configs = task_configs

        # Initialize individual evaluators for each task
        self.task_evaluators = {}
        for task_name, config in task_configs.items():
            if config["task_type"] == "classification":
                self.task_evaluators[task_name] = ClassificationEvaluator(
                    model, config.get("class_names"), logger
                )
            elif config["task_type"] == "segmentation":
                self.task_evaluators[task_name] = SegmentationEvaluator(
                    model, config["num_classes"], logger
                )

    def evaluate(self, dataloader: DataLoader, **kwargs) -> Dict[str, Any]:
        """Evaluate multi-task model."""
        self.logger.info("Starting multi-task evaluation...")
        self._prepare_model_for_evaluation()

        task_results = {}
        all_outputs = {}
        all_targets = {}

        try:
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device, non_blocking=True)

                    # Assuming targets is a dict for multi-task
                    if isinstance(targets, dict):
                        for task_name in targets:
                            targets[task_name] = targets[task_name].to(
                                self.device, non_blocking=True
                            )

                    outputs = self.model(inputs)

                    # Store outputs and targets for each task
                    for task_name in outputs:
                        if task_name not in all_outputs:
                            all_outputs[task_name] = []
                            all_targets[task_name] = []

                        all_outputs[task_name].append(outputs[task_name])
                        if isinstance(targets, dict) and task_name in targets:
                            all_targets[task_name].append(targets[task_name])

        finally:
            self._restore_model_state()

        # Evaluate each task separately
        for task_name in all_outputs:
            if task_name in self.task_evaluators:
                # This would require task-specific evaluation logic
                # For now, return basic information
                task_results[task_name] = {
                    "evaluated": True,
                    "num_batches": len(all_outputs[task_name]),
                }

        return {"task_results": task_results, "total_tasks": len(task_results)}


def create_evaluator(
    model: DINOv3Classifier,
    task_type: str = "classification",
    class_names: Optional[List[str]] = None,
    **kwargs,
) -> BaseEvaluator:
    """Factory function to create appropriate evaluator."""

    if task_type == "classification":
        return ClassificationEvaluator(model, class_names, **kwargs)
    elif task_type == "segmentation":
        return SegmentationEvaluator(model, model.num_classes, **kwargs)
    elif task_type == "multi_task":
        task_configs = kwargs.get("task_configs", {})
        return MultiTaskEvaluator(model, task_configs, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

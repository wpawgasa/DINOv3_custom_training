import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from ..models.dinov3_classifier import DINOv3Classifier
from ..utils.logging import get_logger


class VisualizationTools:
    """Comprehensive visualization tools for model evaluation."""

    def __init__(self, model: DINOv3Classifier, logger=None):
        self.model = model
        self.logger = logger or get_logger("visualization_tools")
        self.device = next(self.model.parameters()).device

        # Set up matplotlib for better plots
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Create confusion matrix visualization."""

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
            cmap = "Blues"
        else:
            fmt = "d"
            cmap = "Blues"

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm)),
            ax=ax,
        )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "ROC Curves",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot ROC curves for multi-class classification."""

        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(y_proba.shape[1]))
        n_classes = y_proba.shape[1]

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            class_name = class_names[i] if class_names else f"Class {i}"
            ax.plot(
                fpr, tpr, color=color, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})"
            )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.5)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"ROC curves saved to {save_path}")

        return fig

    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Precision-Recall Curves",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """Plot precision-recall curves."""

        from sklearn.metrics import average_precision_score, precision_recall_curve
        from sklearn.preprocessing import label_binarize

        y_true_bin = label_binarize(y_true, classes=range(y_proba.shape[1]))
        n_classes = y_proba.shape[1]

        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            avg_precision = average_precision_score(y_true_bin[:, i], y_proba[:, i])

            class_name = class_names[i] if class_names else f"Class {i}"
            ax.plot(
                recall,
                precision,
                color=color,
                lw=2,
                label=f"{class_name} (AP = {avg_precision:.2f})",
            )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"PR curves saved to {save_path}")

        return fig

    def extract_features(
        self,
        dataloader,
        layer_name: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from a specific layer."""

        self.model.eval()

        features = []
        labels = []
        samples_processed = 0

        # Hook to extract features
        extracted_features = {}

        def hook_fn(module, input, output):
            extracted_features["features"] = output.detach()

        # Register hook
        if layer_name:
            # This would need to be adapted based on the specific model architecture
            # For now, extract from the backbone output
            handle = self.model.backbone.register_forward_hook(hook_fn)
        else:
            # Extract from final backbone output
            handle = self.model.backbone.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                for inputs, targets in dataloader:
                    if max_samples and samples_processed >= max_samples:
                        break

                    inputs = inputs.to(self.device)

                    # Forward pass
                    _ = self.model(inputs)

                    # Extract features
                    batch_features = extracted_features["features"]

                    # Flatten features if necessary
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.view(batch_features.size(0), -1)

                    features.append(batch_features.cpu().numpy())
                    labels.append(targets.numpy())

                    samples_processed += inputs.size(0)

        finally:
            handle.remove()

        features = np.vstack(features)
        labels = np.concatenate(labels)

        self.logger.info(f"Extracted features shape: {features.shape}")

        return features, labels

    def visualize_feature_space(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        class_names: Optional[List[str]] = None,
        title: str = "Feature Space Visualization",
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """Visualize high-dimensional features in 2D."""

        self.logger.info(f"Reducing dimensions using {method.upper()}...")

        if method.lower() == "tsne":
            # Use PCA first if features are very high-dimensional
            if features.shape[1] > 50:
                pca = PCA(n_components=50)
                features_reduced = pca.fit_transform(features)
                self.logger.info("Applied PCA preprocessing for t-SNE")
            else:
                features_reduced = features

            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = reducer.fit_transform(features_reduced)

        elif method.lower() == "pca":
            reducer = PCA(n_components=2)
            features_2d = reducer.fit_transform(features)

        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)

        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = labels == label
            class_name = class_names[label] if class_names else f"Class {label}"

            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                label=class_name,
                alpha=0.7,
                s=20,
            )

        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Feature visualization saved to {save_path}")

        return fig

    def generate_attention_maps(
        self,
        images: torch.Tensor,
        layer_names: Optional[List[str]] = None,
        method: str = "gradcam",
    ) -> Dict[str, np.ndarray]:
        """Generate attention maps for input images."""

        self.logger.info(f"Generating attention maps using {method}")

        if method.lower() == "gradcam":
            return self._generate_gradcam_maps(images, layer_names)
        elif method.lower() == "integrated_gradients":
            return self._generate_integrated_gradients(images)
        else:
            raise ValueError(f"Unsupported attention method: {method}")

    def _generate_gradcam_maps(
        self, images: torch.Tensor, layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Generate Grad-CAM attention maps."""

        self.model.eval()
        images = images.to(self.device)
        images.requires_grad_(True)

        # Forward pass
        outputs = self.model(images)

        # Get predicted class
        pred_class = outputs.argmax(dim=1)

        # Backward pass
        class_scores = outputs[:, pred_class[0]]
        class_scores.backward()

        # Get gradients and activations (simplified)
        # This would need to be adapted based on the specific model architecture
        gradients = images.grad.data

        # Generate attention map
        attention_map = torch.mean(torch.abs(gradients), dim=1, keepdim=True)
        attention_map = F.interpolate(
            attention_map, size=images.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / attention_map.max()

        return {"gradcam": attention_map.detach().cpu().numpy()}

    def _generate_integrated_gradients(
        self, images: torch.Tensor, steps: int = 50
    ) -> Dict[str, np.ndarray]:
        """Generate integrated gradients attention maps."""

        self.model.eval()
        images = images.to(self.device)

        # Create baseline (zeros)
        baseline = torch.zeros_like(images)

        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps).to(self.device)

        integrated_gradients = torch.zeros_like(images)

        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (images - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            outputs = self.model(interpolated)
            pred_class = outputs.argmax(dim=1)

            # Backward pass
            class_scores = outputs[:, pred_class[0]]
            gradients = torch.autograd.grad(
                class_scores, interpolated, create_graph=False
            )[0]

            integrated_gradients += gradients

        # Average and scale by input difference
        integrated_gradients = (images - baseline) * integrated_gradients / steps

        # Create attention map
        attention_map = torch.mean(torch.abs(integrated_gradients), dim=1, keepdim=True)

        # Normalize
        attention_map = attention_map - attention_map.min()
        attention_map = attention_map / attention_map.max()

        return {"integrated_gradients": attention_map.detach().cpu().numpy()}

    def visualize_attention_maps(
        self,
        images: torch.Tensor,
        attention_maps: Dict[str, np.ndarray],
        titles: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """Visualize attention maps overlaid on original images."""

        batch_size = images.shape[0]
        n_methods = len(attention_maps)

        fig, axes = plt.subplots(
            batch_size,
            n_methods + 1,
            figsize=(figsize[0] * (n_methods + 1), figsize[1] * batch_size),
        )

        if batch_size == 1:
            axes = axes.reshape(1, -1)

        for i in range(batch_size):
            # Original image
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize

            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original", fontsize=12)
            axes[i, 0].axis("off")

            # Attention maps
            for j, (method_name, attention_data) in enumerate(attention_maps.items()):
                attention = attention_data[i, 0]  # Remove channel dim

                # Overlay attention on image
                axes[i, j + 1].imshow(img)
                axes[i, j + 1].imshow(attention, alpha=0.6, cmap="jet")

                title = titles[j] if titles and j < len(titles) else method_name.title()
                axes[i, j + 1].set_title(title, fontsize=12)
                axes[i, j + 1].axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Attention maps saved to {save_path}")

        return fig

    def create_evaluation_summary_plot(
        self,
        results: Dict[str, Any],
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """Create comprehensive evaluation summary plot."""

        fig = plt.figure(figsize=figsize)

        # Create grid for subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Overall accuracy
        ax1 = fig.add_subplot(gs[0, 0])
        accuracy = results.get("overall_accuracy", 0)
        ax1.bar(["Accuracy"], [accuracy], color="skyblue")
        ax1.set_ylim(0, 100)
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Overall Accuracy", fontsize=12)

        # Per-class metrics (if available)
        if "per_class_f1" in results:
            ax2 = fig.add_subplot(gs[0, 1])
            f1_scores = results["per_class_f1"]
            class_indices = range(len(f1_scores))
            ax2.bar(class_indices, f1_scores, color="lightcoral")
            ax2.set_xlabel("Class")
            ax2.set_ylabel("F1 Score")
            ax2.set_title("Per-Class F1 Scores", fontsize=12)

        # Confusion matrix (if available)
        if "confusion_matrix_normalized" in results:
            ax3 = fig.add_subplot(gs[0, 2])
            cm = np.array(results["confusion_matrix_normalized"])
            im = ax3.imshow(cm, cmap="Blues")
            ax3.set_title("Confusion Matrix", fontsize=12)
            plt.colorbar(im, ax=ax3)

        # ROC curve data (if available)
        if "roc_curve" in results:
            ax4 = fig.add_subplot(gs[1, 0])
            roc_data = results["roc_curve"]
            ax4.plot(roc_data["fpr"], roc_data["tpr"], "b-", lw=2)
            ax4.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax4.set_xlabel("False Positive Rate")
            ax4.set_ylabel("True Positive Rate")
            ax4.set_title("ROC Curve", fontsize=12)

        # Precision-Recall curve (if available)
        if "pr_curve" in results:
            ax5 = fig.add_subplot(gs[1, 1])
            pr_data = results["pr_curve"]
            ax5.plot(pr_data["recall"], pr_data["precision"], "g-", lw=2)
            ax5.set_xlabel("Recall")
            ax5.set_ylabel("Precision")
            ax5.set_title("Precision-Recall Curve", fontsize=12)

        # Model info text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")

        info_text = []
        if "total_samples" in results:
            info_text.append(f"Total Samples: {results['total_samples']}")
        if "avg_inference_time_per_sample" in results:
            info_text.append(
                f"Avg Inference Time: {results['avg_inference_time_per_sample']:.4f}s"
            )
        if "macro_f1" in results:
            info_text.append(f"Macro F1: {results['macro_f1']:.3f}")
        if "weighted_f1" in results:
            info_text.append(f"Weighted F1: {results['weighted_f1']:.3f}")

        ax6.text(
            0.1,
            0.9,
            "\n".join(info_text),
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
        )
        ax6.set_title("Evaluation Summary", fontsize=12)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Summary plot saved to {save_path}")

        return fig


def create_visualization_tools(model: DINOv3Classifier, **kwargs) -> VisualizationTools:
    """Create visualization tools."""
    return VisualizationTools(model, **kwargs)

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from ..utils.logging import get_logger
from ..utils.schemas import ModelVariant, TaskType, TrainingMode


class TaskHead(nn.Module):
    """Base class for task-specific heads."""

    def __init__(self, input_dim: int, task_type: TaskType):
        super().__init__()
        self.input_dim = input_dim
        self.task_type = task_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ClassificationHead(TaskHead):
    """Classification head with dropout and regularization."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__(input_dim, TaskType.CLASSIFICATION)

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, hidden_dim))

                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                elif use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))

                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "swish":
                    layers.append(nn.SiLU())

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

                current_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SegmentationHead(TaskHead):
    """Segmentation head for dense prediction tasks."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.1,
    ):
        super().__init__(input_dim, TaskType.SEGMENTATION)

        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskHead(TaskHead):
    """Multi-task head supporting multiple outputs."""

    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict[str, Any]],
        shared_hidden_dim: Optional[int] = None,
    ):
        super().__init__(
            input_dim, TaskType.CLASSIFICATION
        )  # Default to classification

        self.task_names = list(task_configs.keys())

        # Shared feature extractor
        if shared_hidden_dim:
            self.shared_features = nn.Sequential(
                nn.Linear(input_dim, shared_hidden_dim),
                nn.LayerNorm(shared_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            head_input_dim = shared_hidden_dim
        else:
            self.shared_features = nn.Identity()
            head_input_dim = input_dim

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            if config["task_type"] == "classification":
                head = ClassificationHead(
                    input_dim=head_input_dim,
                    num_classes=config["num_classes"],
                    **config.get("head_kwargs", {}),
                )
            elif config["task_type"] == "segmentation":
                head = SegmentationHead(
                    input_dim=head_input_dim,
                    num_classes=config["num_classes"],
                    **config.get("head_kwargs", {}),
                )
            else:
                raise ValueError(f"Unsupported task type: {config['task_type']}")

            self.task_heads[task_name] = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_features(x)

        outputs = {}
        for task_name in self.task_names:
            outputs[task_name] = self.task_heads[task_name](shared_features)

        return outputs


class DINOv3Backbone(nn.Module):
    """DINOv3 backbone wrapper supporting different variants."""

    MODEL_CONFIGS = {
        # ViT models
        "dinov3_vits16": {
            "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "embed_dim": 384,
            "patch_size": 16,
        },
        "dinov3_vits16plus": {
            "model_name": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
            "embed_dim": 384,
            "patch_size": 16,
        },
        "dinov3_vitb16": {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "embed_dim": 768,
            "patch_size": 16,
        },
        "dinov3_vitl16": {
            "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "embed_dim": 1024,
            "patch_size": 16,
        },
        "dinov3_vitl16_sat": {
            "model_name": "facebook/dinov3-vitl16-pretrain-sat493m",
            "embed_dim": 1024,
            "patch_size": 16,
        },
        "dinov3_vith16plus": {
            "model_name": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            "embed_dim": 1536,
            "patch_size": 16,
        },
        "dinov3_vit7b16": {
            "model_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
            "embed_dim": 3072,
            "patch_size": 16,
        },
        "dinov3_vit7b16_sat": {
            "model_name": "facebook/dinov3-vit7b16-pretrain-sat493m",
            "embed_dim": 3072,
            "patch_size": 16,
        },
        # ConvNeXt models
        "dinov3_convnext_tiny": {
            "model_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            "embed_dim": 768,
            "is_convnext": True,
        },
        "dinov3_convnext_small": {
            "model_name": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
            "embed_dim": 768,
            "is_convnext": True,
        },
        "dinov3_convnext_base": {
            "model_name": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
            "embed_dim": 1024,
            "is_convnext": True,
        },
        "dinov3_convnext_large": {
            "model_name": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            "embed_dim": 1536,
            "is_convnext": True,
        },
    }

    def __init__(
        self,
        variant: str,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        logger=None,
    ):
        super().__init__()

        self.variant = variant
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.logger = logger or get_logger("dinov3_backbone")

        if variant not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown variant: {variant}. Available: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.config = self.MODEL_CONFIGS[variant]
        self.embed_dim = self.config["embed_dim"]
        self.is_convnext = self.config.get("is_convnext", False)

        self._load_backbone()

        if freeze_backbone:
            self.freeze()

    def _load_backbone(self):
        try:
            if self.pretrained:
                self.backbone = AutoModel.from_pretrained(self.config["model_name"])
            else:
                config = AutoConfig.from_pretrained(self.config["model_name"])
                self.backbone = AutoModel.from_config(config)

            self.logger.info(
                f"Loaded {self.variant} backbone (pretrained={self.pretrained})"
            )

        except Exception as e:
            self.logger.error(f"Failed to load backbone {self.variant}: {e}")
            # Fallback warning
            self.logger.warning(
                f"Consider checking if model {self.config['model_name']} is available"
            )
            raise

    def freeze(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info(f"Froze backbone parameters for {self.variant}")

    def unfreeze(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.logger.info(f"Unfroze backbone parameters for {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            outputs = self.backbone(x)

            if self.is_convnext:
                # ConvNeXt models typically return features
                if hasattr(outputs, "last_hidden_state"):
                    features = outputs.last_hidden_state
                    # Global average pooling for ConvNeXt
                    if len(features.shape) == 4:  # [B, C, H, W]
                        features = features.mean(dim=[2, 3])  # [B, C]
                    elif len(features.shape) == 3:  # [B, N, C]
                        features = features.mean(dim=1)  # [B, C]
                else:
                    features = outputs
            else:
                # ViT models - take CLS token
                if hasattr(outputs, "last_hidden_state"):
                    features = outputs.last_hidden_state[:, 0]  # CLS token
                elif hasattr(outputs, "pooler_output"):
                    features = outputs.pooler_output
                else:
                    # Fallback: assume outputs is the feature tensor
                    if len(outputs.shape) == 3:  # [B, N, C]
                        features = outputs[:, 0]  # Take first token (CLS)
                    else:
                        features = outputs

            return features

        except Exception as e:
            self.logger.error(f"Error in backbone forward pass: {e}")
            raise


class DINOv3Classifier(nn.Module):
    """Complete DINOv3 classifier with flexible task heads and training modes."""

    def __init__(
        self,
        variant: ModelVariant,
        task_type: TaskType,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1,
        use_head_norm: bool = True,
        head_hidden_dims: Optional[List[int]] = None,
        multi_task_configs: Optional[Dict[str, Dict]] = None,
        logger=None,
    ):
        super().__init__()

        self.variant = variant.value if isinstance(variant, ModelVariant) else variant
        self.task_type = (
            task_type.value if isinstance(task_type, TaskType) else task_type
        )
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.logger = logger or get_logger("dinov3_classifier")

        # Load backbone
        self.backbone = DINOv3Backbone(
            variant=self.variant,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            logger=self.logger,
        )

        self.embed_dim = self.backbone.embed_dim

        # Create task head(s)
        if multi_task_configs:
            self.head = MultiTaskHead(
                input_dim=self.embed_dim, task_configs=multi_task_configs
            )
            self.is_multi_task = True
        else:
            if self.task_type == "classification":
                self.head = ClassificationHead(
                    input_dim=self.embed_dim,
                    num_classes=num_classes,
                    hidden_dims=head_hidden_dims,
                    dropout_rate=dropout_rate,
                    use_layer_norm=use_head_norm,
                )
            elif self.task_type == "segmentation":
                self.head = SegmentationHead(
                    input_dim=self.embed_dim,
                    num_classes=num_classes,
                    dropout_rate=dropout_rate,
                )
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            self.is_multi_task = False

        self.logger.info(
            f"Created {self.variant} classifier for {self.task_type} with {num_classes} classes"
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Extract features from backbone
        features = self.backbone(x)

        # Apply task head(s)
        outputs = self.head(features)

        return outputs

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> List[Dict]:
        """Get parameter groups for differential learning rates."""
        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": backbone_lr,
                "name": "backbone",
            },
            {"params": self.head.parameters(), "lr": head_lr, "name": "head"},
        ]

        return param_groups

    def set_training_mode(self, mode: TrainingMode):
        """Set training mode (linear probe, full fine-tune, etc.)."""
        if isinstance(mode, TrainingMode):
            mode = mode.value

        if mode == "linear_probe":
            self.backbone.freeze()
            self.logger.info("Set to linear probing mode (backbone frozen)")
        elif mode == "full_fine_tune":
            self.backbone.unfreeze()
            self.logger.info("Set to full fine-tuning mode (backbone unfrozen)")
        elif mode == "ssl_continue":
            # For SSL continuation, typically the whole model is trainable
            self.backbone.unfreeze()
            self.logger.info("Set to SSL continuation mode")
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())

        # Estimate model size in MB (float32)
        model_size_mb = (total_params * 4) / (1024 * 1024)

        return {
            "variant": self.variant,
            "task_type": self.task_type,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_parameters": backbone_params,
            "head_parameters": head_params,
            "model_size_mb": model_size_mb,
            "embed_dim": self.embed_dim,
            "is_multi_task": self.is_multi_task,
            "backbone_frozen": self.freeze_backbone,
        }

    def load_pretrained_weights(self, checkpoint_path: Union[str, Path]):
        """Load pretrained weights from checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Handle potential key mismatches
            model_keys = set(self.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            missing_keys = model_keys - checkpoint_keys
            unexpected_keys = checkpoint_keys - model_keys

            if missing_keys:
                self.logger.warning(
                    f"Missing keys in checkpoint: {list(missing_keys)[:10]}..."
                )

            if unexpected_keys:
                self.logger.warning(
                    f"Unexpected keys in checkpoint: {list(unexpected_keys)[:10]}..."
                )

            # Load weights
            self.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded pretrained weights from {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise

    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        epoch: int,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        **kwargs,
    ):
        """Save model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_model_info(),
            "epoch": epoch,
            **kwargs,
        }

        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state

        if scheduler_state:
            checkpoint["scheduler_state_dict"] = scheduler_state

        if metrics:
            checkpoint["metrics"] = metrics

        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

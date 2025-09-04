from typing import Dict, Any, Optional, Union
from pathlib import Path
import warnings

from omegaconf import DictConfig

from .dinov3_classifier import DINOv3Classifier
from ..utils.schemas import ModelConfig, ModelVariant, TaskType, TrainingMode
from ..utils.config import validate_config
from ..utils.logging import get_logger


class ModelFactory:
    """Factory class for creating DINOv3 models with different configurations."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger("model_factory")
        
        # Model size recommendations (in MB) for different variants
        self.model_memory_estimates = {
            'dinov3_vits16': 2.1,
            'dinov3_vits16plus': 2.3,
            'dinov3_vitb16': 3.4,
            'dinov3_vitl16': 6.2,
            'dinov3_vitl16_sat': 6.2,
            'dinov3_vith16plus': 12.8,
            'dinov3_vit7b16': 48.0,
            'dinov3_vit7b16_sat': 48.0,
            'dinov3_convnext_tiny': 2.8,
            'dinov3_convnext_small': 3.1,
            'dinov3_convnext_base': 4.1,
            'dinov3_convnext_large': 8.3,
        }
    
    def create_model(
        self,
        config: Union[ModelConfig, DictConfig, Dict[str, Any]],
        **kwargs
    ) -> DINOv3Classifier:
        """Create a DINOv3 model from configuration."""
        
        # Convert config to ModelConfig if needed
        if isinstance(config, dict):
            config = DictConfig(config)
        
        if isinstance(config, DictConfig):
            # Validate configuration
            try:
                validated_config = ModelConfig(**config)
            except Exception as e:
                raise ValueError(f"Invalid model configuration: {e}")
        elif isinstance(config, ModelConfig):
            validated_config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        
        # Extract model parameters
        model_params = {
            'variant': validated_config.variant,
            'task_type': validated_config.task,
            'num_classes': validated_config.num_classes,
            'pretrained': validated_config.pretrained,
            'freeze_backbone': validated_config.freeze_backbone,
            'dropout_rate': validated_config.dropout_rate,
            'use_head_norm': validated_config.use_head_norm,
            'logger': self.logger,
            **kwargs
        }
        
        # Create and return model
        model = DINOv3Classifier(**model_params)
        
        # Log model info
        model_info = model.get_model_info()
        self.logger.info(f"Created model: {model_info['variant']}")
        self.logger.info(f"Parameters: {model_info['total_parameters']:,}")
        self.logger.info(f"Model size: {model_info['model_size_mb']:.1f} MB")
        
        return model
    
    def create_classification_model(
        self,
        variant: Union[str, ModelVariant],
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden_dims: Optional[list] = None,
        **kwargs
    ) -> DINOv3Classifier:
        """Convenience method for creating classification models."""
        
        config = {
            'variant': variant,
            'task': TaskType.CLASSIFICATION,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': freeze_backbone,
        }
        
        model_params = {
            'head_hidden_dims': hidden_dims,
            **kwargs
        }
        
        return self.create_model(config, **model_params)
    
    def create_segmentation_model(
        self,
        variant: Union[str, ModelVariant],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> DINOv3Classifier:
        """Convenience method for creating segmentation models."""
        
        config = {
            'variant': variant,
            'task': TaskType.SEGMENTATION,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': False,  # Segmentation typically needs unfrozen backbone
        }
        
        return self.create_model(config, **kwargs)
    
    def create_multi_task_model(
        self,
        variant: Union[str, ModelVariant],
        task_configs: Dict[str, Dict[str, Any]],
        pretrained: bool = True,
        shared_hidden_dim: Optional[int] = None,
        **kwargs
    ) -> DINOv3Classifier:
        """Create a multi-task model."""
        
        # Use the first task's num_classes as default (required by schema)
        first_task = list(task_configs.values())[0]
        num_classes = first_task.get('num_classes', 2)
        
        config = {
            'variant': variant,
            'task': TaskType.CLASSIFICATION,  # Default
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': False,
        }
        
        model_params = {
            'multi_task_configs': task_configs,
            **kwargs
        }
        
        return self.create_model(config, **model_params)
    
    def create_linear_probe_model(
        self,
        variant: Union[str, ModelVariant],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> DINOv3Classifier:
        """Create a model for linear probing."""
        
        config = {
            'variant': variant,
            'task': TaskType.CLASSIFICATION,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': True,
        }
        
        model = self.create_model(config, **kwargs)
        model.set_training_mode(TrainingMode.LINEAR_PROBE)
        
        return model
    
    def create_fine_tuning_model(
        self,
        variant: Union[str, ModelVariant],
        num_classes: int,
        pretrained: bool = True,
        **kwargs
    ) -> DINOv3Classifier:
        """Create a model for full fine-tuning."""
        
        config = {
            'variant': variant,
            'task': TaskType.CLASSIFICATION,
            'num_classes': num_classes,
            'pretrained': pretrained,
            'freeze_backbone': False,
        }
        
        model = self.create_model(config, **kwargs)
        model.set_training_mode(TrainingMode.FULL_FINE_TUNE)
        
        return model
    
    def get_recommended_variant(
        self,
        dataset_size: int,
        available_memory_gb: float,
        target_accuracy: str = "balanced"
    ) -> str:
        """Get recommended model variant based on constraints."""
        
        available_memory_mb = available_memory_gb * 1024
        
        # Filter models by memory constraints
        suitable_models = []
        for variant, memory_req in self.model_memory_estimates.items():
            # Account for batch processing (estimate 4x memory usage during training)
            training_memory = memory_req * 4
            if training_memory <= available_memory_mb:
                suitable_models.append((variant, memory_req))
        
        if not suitable_models:
            self.logger.warning("No models fit in available memory. Consider using smaller batch size or more memory.")
            return "dinov3_vits16"  # Smallest model as fallback
        
        # Sort by model size
        suitable_models.sort(key=lambda x: x[1])
        
        # Make recommendation based on dataset size and target accuracy
        if dataset_size < 1000:
            # Small dataset - use smaller model to avoid overfitting
            recommended = suitable_models[0][0]
            self.logger.info(f"Small dataset detected. Recommending smaller model: {recommended}")
        elif dataset_size < 10000:
            # Medium dataset - balanced choice
            mid_idx = len(suitable_models) // 2
            recommended = suitable_models[mid_idx][0]
        else:
            # Large dataset - can use larger model
            if target_accuracy == "high":
                recommended = suitable_models[-1][0]  # Largest suitable model
            else:
                recommended = suitable_models[-2][0] if len(suitable_models) > 1 else suitable_models[-1][0]
        
        memory_req = self.model_memory_estimates[recommended]
        self.logger.info(f"Recommended variant: {recommended} (est. {memory_req} MB)")
        
        return recommended
    
    def load_model_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        config_override: Optional[Dict[str, Any]] = None
    ) -> DINOv3Classifier:
        """Load model from checkpoint file."""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract model config
            if 'model_config' in checkpoint:
                model_info = checkpoint['model_config']
                
                config = {
                    'variant': model_info['variant'],
                    'task': model_info['task_type'],
                    'num_classes': model_info['num_classes'],
                    'pretrained': False,  # Don't reload pretrained weights
                    'freeze_backbone': model_info.get('backbone_frozen', False),
                }
                
                # Apply any overrides
                if config_override:
                    config.update(config_override)
                
                # Create model
                model = self.create_model(config)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                
                self.logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
                
                # Log additional info if available
                if 'epoch' in checkpoint:
                    self.logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
                
                return model
                
            else:
                raise ValueError("Checkpoint does not contain model configuration")
                
        except Exception as e:
            self.logger.error(f"Failed to load model from checkpoint: {e}")
            raise
    
    def get_model_variants(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available model variants."""
        from .dinov3_classifier import DINOv3Backbone
        
        variants_info = {}
        for variant, config in DINOv3Backbone.MODEL_CONFIGS.items():
            variants_info[variant] = {
                'model_name': config['model_name'],
                'embed_dim': config['embed_dim'],
                'patch_size': config.get('patch_size'),
                'is_convnext': config.get('is_convnext', False),
                'estimated_memory_mb': self.model_memory_estimates.get(variant, 'unknown'),
                'recommended_for': self._get_variant_recommendations(variant)
            }
        
        return variants_info
    
    def _get_variant_recommendations(self, variant: str) -> str:
        """Get usage recommendations for a variant."""
        if 'vits16' in variant:
            return "Edge deployment, small datasets, fast inference"
        elif 'vitb16' in variant:
            return "Balanced performance and speed, general use"
        elif 'vitl16' in variant:
            return "High accuracy tasks, medium datasets"
        elif 'vith16plus' in variant or 'vit7b16' in variant:
            return "Maximum accuracy, large datasets, research"
        elif 'convnext' in variant:
            return "CNN alternative, good for dense prediction tasks"
        else:
            return "General purpose"


# Global factory instance
_model_factory = None

def get_model_factory() -> ModelFactory:
    """Get global model factory instance."""
    global _model_factory
    if _model_factory is None:
        _model_factory = ModelFactory()
    return _model_factory


# Convenience functions
def create_model(config: Union[ModelConfig, DictConfig, Dict[str, Any]], **kwargs) -> DINOv3Classifier:
    """Create a model using the global factory."""
    return get_model_factory().create_model(config, **kwargs)


def create_classification_model(
    variant: Union[str, ModelVariant],
    num_classes: int,
    **kwargs
) -> DINOv3Classifier:
    """Create a classification model."""
    return get_model_factory().create_classification_model(variant, num_classes, **kwargs)


def create_linear_probe_model(
    variant: Union[str, ModelVariant],
    num_classes: int,
    **kwargs
) -> DINOv3Classifier:
    """Create a linear probing model."""
    return get_model_factory().create_linear_probe_model(variant, num_classes, **kwargs)


def create_fine_tuning_model(
    variant: Union[str, ModelVariant],
    num_classes: int,
    **kwargs
) -> DINOv3Classifier:
    """Create a fine-tuning model."""
    return get_model_factory().create_fine_tuning_model(variant, num_classes, **kwargs)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config_override: Optional[Dict[str, Any]] = None
) -> DINOv3Classifier:
    """Load model from checkpoint."""
    return get_model_factory().load_model_from_checkpoint(checkpoint_path, config_override)


def get_recommended_variant(
    dataset_size: int,
    available_memory_gb: float,
    target_accuracy: str = "balanced"
) -> str:
    """Get recommended model variant."""
    return get_model_factory().get_recommended_variant(dataset_size, available_memory_gb, target_accuracy)
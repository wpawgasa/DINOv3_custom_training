from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"


class ModelVariant(str, Enum):
    VIT_S16 = "dinov3_vits16"
    VIT_S16_PLUS = "dinov3_vits16plus"
    VIT_B16 = "dinov3_vitb16"
    VIT_L16 = "dinov3_vitl16"
    VIT_H16_PLUS = "dinov3_vith16plus"
    VIT_7B16 = "dinov3_vit7b16"
    CONVNEXT_TINY = "dinov3_convnext_tiny"
    CONVNEXT_SMALL = "dinov3_convnext_small"
    CONVNEXT_BASE = "dinov3_convnext_base"
    CONVNEXT_LARGE = "dinov3_convnext_large"


class TrainingMode(str, Enum):
    LINEAR_PROBE = "linear_probe"
    FULL_FINE_TUNE = "full_fine_tune"
    SSL_CONTINUE = "ssl_continue"


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM = "adam"


class SchedulerType(str, Enum):
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    WARMUP_COSINE = "warmup_cosine"


class ModelConfig(BaseModel):
    variant: ModelVariant
    task: TaskType
    num_classes: int = Field(gt=0)
    freeze_backbone: bool = False
    pretrained: bool = True
    dropout_rate: float = Field(0.1, ge=0.0, le=1.0)
    use_head_norm: bool = True
    
    class Config:
        use_enum_values = True


class DataConfig(BaseModel):
    train_data_path: str
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True
    shuffle_train: bool = True
    image_size: int = Field(224, gt=0)
    mean: List[float] = Field([0.485, 0.456, 0.406])
    std: List[float] = Field([0.229, 0.224, 0.225])
    
    @validator('mean', 'std')
    def validate_normalization_stats(cls, v):
        if len(v) != 3:
            raise ValueError("Mean and std must have exactly 3 values for RGB channels")
        return v


class AugmentationConfig(BaseModel):
    use_augmentation: bool = True
    horizontal_flip_prob: float = Field(0.5, ge=0.0, le=1.0)
    rotation_degrees: int = Field(15, ge=0, le=180)
    color_jitter_brightness: float = Field(0.2, ge=0.0, le=1.0)
    color_jitter_contrast: float = Field(0.2, ge=0.0, le=1.0)
    color_jitter_saturation: float = Field(0.2, ge=0.0, le=1.0)
    color_jitter_hue: float = Field(0.1, ge=0.0, le=0.5)
    gaussian_blur_prob: float = Field(0.1, ge=0.0, le=1.0)
    normalize_before_augment: bool = False


class OptimizerConfig(BaseModel):
    type: OptimizerType
    learning_rate: float = Field(1e-4, gt=0.0)
    backbone_lr: Optional[float] = Field(None, gt=0.0)
    head_lr: Optional[float] = Field(None, gt=0.0)
    weight_decay: float = Field(0.01, ge=0.0)
    momentum: float = Field(0.9, ge=0.0, le=1.0)
    eps: float = Field(1e-8, gt=0.0)
    
    class Config:
        use_enum_values = True
    
    @root_validator
    def validate_differential_lr(cls, values):
        training_mode = values.get('training_mode')
        backbone_lr = values.get('backbone_lr')
        head_lr = values.get('head_lr')
        
        if training_mode == TrainingMode.FULL_FINE_TUNE:
            if backbone_lr is None or head_lr is None:
                raise ValueError("backbone_lr and head_lr must be specified for full fine-tuning")
        
        return values


class SchedulerConfig(BaseModel):
    type: SchedulerType
    step_size: Optional[int] = Field(None, gt=0)
    gamma: float = Field(0.1, gt=0.0, le=1.0)
    warmup_steps: Optional[int] = Field(None, ge=0)
    max_steps: Optional[int] = Field(None, gt=0)
    min_lr: float = Field(1e-6, ge=0.0)
    
    class Config:
        use_enum_values = True


class TrainingConfig(BaseModel):
    mode: TrainingMode
    max_epochs: int = Field(100, gt=0)
    early_stopping_patience: Optional[int] = Field(None, gt=0)
    save_every_n_epochs: int = Field(10, gt=0)
    validation_frequency: int = Field(1, gt=0)
    mixed_precision: bool = True
    gradient_accumulation_steps: int = Field(1, gt=0)
    max_grad_norm: Optional[float] = Field(None, gt=0.0)
    resume_from_checkpoint: Optional[str] = None
    
    optimizer: OptimizerConfig
    scheduler: Optional[SchedulerConfig] = None
    
    class Config:
        use_enum_values = True


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    log_frequency: int = Field(10, gt=0)
    save_top_k_models: int = Field(3, gt=0)


class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(["accuracy", "precision", "recall", "f1"])
    compute_confusion_matrix: bool = True
    save_predictions: bool = False
    batch_size: int = Field(32, gt=0)
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = ["accuracy", "precision", "recall", "f1", "auc", "top5_accuracy"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid options: {valid_metrics}")
        return v


class DeploymentConfig(BaseModel):
    export_format: Literal["onnx", "torchscript", "tensorrt"] = "onnx"
    quantization: bool = False
    optimization_level: Literal["basic", "advanced"] = "basic"
    batch_size: int = Field(1, gt=0)
    input_shape: List[int] = Field([3, 224, 224])


class ExperimentConfig(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    seed: int = Field(42, ge=0)
    output_dir: str = "./outputs"
    
    model: ModelConfig
    data: DataConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    deployment: Optional[DeploymentConfig] = None
    
    @root_validator
    def validate_experiment_consistency(cls, values):
        model_config = values.get('model')
        training_config = values.get('training')
        
        if model_config and training_config:
            if model_config.freeze_backbone and training_config.mode != TrainingMode.LINEAR_PROBE:
                raise ValueError("freeze_backbone=True is only compatible with linear_probe mode")
            
            if training_config.mode == TrainingMode.LINEAR_PROBE and not model_config.freeze_backbone:
                raise ValueError("linear_probe mode requires freeze_backbone=True")
        
        return values
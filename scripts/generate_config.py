#!/usr/bin/env python3
"""
DINOv3 Configuration Generation Helper

Interactive script for generating configuration files for training, evaluation,
and deployment based on user requirements and best practices.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.schemas import ModelVariant, TaskType, TrainingMode
from utils.logging import setup_logging


@dataclass
class ConfigTemplate:
    """Template for configuration generation."""
    name: str
    description: str
    use_cases: List[str]
    parameters: Dict[str, Any]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DINOv3 configuration files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./configs",
        help="Output directory for configuration files"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        help="Name for the generated configuration"
    )
    
    # Generation mode
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "template", "wizard"],
        help="Configuration generation mode"
    )
    parser.add_argument(
        "--template",
        type=str,
        choices=["medical_linear_probe", "satellite_fine_tune", "industrial_ssl", "natural_benchmark"],
        help="Pre-defined configuration template"
    )
    
    # Model settings
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=[v.value for v in ModelVariant],
        help="DINOv3 model variant"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=[t.value for t in TaskType],
        help="Task type"
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=[m.value for m in TrainingMode],
        help="Training mode"
    )
    
    # Dataset settings
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (height width)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["natural", "medical", "satellite", "industrial"],
        help="Domain type for data augmentations"
    )
    
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    # Hardware settings
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs available"
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        default=16,
        help="Available GPU memory in GB"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        type=str,
        default="yaml",
        choices=["yaml", "json"],
        help="Output configuration format"
    )
    parser.add_argument(
        "--split-configs",
        action="store_true",
        help="Generate separate files for model, training, and data configs"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def get_model_recommendations(
    task_type: str,
    num_classes: int,
    memory_gb: int,
    performance_priority: str = "balanced"
) -> Dict[str, Any]:
    """Get model variant recommendations based on requirements."""
    
    # Memory requirements for different variants (approximate)
    memory_requirements = {
        ModelVariant.VIT_S16.value: 2.1,
        ModelVariant.VIT_B16.value: 3.4,
        ModelVariant.VIT_L16.value: 6.2,
        ModelVariant.VIT_H16_PLUS.value: 12.8,
        ModelVariant.CONVNEXT_TINY.value: 2.8,
        ModelVariant.CONVNEXT_BASE.value: 4.1,
    }
    
    # Filter variants by memory constraint
    suitable_variants = [
        variant for variant, mem_req in memory_requirements.items()
        if mem_req <= memory_gb
    ]
    
    if not suitable_variants:
        suitable_variants = [ModelVariant.VIT_S16.value]  # Fallback to smallest
    
    # Recommend based on performance priority
    if performance_priority == "speed":
        recommended = ModelVariant.VIT_S16.value
    elif performance_priority == "accuracy":
        recommended = max(suitable_variants, key=lambda v: memory_requirements[v])
    else:  # balanced
        recommended = ModelVariant.VIT_B16.value if ModelVariant.VIT_B16.value in suitable_variants else suitable_variants[0]
    
    return {
        'recommended': recommended,
        'alternatives': [v for v in suitable_variants if v != recommended],
        'memory_usage': memory_requirements[recommended]
    }


def get_training_recommendations(
    model_variant: str,
    training_mode: str,
    num_classes: int,
    domain: str,
    num_gpus: int = 1
) -> Dict[str, Any]:
    """Get training hyperparameter recommendations."""
    
    # Base recommendations by model size
    base_configs = {
        ModelVariant.VIT_S16.value: {
            'learning_rate': 1e-4,
            'batch_size': 64,
            'warmup_epochs': 5,
            'weight_decay': 0.05
        },
        ModelVariant.VIT_B16.value: {
            'learning_rate': 5e-5,
            'batch_size': 32,
            'warmup_epochs': 10,
            'weight_decay': 0.05
        },
        ModelVariant.VIT_L16.value: {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'warmup_epochs': 15,
            'weight_decay': 0.1
        },
        ModelVariant.VIT_H16_PLUS.value: {
            'learning_rate': 1e-5,
            'batch_size': 8,
            'warmup_epochs': 20,
            'weight_decay': 0.1
        }
    }
    
    config = base_configs.get(model_variant, base_configs[ModelVariant.VIT_B16.value]).copy()
    
    # Adjust for training mode
    if training_mode == TrainingMode.LINEAR_PROBE.value:
        config['learning_rate'] *= 10  # Higher LR for linear probe
        config['epochs'] = 100
        config['freeze_backbone'] = True
    elif training_mode == TrainingMode.FULL_FINE_TUNE.value:
        config['epochs'] = 50
        config['freeze_backbone'] = False
        config['differential_lr'] = True
        config['backbone_lr_ratio'] = 0.1
    elif training_mode == TrainingMode.SSL_CONTINUE.value:
        config['learning_rate'] *= 0.5  # Lower LR for SSL
        config['epochs'] = 200
        config['freeze_backbone'] = False
    
    # Adjust for domain
    if domain == "medical":
        config['learning_rate'] *= 0.5  # More conservative for medical
        config['weight_decay'] *= 1.5
    elif domain == "satellite":
        config['batch_size'] = min(config['batch_size'], 16)  # Often high-res images
    
    # Scale batch size with GPUs
    if num_gpus > 1:
        config['batch_size'] *= num_gpus
        config['learning_rate'] *= num_gpus ** 0.5  # Scale learning rate
    
    return config


def get_augmentation_recommendations(domain: str, task_type: str) -> Dict[str, Any]:
    """Get data augmentation recommendations by domain."""
    
    base_augmentations = {
        'resize_shortest': True,
        'center_crop': True,
        'horizontal_flip': True,
        'normalize': True
    }
    
    if domain == "natural":
        augmentations = {
            **base_augmentations,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1},
            'random_rotation': 15,
            'random_perspective': 0.1,
            'gaussian_blur': 0.1
        }
    
    elif domain == "medical":
        # Conservative augmentations for medical images
        augmentations = {
            **base_augmentations,
            'horizontal_flip': False,  # Anatomy orientation matters
            'color_jitter': {'brightness': 0.1, 'contrast': 0.1},
            'random_rotation': 5,
            'gaussian_noise': 0.05
        }
    
    elif domain == "satellite":
        augmentations = {
            **base_augmentations,
            'color_jitter': {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.2},
            'random_rotation': 30,  # Satellite images can be rotated
            'gaussian_blur': 0.15,
            'random_perspective': 0.2
        }
    
    elif domain == "industrial":
        augmentations = {
            **base_augmentations,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.3},
            'random_rotation': 10,
            'gaussian_noise': 0.1,
            'motion_blur': 0.1
        }
    
    else:
        augmentations = base_augmentations
    
    return augmentations


def create_config_template(
    name: str,
    model_variant: str,
    task_type: str,
    training_mode: str,
    num_classes: int,
    image_size: List[int],
    domain: str,
    training_config: Dict[str, Any],
    augmentation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a complete configuration template."""
    
    config = {
        'name': name,
        'description': f'DINOv3 {training_mode} configuration for {domain} {task_type}',
        'seed': 42,
        
        'model': {
            'variant': model_variant,
            'task_type': task_type,
            'num_classes': num_classes,
            'pretrained': True,
            'dropout': 0.1 if task_type == TaskType.CLASSIFICATION.value else 0.0
        },
        
        'data': {
            'image_size': image_size,
            'batch_size': training_config['batch_size'],
            'num_workers': 4,
            'pin_memory': True,
            'shuffle_train': True,
            'train_data_path': 'path/to/train/data',
            'val_data_path': 'path/to/val/data'
        },
        
        'training': {
            'mode': training_mode,
            'epochs': training_config['epochs'],
            'learning_rate': training_config['learning_rate'],
            'weight_decay': training_config['weight_decay'],
            'warmup_epochs': training_config['warmup_epochs'],
            'freeze_backbone': training_config.get('freeze_backbone', False),
            'mixed_precision': True,
            'gradient_clipping': 1.0,
            'early_stopping': {
                'patience': 10,
                'min_delta': 1e-4
            }
        },
        
        'optimizer': {
            'name': 'adamw',
            'betas': [0.9, 0.999],
            'eps': 1e-8
        },
        
        'scheduler': {
            'name': 'cosine_annealing',
            'min_lr': 1e-6,
            'warmup_type': 'linear'
        },
        
        'augmentation': {
            'domain': domain,
            'train': augmentation_config,
            'val': {
                'resize_shortest': True,
                'center_crop': True,
                'normalize': True
            }
        },
        
        'logging': {
            'log_interval': 10,
            'save_interval': 5,
            'use_tensorboard': True,
            'use_wandb': False,
            'wandb_project': f'dinov3-{domain}',
            'wandb_entity': None
        },
        
        'evaluation': {
            'metrics': ['accuracy', 'top5_accuracy'] if task_type == TaskType.CLASSIFICATION.value else ['miou', 'pixel_accuracy'],
            'save_predictions': False,
            'visualize_predictions': True
        }
    }
    
    # Add differential learning rates for full fine-tuning
    if training_config.get('differential_lr', False):
        config['training']['differential_lr'] = {
            'backbone_lr_ratio': training_config.get('backbone_lr_ratio', 0.1),
            'head_lr_ratio': 1.0
        }
    
    return config


def get_predefined_templates() -> Dict[str, ConfigTemplate]:
    """Get predefined configuration templates."""
    
    templates = {
        'medical_linear_probe': ConfigTemplate(
            name='medical_linear_probe',
            description='Linear probing for medical image classification',
            use_cases=['X-ray classification', 'Skin lesion detection', 'Retinal disease diagnosis'],
            parameters={
                'model_variant': ModelVariant.VIT_B16.value,
                'task_type': TaskType.CLASSIFICATION.value,
                'training_mode': TrainingMode.LINEAR_PROBE.value,
                'domain': 'medical',
                'num_classes': 5,
                'image_size': [224, 224],
                'batch_size': 32,
                'learning_rate': 1e-3,
                'epochs': 100
            }
        ),
        
        'satellite_fine_tune': ConfigTemplate(
            name='satellite_fine_tune',
            description='Full fine-tuning for satellite image classification',
            use_cases=['Land use classification', 'Crop monitoring', 'Urban planning'],
            parameters={
                'model_variant': ModelVariant.VIT_L16.value,
                'task_type': TaskType.CLASSIFICATION.value,
                'training_mode': TrainingMode.FULL_FINE_TUNE.value,
                'domain': 'satellite',
                'num_classes': 10,
                'image_size': [256, 256],
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 50
            }
        ),
        
        'industrial_ssl': ConfigTemplate(
            name='industrial_ssl',
            description='Self-supervised learning for industrial inspection',
            use_cases=['Defect detection', 'Quality control', 'Anomaly detection'],
            parameters={
                'model_variant': ModelVariant.VIT_B16.value,
                'task_type': TaskType.CLASSIFICATION.value,
                'training_mode': TrainingMode.SSL_CONTINUE.value,
                'domain': 'industrial',
                'num_classes': 2,
                'image_size': [224, 224],
                'batch_size': 32,
                'learning_rate': 2e-5,
                'epochs': 200
            }
        ),
        
        'natural_benchmark': ConfigTemplate(
            name='natural_benchmark',
            description='Benchmark configuration for natural images',
            use_cases=['ImageNet classification', 'CIFAR benchmarking', 'Transfer learning baseline'],
            parameters={
                'model_variant': ModelVariant.VIT_B16.value,
                'task_type': TaskType.CLASSIFICATION.value,
                'training_mode': TrainingMode.FULL_FINE_TUNE.value,
                'domain': 'natural',
                'num_classes': 1000,
                'image_size': [224, 224],
                'batch_size': 64,
                'learning_rate': 5e-5,
                'epochs': 50
            }
        )
    }
    
    return templates


def interactive_config_generation() -> Dict[str, Any]:
    """Interactive configuration generation with user input."""
    print("=== DINOv3 Configuration Generator ===")
    print("Let's create a configuration tailored to your needs.\n")
    
    # Basic information
    config_name = input("Configuration name: ").strip() or "custom_dinov3"
    
    # Task type
    print("\nTask types:")
    for i, task in enumerate(TaskType, 1):
        print(f"  {i}. {task.value}")
    
    task_choice = int(input("Select task type (1-2): ") or 1)
    task_type = list(TaskType)[task_choice - 1].value
    
    # Domain
    print("\nDomains:")
    domains = ["natural", "medical", "satellite", "industrial"]
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")
    
    domain_choice = int(input("Select domain (1-4): ") or 1)
    domain = domains[domain_choice - 1]
    
    # Number of classes
    num_classes = int(input(f"Number of classes in your {task_type} task: ") or 10)
    
    # Training mode
    print("\nTraining modes:")
    for i, mode in enumerate(TrainingMode, 1):
        print(f"  {i}. {mode.value}")
    
    mode_choice = int(input("Select training mode (1-3): ") or 1)
    training_mode = list(TrainingMode)[mode_choice - 1].value
    
    # Hardware constraints
    memory_gb = int(input("Available GPU memory (GB): ") or 16)
    num_gpus = int(input("Number of GPUs: ") or 1)
    
    # Performance priority
    print("\nPerformance priorities:")
    priorities = ["speed", "balanced", "accuracy"]
    for i, priority in enumerate(priorities, 1):
        print(f"  {i}. {priority}")
    
    priority_choice = int(input("Select priority (1-3): ") or 2)
    performance_priority = priorities[priority_choice - 1]
    
    # Get recommendations
    model_rec = get_model_recommendations(task_type, num_classes, memory_gb, performance_priority)
    training_rec = get_training_recommendations(
        model_rec['recommended'], training_mode, num_classes, domain, num_gpus
    )
    augmentation_rec = get_augmentation_recommendations(domain, task_type)
    
    print(f"\nRecommendations:")
    print(f"  Model: {model_rec['recommended']} (Memory: {model_rec['memory_usage']:.1f}GB)")
    print(f"  Batch size: {training_rec['batch_size']}")
    print(f"  Learning rate: {training_rec['learning_rate']}")
    print(f"  Epochs: {training_rec['epochs']}")
    
    # Create configuration
    config = create_config_template(
        config_name,
        model_rec['recommended'],
        task_type,
        training_mode,
        num_classes,
        [224, 224],
        domain,
        training_rec,
        augmentation_rec
    )
    
    return config


def generate_from_template(template_name: str, **overrides) -> Dict[str, Any]:
    """Generate configuration from predefined template."""
    templates = get_predefined_templates()
    
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = templates[template_name]
    params = template.parameters.copy()
    
    # Apply overrides
    params.update(overrides)
    
    # Get recommendations
    training_rec = get_training_recommendations(
        params['model_variant'],
        params['training_mode'],
        params['num_classes'],
        params['domain'],
        params.get('num_gpus', 1)
    )
    
    # Update with template-specific values
    training_rec.update({
        'batch_size': params['batch_size'],
        'learning_rate': params['learning_rate'],
        'epochs': params['epochs']
    })
    
    augmentation_rec = get_augmentation_recommendations(params['domain'], params['task_type'])
    
    # Create configuration
    config = create_config_template(
        template.name,
        params['model_variant'],
        params['task_type'],
        params['training_mode'],
        params['num_classes'],
        params['image_size'],
        params['domain'],
        training_rec,
        augmentation_rec
    )
    
    return config


def save_config(
    config: Dict[str, Any],
    output_path: Path,
    format_type: str = "yaml",
    split_configs: bool = False
):
    """Save configuration to file(s)."""
    
    if split_configs:
        # Save as separate files
        model_config = {'model': config['model']}
        training_config = {
            'training': config['training'],
            'optimizer': config['optimizer'],
            'scheduler': config['scheduler']
        }
        data_config = {
            'data': config['data'],
            'augmentation': config['augmentation']
        }
        
        configs_to_save = [
            (model_config, output_path.parent / f"{config['name']}_model.{format_type}"),
            (training_config, output_path.parent / f"{config['name']}_training.{format_type}"),
            (data_config, output_path.parent / f"{config['name']}_data.{format_type}")
        ]
    else:
        # Save as single file
        configs_to_save = [(config, output_path)]
    
    for cfg, path in configs_to_save:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "yaml":
            with open(path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, indent=2)
        else:  # json
            with open(path, 'w') as f:
                json.dump(cfg, f, indent=2)
        
        print(f"Configuration saved: {path}")


def main():
    """Main configuration generation function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(
        name="dinov3_config_generator",
        level="DEBUG" if args.debug else "INFO"
    )
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate configuration based on mode
        if args.mode == "interactive":
            config = interactive_config_generation()
        
        elif args.mode == "template":
            if not args.template:
                print("Available templates:")
                templates = get_predefined_templates()
                for name, template in templates.items():
                    print(f"  {name}: {template.description}")
                return
            
            # Collect overrides from command line arguments
            overrides = {}
            if args.model_variant:
                overrides['model_variant'] = args.model_variant
            if args.task_type:
                overrides['task_type'] = args.task_type
            if args.training_mode:
                overrides['training_mode'] = args.training_mode
            if args.num_classes:
                overrides['num_classes'] = args.num_classes
            if args.domain:
                overrides['domain'] = args.domain
            if args.batch_size:
                overrides['batch_size'] = args.batch_size
            if args.learning_rate:
                overrides['learning_rate'] = args.learning_rate
            if args.epochs:
                overrides['epochs'] = args.epochs
            
            config = generate_from_template(args.template, **overrides)
        
        elif args.mode == "wizard":
            # TODO: Implement step-by-step wizard
            print("Wizard mode not yet implemented, falling back to interactive mode")
            config = interactive_config_generation()
        
        # Determine output filename
        config_name = args.config_name or config['name']
        if args.split_configs:
            output_path = output_dir / f"{config_name}_main.{args.format}"
        else:
            output_path = output_dir / f"{config_name}.{args.format}"
        
        # Save configuration
        save_config(config, output_path, args.format, args.split_configs)
        
        # Print summary
        print(f"\nConfiguration Summary:")
        print(f"  Name: {config['name']}")
        print(f"  Model: {config['model']['variant']}")
        print(f"  Task: {config['model']['task_type']}")
        print(f"  Mode: {config['training']['mode']}")
        print(f"  Domain: {config['augmentation']['domain']}")
        print(f"  Classes: {config['model']['num_classes']}")
        print(f"  Batch size: {config['data']['batch_size']}")
        print(f"  Learning rate: {config['training']['learning_rate']}")
        print(f"  Epochs: {config['training']['epochs']}")
        
        logger.info("Configuration generation completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("Configuration generation interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Configuration generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
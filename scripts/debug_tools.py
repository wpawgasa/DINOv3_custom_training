#!/usr/bin/env python3
"""
DINOv3 Development and Debugging Tools

Collection of utilities for debugging training issues, profiling performance,
and analyzing model behavior during development.
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import json
import psutil
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.profiler as profiler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from models.model_factory import create_model, get_model_factory
from data.dataset import create_dataset
from data.augmentations import create_transforms
from utils.config import load_hierarchical_config
from utils.schemas import ExperimentConfig
from utils.logging import setup_logging
from utils.device import get_device_manager


class MemoryTracker:
    """Track GPU and CPU memory usage over time."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.measurements = []
        self.running = False
        self.thread = None
        
    def start(self, interval: float = 1.0):
        """Start memory tracking."""
        self.running = True
        self.measurements = []
        self.thread = threading.Thread(target=self._track_memory, args=(interval,))
        self.thread.start()
        
    def stop(self):
        """Stop memory tracking."""
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _track_memory(self, interval: float):
        """Internal memory tracking loop."""
        while self.running:
            measurement = {
                'timestamp': time.time(),
                'cpu_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                'cpu_memory_percent': psutil.virtual_memory().percent
            }
            
            if self.device == 'cuda' and torch.cuda.is_available():
                measurement.update({
                    'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                    'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                    'gpu_memory_percent': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
                })
            
            self.measurements.append(measurement)
            time.sleep(interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.measurements:
            return {}
        
        cpu_usage = [m['cpu_memory_mb'] for m in self.measurements]
        stats = {
            'cpu_memory': {
                'min_mb': min(cpu_usage),
                'max_mb': max(cpu_usage),
                'avg_mb': np.mean(cpu_usage),
                'peak_percent': max(m['cpu_memory_percent'] for m in self.measurements)
            }
        }
        
        if self.device == 'cuda' and 'gpu_memory_allocated_mb' in self.measurements[0]:
            gpu_usage = [m['gpu_memory_allocated_mb'] for m in self.measurements]
            gpu_reserved = [m['gpu_memory_reserved_mb'] for m in self.measurements]
            
            stats['gpu_memory'] = {
                'min_allocated_mb': min(gpu_usage),
                'max_allocated_mb': max(gpu_usage),
                'avg_allocated_mb': np.mean(gpu_usage),
                'max_reserved_mb': max(gpu_reserved),
                'peak_percent': max(m['gpu_memory_percent'] for m in self.measurements)
            }
        
        return stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DINOv3 debugging and development tools",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tool selection
    parser.add_argument(
        "--tool",
        type=str,
        required=True,
        choices=[
            "model_summary",
            "data_loader_test", 
            "memory_profile",
            "training_step_debug",
            "gradient_analysis",
            "benchmark_inference",
            "config_validate",
            "dataset_analysis",
            "profiler"
        ],
        help="Debug tool to run"
    )
    
    # Common arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        help="Path to dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./debug_output",
        help="Output directory for debug results"
    )
    
    # Tool-specific arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to process"
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=100,
        help="Number of steps to profile"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use"
    )
    
    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save intermediate outputs"
    )
    
    return parser.parse_args()


def model_summary_tool(config_path: str, output_dir: Path, verbose: bool = False):
    """Generate detailed model summary and architecture analysis."""
    print("=== Model Summary Tool ===")
    
    # Load configuration
    config_dict = load_hierarchical_config(base_config=config_path)
    config = ExperimentConfig(**config_dict)
    
    # Create model
    model = create_model(config.model)
    model.eval()
    
    # Basic model info
    model_info = model.get_model_info()
    print(f"\nModel: {model_info['variant']}")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.1f} MB")
    
    # Detailed layer analysis
    if verbose:
        print(f"\nDetailed Architecture:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                print(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    # Parameter distribution
    param_sizes = []
    param_names = []
    for name, param in model.named_parameters():
        param_sizes.append(param.numel())
        param_names.append(name)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameter distribution histogram
    ax1.hist(param_sizes, bins=20, alpha=0.7)
    ax1.set_xlabel('Parameter Count')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Parameter Distribution Across Layers')
    ax1.set_yscale('log')
    
    # Top 10 largest layers
    top_layers = sorted(zip(param_names, param_sizes), key=lambda x: x[1], reverse=True)[:10]
    names, sizes = zip(*top_layers)
    
    ax2.barh(range(len(names)), sizes)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([n.split('.')[-1] for n in names])
    ax2.set_xlabel('Parameter Count')
    ax2.set_title('Top 10 Largest Layers')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Model analysis saved to: {output_dir / 'model_analysis.png'}")
    
    # Save detailed summary
    summary = {
        'model_info': model_info,
        'config': OmegaConf.to_container(config, resolve=True),
        'layer_details': [
            {'name': name, 'type': module.__class__.__name__, 'parameters': sum(p.numel() for p in module.parameters())}
            for name, module in model.named_modules()
            if len(list(module.children())) == 0
        ],
        'parameter_stats': {
            'total_layers': len(param_names),
            'mean_params_per_layer': np.mean(param_sizes),
            'std_params_per_layer': np.std(param_sizes),
            'largest_layer': max(top_layers, key=lambda x: x[1])
        }
    }
    
    with open(output_dir / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def data_loader_test_tool(config_path: str, data_path: str, output_dir: Path, batch_size: int, num_batches: int):
    """Test data loader performance and integrity."""
    print("=== Data Loader Test Tool ===")
    
    # Load configuration
    config_dict = load_hierarchical_config(base_config=config_path)
    config = ExperimentConfig(**config_dict)
    
    # Create transforms
    transform_manager = create_transforms(
        domain=config.augmentation.get('domain', 'natural'),
        image_size=config.data.image_size
    )
    
    # Create dataset
    dataset = create_dataset(
        data_path=data_path,
        annotation_format="imagefolder",
        transform=transform_manager.get_train_transform(),
        cache_images=False
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test data loading
    loading_times = []
    batch_stats = []
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        batch_start = time.time()
        
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets, _ = batch
        
        # Analyze batch
        batch_info = {
            'batch_idx': i,
            'batch_size': inputs.shape[0],
            'input_shape': list(inputs.shape),
            'input_dtype': str(inputs.dtype),
            'target_shape': list(targets.shape) if hasattr(targets, 'shape') else len(targets),
            'input_min': float(inputs.min()),
            'input_max': float(inputs.max()),
            'input_mean': float(inputs.mean()),
            'input_std': float(inputs.std()),
            'loading_time': time.time() - batch_start
        }
        
        batch_stats.append(batch_info)
        loading_times.append(batch_info['loading_time'])
        
        if i % 5 == 0:
            print(f"Batch {i}: {batch_info['loading_time']:.3f}s")
    
    total_time = time.time() - start_time
    
    # Generate statistics
    stats = {
        'dataset_info': {
            'total_samples': len(dataset),
            'num_classes': len(dataset.classes) if hasattr(dataset, 'classes') else 'unknown',
            'classes': dataset.classes if hasattr(dataset, 'classes') else None
        },
        'loading_performance': {
            'total_batches_processed': len(batch_stats),
            'total_time': total_time,
            'avg_loading_time': np.mean(loading_times),
            'min_loading_time': np.min(loading_times),
            'max_loading_time': np.max(loading_times),
            'std_loading_time': np.std(loading_times),
            'throughput_samples_per_sec': sum(b['batch_size'] for b in batch_stats) / total_time
        },
        'data_statistics': {
            'avg_input_min': np.mean([b['input_min'] for b in batch_stats]),
            'avg_input_max': np.mean([b['input_max'] for b in batch_stats]),
            'avg_input_mean': np.mean([b['input_mean'] for b in batch_stats]),
            'avg_input_std': np.mean([b['input_std'] for b in batch_stats])
        },
        'batch_details': batch_stats
    }
    
    print(f"\nData Loader Performance:")
    print(f"  Average loading time: {stats['loading_performance']['avg_loading_time']:.3f}s")
    print(f"  Throughput: {stats['loading_performance']['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Input range: [{stats['data_statistics']['avg_input_min']:.3f}, {stats['data_statistics']['avg_input_max']:.3f}]")
    
    # Save results
    with open(output_dir / 'dataloader_test.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Plot loading times
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loading_times)
    plt.xlabel('Batch Index')
    plt.ylabel('Loading Time (s)')
    plt.title('Data Loading Times')
    
    plt.subplot(1, 2, 2)
    plt.hist(loading_times, bins=20, alpha=0.7)
    plt.xlabel('Loading Time (s)')
    plt.ylabel('Frequency')
    plt.title('Loading Time Distribution')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataloader_performance.png', dpi=150)
    print(f"Performance plots saved to: {output_dir / 'dataloader_performance.png'}")


def memory_profile_tool(config_path: str, model_path: Optional[str], output_dir: Path, device: str):
    """Profile memory usage during model operations."""
    print("=== Memory Profile Tool ===")
    
    # Setup device
    device_manager = get_device_manager()
    if device != "auto":
        device_manager.set_device(device)
    
    print(f"Using device: {device_manager.device}")
    
    # Load configuration
    config_dict = load_hierarchical_config(base_config=config_path)
    config = ExperimentConfig(**config_dict)
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker(device_manager.device.type)
    
    # Start tracking
    memory_tracker.start(interval=0.1)
    
    try:
        print("Creating model...")
        model = create_model(config.model)
        model = model.to(device_manager.device)
        
        # Load weights if provided
        if model_path:
            print("Loading model weights...")
            checkpoint = torch.load(model_path, map_location=device_manager.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
        
        # Test forward pass with different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        memory_results = {}
        
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch size {batch_size}...")
                
                # Clear cache
                if device_manager.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Create dummy input
                dummy_input = torch.randn(
                    batch_size, 3, 
                    config.data.image_size[0], 
                    config.data.image_size[1]
                ).to(device_manager.device)
                
                # Forward pass
                model.eval()
                with torch.no_grad():
                    output = model(dummy_input)
                
                # Record memory usage
                if device_manager.device.type == 'cuda':
                    memory_results[batch_size] = {
                        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
                    }
                else:
                    memory_results[batch_size] = {
                        'cpu_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
                    }
                
                print(f"  Memory usage: {memory_results[batch_size]}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  Out of memory at batch size {batch_size}")
                    memory_results[batch_size] = {'error': 'OOM'}
                    break
                else:
                    raise
    
    finally:
        memory_tracker.stop()
    
    # Get memory tracking stats
    tracking_stats = memory_tracker.get_stats()
    
    # Combine results
    profile_results = {
        'device': str(device_manager.device),
        'model_variant': config.model.variant,
        'input_size': config.data.image_size,
        'batch_size_analysis': memory_results,
        'continuous_tracking': tracking_stats,
        'measurements': memory_tracker.measurements
    }
    
    # Save results
    with open(output_dir / 'memory_profile.json', 'w') as f:
        json.dump(profile_results, f, indent=2, default=str)
    
    # Create memory usage plots
    if memory_tracker.measurements:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU memory over time
        times = [m['timestamp'] - memory_tracker.measurements[0]['timestamp'] for m in memory_tracker.measurements]
        cpu_memory = [m['cpu_memory_mb'] for m in memory_tracker.measurements]
        
        axes[0, 0].plot(times, cpu_memory)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('CPU Memory (MB)')
        axes[0, 0].set_title('CPU Memory Usage Over Time')
        
        # GPU memory over time (if available)
        if 'gpu_memory_allocated_mb' in memory_tracker.measurements[0]:
            gpu_memory = [m['gpu_memory_allocated_mb'] for m in memory_tracker.measurements]
            axes[0, 1].plot(times, gpu_memory)
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('GPU Memory (MB)')
            axes[0, 1].set_title('GPU Memory Usage Over Time')
        
        # Memory vs batch size
        if memory_results and device_manager.device.type == 'cuda':
            valid_results = {k: v for k, v in memory_results.items() if 'error' not in v}
            if valid_results:
                batch_sizes_plot = list(valid_results.keys())
                allocated_mb = [v['allocated_mb'] for v in valid_results.values()]
                
                axes[1, 0].plot(batch_sizes_plot, allocated_mb, 'o-')
                axes[1, 0].set_xlabel('Batch Size')
                axes[1, 0].set_ylabel('GPU Memory (MB)')
                axes[1, 0].set_title('Memory Usage vs Batch Size')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'memory_profile.png', dpi=150)
        print(f"Memory profile plots saved to: {output_dir / 'memory_profile.png'}")
    
    print("\nMemory Profile Summary:")
    if tracking_stats.get('gpu_memory'):
        gpu_stats = tracking_stats['gpu_memory']
        print(f"  Peak GPU memory: {gpu_stats['max_allocated_mb']:.1f} MB")
        print(f"  Average GPU memory: {gpu_stats['avg_allocated_mb']:.1f} MB")
    
    if tracking_stats.get('cpu_memory'):
        cpu_stats = tracking_stats['cpu_memory']
        print(f"  Peak CPU memory: {cpu_stats['max_mb']:.1f} MB")
        print(f"  Average CPU memory: {cpu_stats['avg_mb']:.1f} MB")


def config_validate_tool(config_path: str, output_dir: Path):
    """Validate configuration file and check for potential issues."""
    print("=== Configuration Validation Tool ===")
    
    try:
        # Load and validate configuration
        config_dict = load_hierarchical_config(base_config=config_path)
        config = ExperimentConfig(**config_dict)
        
        print("✓ Configuration loaded and validated successfully")
        
        # Check for potential issues
        issues = []
        warnings = []
        
        # Model checks
        model_factory = get_model_factory()
        if config.model.variant not in [v.value for v in ModelVariant]:
            issues.append(f"Unknown model variant: {config.model.variant}")
        
        # Training checks
        if config.training.learning_rate > 1e-2:
            warnings.append(f"Learning rate seems high: {config.training.learning_rate}")
        
        if config.training.epochs > 1000:
            warnings.append(f"Very high epoch count: {config.training.epochs}")
        
        # Data checks  
        if config.data.batch_size > 256:
            warnings.append(f"Very large batch size: {config.data.batch_size}")
        
        if config.data.num_workers > 16:
            warnings.append(f"High number of workers: {config.data.num_workers}")
        
        # Memory estimation
        estimated_memory = estimate_memory_requirements(config)
        
        validation_results = {
            'config_path': config_path,
            'validation_status': 'passed',
            'config': OmegaConf.to_container(config, resolve=True),
            'issues': issues,
            'warnings': warnings,
            'memory_estimation': estimated_memory
        }
        
        # Print summary
        print(f"\nValidation Results:")
        print(f"  Status: {'PASSED' if not issues else 'FAILED'}")
        print(f"  Issues: {len(issues)}")
        print(f"  Warnings: {len(warnings)}")
        
        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"  ❌ {issue}")
        
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        print(f"\nEstimated Memory Requirements:")
        print(f"  Model size: {estimated_memory['model_size_mb']:.1f} MB")
        print(f"  Batch memory: {estimated_memory['batch_memory_mb']:.1f} MB")
        print(f"  Total estimated: {estimated_memory['total_memory_mb']:.1f} MB")
        
    except Exception as e:
        validation_results = {
            'config_path': config_path,
            'validation_status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(f"❌ Configuration validation failed: {e}")
    
    # Save results
    with open(output_dir / 'config_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)


def estimate_memory_requirements(config: ExperimentConfig) -> Dict[str, float]:
    """Estimate memory requirements for the configuration."""
    
    # Model size estimates (approximate, in MB)
    model_sizes = {
        'dinov3_vits16': 86,
        'dinov3_vitb16': 345,
        'dinov3_vitl16': 1200,
        'dinov3_vith16plus': 3400,
        'dinov3_convnext_tiny': 115,
        'dinov3_convnext_base': 356
    }
    
    model_size_mb = model_sizes.get(config.model.variant, 345)  # Default to ViT-B
    
    # Estimate batch memory (image + gradient + activations)
    h, w = config.data.image_size
    batch_size = config.data.batch_size
    
    # Image memory: batch_size * 3 * H * W * 4 bytes (float32)
    image_memory_mb = batch_size * 3 * h * w * 4 / (1024 * 1024)
    
    # Rough estimate for activations and gradients (3x model size)
    activation_memory_mb = model_size_mb * 3
    
    # Optimizer states (roughly 2x parameters for AdamW)
    optimizer_memory_mb = model_size_mb * 2
    
    total_memory_mb = model_size_mb + image_memory_mb + activation_memory_mb + optimizer_memory_mb
    
    return {
        'model_size_mb': model_size_mb,
        'batch_memory_mb': image_memory_mb,
        'activation_memory_mb': activation_memory_mb,
        'optimizer_memory_mb': optimizer_memory_mb,
        'total_memory_mb': total_memory_mb
    }


def main():
    """Main debugging tool function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(
        name="dinov3_debug_tools",
        level="DEBUG" if args.verbose else "INFO"
    )
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run selected tool
        if args.tool == "model_summary":
            if not args.config:
                raise ValueError("--config required for model_summary tool")
            model_summary_tool(args.config, output_dir, args.verbose)
            
        elif args.tool == "data_loader_test":
            if not args.config or not args.data_path:
                raise ValueError("--config and --data-path required for data_loader_test tool")
            data_loader_test_tool(args.config, args.data_path, output_dir, args.batch_size, args.num_batches)
            
        elif args.tool == "memory_profile":
            if not args.config:
                raise ValueError("--config required for memory_profile tool")
            memory_profile_tool(args.config, args.model_path, output_dir, args.device)
            
        elif args.tool == "config_validate":
            if not args.config:
                raise ValueError("--config required for config_validate tool")
            config_validate_tool(args.config, output_dir)
            
        else:
            print(f"Tool '{args.tool}' not yet implemented")
        
        print(f"\nDebug results saved to: {output_dir}")
        logger.info(f"Debug tool '{args.tool}' completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Debug tool interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Debug tool failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
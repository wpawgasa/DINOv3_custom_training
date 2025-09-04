#!/usr/bin/env python3
"""
DINOv3 Model Conversion Utility

Script for converting DINOv3 models between different formats and optimizing
them for deployment (ONNX, TensorRT, quantization, etc.).
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from omegaconf import OmegaConf

from models.model_factory import create_model
from utils.config import load_hierarchical_config
from utils.schemas import ExperimentConfig
from utils.logging import setup_logging
from utils.device import get_device_manager


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert DINOv3 models to different formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input model
    parser.add_argument(
        "--input-model", "-i",
        type=str,
        required=True,
        help="Path to input model checkpoint"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to model configuration file (optional, will try to load from checkpoint)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./converted_models",
        help="Output directory for converted models"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output model name (default: derived from input)"
    )
    
    # Conversion targets
    parser.add_argument(
        "--format",
        type=str,
        nargs="*",
        default=["onnx"],
        choices=["onnx", "torchscript", "tensorrt", "coreml", "tflite"],
        help="Target conversion formats"
    )
    
    # Optimization options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization"
    )
    parser.add_argument(
        "--quantization-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static", "qat"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--optimize-for-mobile",
        action="store_true",
        help="Optimize model for mobile deployment"
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Use half precision (FP16)"
    )
    
    # Model pruning
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Apply model pruning"
    )
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        default=0.2,
        help="Pruning ratio (fraction of parameters to remove)"
    )
    parser.add_argument(
        "--structured-pruning",
        action="store_true",
        help="Use structured pruning instead of unstructured"
    )
    
    # Input specifications for conversion
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (height width)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for conversion"
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Support dynamic batch size in converted model"
    )
    
    # Validation and benchmarking
    parser.add_argument(
        "--validate-conversion",
        action="store_true",
        help="Validate converted model outputs"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark converted model performance"
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        help="Path to calibration dataset for static quantization"
    )
    
    # Debug and development
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def load_model_for_conversion(
    model_path: str,
    config_path: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and configuration for conversion."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        print("Using configuration from checkpoint")
    elif config_path:
        config_dict = load_hierarchical_config(base_config=config_path)
        print(f"Using configuration from: {config_path}")
    else:
        raise ValueError("No configuration found in checkpoint and no config file provided")
    
    # Create experiment config
    config = ExperimentConfig(**config_dict)
    
    # Create model
    model = create_model(config.model)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel prefixes
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Model loaded successfully")
    return model, config_dict


def quantize_model(
    model: nn.Module,
    quantization_mode: str,
    calibration_data: Optional[str] = None
) -> nn.Module:
    """Apply quantization to the model."""
    print(f"Applying {quantization_mode} quantization...")
    
    if quantization_mode == "dynamic":
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
    
    elif quantization_mode == "static":
        # Static quantization (requires calibration data)
        if not calibration_data:
            raise ValueError("Static quantization requires calibration data")
        
        # TODO: Implement static quantization with calibration
        print("Static quantization not yet implemented")
        quantized_model = model
    
    elif quantization_mode == "qat":
        # Quantization Aware Training (would require retraining)
        print("QAT requires retraining - returning original model")
        quantized_model = model
    
    else:
        raise ValueError(f"Unknown quantization mode: {quantization_mode}")
    
    return quantized_model


def prune_model(
    model: nn.Module,
    pruning_ratio: float,
    structured: bool = False
) -> nn.Module:
    """Apply pruning to the model."""
    print(f"Applying {'structured' if structured else 'unstructured'} pruning (ratio: {pruning_ratio})...")
    
    try:
        import torch.nn.utils.prune as prune
    except ImportError:
        print("Pruning not available in this PyTorch version")
        return model
    
    # Identify modules to prune
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            modules_to_prune.append((module, 'weight'))
    
    if structured:
        # Structured pruning (prune entire channels/filters)
        for module, param_name in modules_to_prune:
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name=param_name, amount=pruning_ratio, n=2, dim=0)
            elif isinstance(module, nn.Linear):
                prune.ln_structured(module, name=param_name, amount=pruning_ratio, n=2, dim=0)
    else:
        # Unstructured pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
    
    # Remove pruning reparameterization to make pruning permanent
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    return model


def convert_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, int, int, int],
    dynamic_batch: bool = False,
    half_precision: bool = False
) -> bool:
    """Convert model to ONNX format."""
    print(f"Converting to ONNX: {output_path}")
    
    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        if half_precision:
            model = model.half()
            dummy_input = dummy_input.half()
        
        # Set dynamic axes for dynamic batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print("ONNX conversion successful")
        return True
    
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        return False


def convert_to_torchscript(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, int, int, int],
    optimize_for_mobile: bool = False
) -> bool:
    """Convert model to TorchScript format."""
    print(f"Converting to TorchScript: {output_path}")
    
    try:
        # Create dummy input for tracing
        dummy_input = torch.randn(*input_shape)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Optimize for mobile if requested
        if optimize_for_mobile:
            try:
                from torch.utils.mobile_optimizer import optimize_for_mobile
                traced_model = optimize_for_mobile(traced_model)
                print("Applied mobile optimization")
            except ImportError:
                print("Mobile optimization not available")
        
        # Save the model
        traced_model.save(str(output_path))
        
        print("TorchScript conversion successful")
        return True
    
    except Exception as e:
        print(f"TorchScript conversion failed: {e}")
        return False


def convert_to_tensorrt(
    onnx_path: Path,
    output_path: Path,
    input_shape: Tuple[int, int, int, int],
    half_precision: bool = False
) -> bool:
    """Convert ONNX model to TensorRT format."""
    print(f"Converting to TensorRT: {output_path}")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("TensorRT not available - please install tensorrt and pycuda")
        return False
    
    try:
        # Create TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model")
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if half_precision:
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build TensorRT engine")
            return False
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print("TensorRT conversion successful")
        return True
    
    except Exception as e:
        print(f"TensorRT conversion failed: {e}")
        return False


def validate_converted_model(
    original_model: nn.Module,
    converted_model_path: Path,
    format_type: str,
    input_shape: Tuple[int, int, int, int],
    tolerance: float = 1e-3
) -> bool:
    """Validate that converted model produces similar outputs."""
    print(f"Validating {format_type} conversion...")
    
    # Create test input
    test_input = torch.randn(*input_shape)
    
    # Get original model output
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(test_input)
    
    if isinstance(original_output, dict):
        original_output = original_output['logits']
    
    try:
        if format_type == "onnx":
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(converted_model_path))
            input_name = session.get_inputs()[0].name
            converted_output = session.run(None, {input_name: test_input.numpy()})[0]
            converted_output = torch.from_numpy(converted_output)
        
        elif format_type == "torchscript":
            traced_model = torch.jit.load(str(converted_model_path))
            traced_model.eval()
            with torch.no_grad():
                converted_output = traced_model(test_input)
        
        else:
            print(f"Validation not implemented for format: {format_type}")
            return True
        
        # Compare outputs
        if isinstance(converted_output, dict):
            converted_output = converted_output['logits']
        
        diff = torch.abs(original_output - converted_output).max().item()
        
        if diff < tolerance:
            print(f"Validation passed - max difference: {diff:.6f}")
            return True
        else:
            print(f"Validation failed - max difference: {diff:.6f} (tolerance: {tolerance})")
            return False
    
    except Exception as e:
        print(f"Validation failed with error: {e}")
        return False


def benchmark_model(
    model_path: Path,
    format_type: str,
    input_shape: Tuple[int, int, int, int],
    num_runs: int = 100
) -> Dict[str, float]:
    """Benchmark converted model performance."""
    print(f"Benchmarking {format_type} model...")
    
    import time
    
    # Create test input
    test_input = torch.randn(*input_shape)
    
    try:
        if format_type == "onnx":
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name
            input_dict = {input_name: test_input.numpy()}
            
            # Warmup
            for _ in range(10):
                session.run(None, input_dict)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                session.run(None, input_dict)
            end_time = time.time()
        
        elif format_type == "torchscript":
            model = torch.jit.load(str(model_path))
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(test_input)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    model(test_input)
            end_time = time.time()
        
        else:
            print(f"Benchmarking not implemented for format: {format_type}")
            return {}
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        results = {
            'total_time': total_time,
            'average_time': avg_time,
            'fps': fps,
            'num_runs': num_runs
        }
        
        print(f"Benchmark results:")
        print(f"  Average inference time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")
        
        return results
    
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        return {}


def main():
    """Main conversion function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(
        name="dinov3_conversion",
        level=log_level,
        log_dir=Path(args.output_dir) / "logs"
    )
    
    try:
        # Setup device
        device_manager = get_device_manager()
        
        # Load model
        model, config_dict = load_model_for_conversion(args.input_model, args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine output name
        if args.output_name:
            base_name = args.output_name
        else:
            base_name = Path(args.input_model).stem
        
        # Apply optimizations
        optimized_model = model
        
        if args.quantize:
            optimized_model = quantize_model(
                optimized_model,
                args.quantization_mode,
                args.calibration_data
            )
        
        if args.prune:
            optimized_model = prune_model(
                optimized_model,
                args.pruning_ratio,
                args.structured_pruning
            )
        
        # Input shape for conversion
        input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])
        
        # Perform conversions
        conversion_results = {}
        
        for format_type in args.format:
            print(f"\nConverting to {format_type.upper()}...")
            
            if format_type == "onnx":
                output_path = output_dir / f"{base_name}.onnx"
                success = convert_to_onnx(
                    optimized_model,
                    output_path,
                    input_shape,
                    args.dynamic_batch,
                    args.half_precision
                )
                
            elif format_type == "torchscript":
                output_path = output_dir / f"{base_name}.pt"
                success = convert_to_torchscript(
                    optimized_model,
                    output_path,
                    input_shape,
                    args.optimize_for_mobile
                )
            
            elif format_type == "tensorrt":
                # First convert to ONNX if not already done
                onnx_path = output_dir / f"{base_name}.onnx"
                if not onnx_path.exists():
                    convert_to_onnx(optimized_model, onnx_path, input_shape, False, args.half_precision)
                
                output_path = output_dir / f"{base_name}.engine"
                success = convert_to_tensorrt(
                    onnx_path,
                    output_path,
                    input_shape,
                    args.half_precision
                )
            
            else:
                print(f"Conversion to {format_type} not yet implemented")
                success = False
            
            conversion_results[format_type] = {
                'success': success,
                'output_path': str(output_path) if success else None
            }
            
            # Validate conversion if requested
            if success and args.validate_conversion and format_type in ["onnx", "torchscript"]:
                validation_success = validate_converted_model(
                    model, output_path, format_type, input_shape
                )
                conversion_results[format_type]['validation'] = validation_success
            
            # Benchmark if requested
            if success and args.benchmark and format_type in ["onnx", "torchscript"]:
                benchmark_results = benchmark_model(output_path, format_type, input_shape)
                conversion_results[format_type]['benchmark'] = benchmark_results
        
        # Save conversion report
        report = {
            'input_model': args.input_model,
            'config': config_dict,
            'conversion_args': vars(args),
            'input_shape': input_shape,
            'results': conversion_results
        }
        
        with open(output_dir / 'conversion_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\nConversion Summary:")
        print(f"Results saved to: {output_dir}")
        
        for format_type, result in conversion_results.items():
            status = "✓" if result['success'] else "✗"
            print(f"  {format_type.upper()}: {status}")
            if result.get('validation'):
                print(f"    Validation: ✓")
            if result.get('benchmark'):
                bench = result['benchmark']
                print(f"    Performance: {bench.get('average_time', 0)*1000:.2f}ms avg")
        
        logger.info("Model conversion completed!")
    
    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
DINOv3 Model Optimization for Deployment

Comprehensive optimization utilities including quantization, pruning, knowledge distillation,
and ONNX optimization for production deployment.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torch.ao.quantization as quantization
    from torch.ao.quantization import QConfigMapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    warnings.warn("Advanced quantization not available in this PyTorch version")

try:
    import torch.nn.utils.prune as prune

    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    warnings.warn("Pruning not available in this PyTorch version")

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn("ONNX optimization not available - install onnx and onnxruntime")


class ModelQuantizer:
    """Model quantization utilities for DINOv3 models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_model = None

    def dynamic_quantization(
        self, dtype: torch.dtype = torch.qint8, module_types: Optional[List] = None
    ) -> nn.Module:
        """Apply dynamic quantization to the model."""
        if not QUANTIZATION_AVAILABLE:
            raise RuntimeError("Quantization not available in this PyTorch version")

        if module_types is None:
            module_types = [nn.Linear, nn.Conv2d]

        print("Applying dynamic quantization...")
        self.original_model = self.model

        quantized_model = torch.quantization.quantize_dynamic(
            self.model, module_types, dtype=dtype
        )

        return quantized_model

    def static_quantization(
        self,
        calibration_dataloader: DataLoader,
        qconfig_mapping: Optional[QConfigMapping] = None,
    ) -> nn.Module:
        """Apply static quantization using calibration data."""
        if not QUANTIZATION_AVAILABLE:
            raise RuntimeError("Advanced quantization not available")

        print("Applying static quantization...")
        self.original_model = self.model

        # Set model to evaluation mode
        self.model.eval()

        # Default quantization config if not provided
        if qconfig_mapping is None:
            qconfig_mapping = QConfigMapping().set_global(
                torch.ao.quantization.get_default_qconfig("qnnpack")
            )

        # Prepare model for quantization
        example_inputs = self._get_example_inputs(calibration_dataloader)
        prepared_model = prepare_fx(self.model, qconfig_mapping, example_inputs)

        # Calibrate with sample data
        print("Calibrating model...")
        prepared_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_dataloader, desc="Calibrating")):
                if i >= 100:  # Limit calibration samples
                    break

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch

                prepared_model(inputs)

        # Convert to quantized model
        quantized_model = convert_fx(prepared_model)

        return quantized_model

    def _get_example_inputs(self, dataloader: DataLoader) -> torch.Tensor:
        """Get example inputs for quantization."""
        batch = next(iter(dataloader))
        if isinstance(batch, (list, tuple)):
            return batch[0][:1]  # Single sample
        return batch[:1]

    def compare_models(
        self,
        quantized_model: nn.Module,
        test_dataloader: DataLoader,
        num_samples: int = 100,
    ) -> Dict[str, Any]:
        """Compare original and quantized model performance."""
        if self.original_model is None:
            raise ValueError("No original model stored for comparison")

        print("Comparing model performance...")

        # Evaluate both models
        original_results = self._evaluate_model(
            self.original_model, test_dataloader, num_samples
        )
        quantized_results = self._evaluate_model(
            quantized_model, test_dataloader, num_samples
        )

        # Calculate model sizes
        original_size = self._calculate_model_size(self.original_model)
        quantized_size = self._calculate_model_size(quantized_model)

        comparison = {
            "original_accuracy": original_results["accuracy"],
            "quantized_accuracy": quantized_results["accuracy"],
            "accuracy_drop": original_results["accuracy"]
            - quantized_results["accuracy"],
            "original_inference_time": original_results["avg_inference_time"],
            "quantized_inference_time": quantized_results["avg_inference_time"],
            "speedup": original_results["avg_inference_time"]
            / quantized_results["avg_inference_time"],
            "original_model_size_mb": original_size,
            "quantized_model_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size,
        }

        return comparison

    def _evaluate_model(
        self, model: nn.Module, dataloader: DataLoader, num_samples: int
    ) -> Dict[str, float]:
        """Evaluate model accuracy and inference time."""
        model.eval()
        correct = 0
        total = 0
        inference_times = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if total >= num_samples:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None

                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = model(inputs)
                end_time.record()

                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

                # Calculate accuracy if targets available
                if targets is not None:
                    if isinstance(outputs, dict):
                        outputs = outputs["logits"]

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "avg_inference_time": np.mean(inference_times),
        }

    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb


class ModelPruner:
    """Model pruning utilities for DINOv3 models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.original_state = None

    def unstructured_pruning(
        self, pruning_ratio: float, pruning_method: str = "l1"
    ) -> nn.Module:
        """Apply unstructured pruning to the model."""
        if not PRUNING_AVAILABLE:
            raise RuntimeError("Pruning not available in this PyTorch version")

        print(
            f"Applying {pruning_method} unstructured pruning (ratio: {pruning_ratio})"
        )

        # Store original state
        self.original_state = self.model.state_dict().copy()

        # Collect parameters to prune
        parameters_to_prune = []
        for module_name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        # Apply pruning
        if pruning_method == "l1":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        elif pruning_method == "l2":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L2Unstructured,
                amount=pruning_ratio,
            )
        elif pruning_method == "random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio,
            )
        else:
            raise ValueError(f"Unknown pruning method: {pruning_method}")

        return self.model

    def structured_pruning(self, pruning_ratio: float, dimension: int = 0) -> nn.Module:
        """Apply structured pruning to the model."""
        if not PRUNING_AVAILABLE:
            raise RuntimeError("Pruning not available in this PyTorch version")

        print(f"Applying structured pruning (ratio: {pruning_ratio}, dim: {dimension})")

        # Store original state
        self.original_state = self.model.state_dict().copy()

        # Apply structured pruning to each relevant layer
        for module_name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(
                    module, name="weight", amount=pruning_ratio, n=2, dim=dimension
                )
            elif isinstance(module, nn.Linear):
                prune.ln_structured(
                    module, name="weight", amount=pruning_ratio, n=2, dim=dimension
                )

        return self.model

    def remove_pruning(self) -> nn.Module:
        """Remove pruning masks to make pruning permanent."""
        if not PRUNING_AVAILABLE:
            raise RuntimeError("Pruning not available in this PyTorch version")

        print("Removing pruning masks...")

        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    # No pruning mask found, skip
                    pass

        return self.model

    def calculate_sparsity(self) -> Dict[str, float]:
        """Calculate sparsity statistics for the pruned model."""
        total_params = 0
        pruned_params = 0
        layer_sparsity = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Check if module has pruning mask
                if hasattr(module, "weight_mask"):
                    mask = module.weight_mask
                    layer_total = mask.numel()
                    layer_pruned = (mask == 0).sum().item()

                    layer_sparsity[name] = layer_pruned / layer_total
                    total_params += layer_total
                    pruned_params += layer_pruned

        global_sparsity = pruned_params / total_params if total_params > 0 else 0.0

        return {
            "global_sparsity": global_sparsity,
            "layer_sparsity": layer_sparsity,
            "total_parameters": total_params,
            "pruned_parameters": pruned_params,
        }


class KnowledgeDistillationTrainer:
    """Knowledge distillation framework for model compression."""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: torch.device,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device
        self.temperature = temperature
        self.alpha = alpha

        # Set teacher to evaluation mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate knowledge distillation loss."""

        # Extract logits if outputs are dictionaries
        if isinstance(student_outputs, dict):
            student_logits = student_outputs["logits"]
        else:
            student_logits = student_outputs

        if isinstance(teacher_outputs, dict):
            teacher_logits = teacher_outputs["logits"]
        else:
            teacher_logits = teacher_outputs

        # Distillation loss (soft targets)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction="batchmean",
        ) * (self.temperature**2)

        # Standard cross-entropy loss (hard targets)
        student_loss = F.cross_entropy(student_logits, targets)

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss, distillation_loss, student_loss

    def train_epoch(
        self, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int
    ) -> Dict[str, float]:
        """Train student model for one epoch."""

        self.student_model.train()

        total_loss = 0.0
        total_distillation_loss = 0.0
        total_student_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()

            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)

            # Get student predictions
            student_outputs = self.student_model(inputs)

            # Calculate loss
            loss, distill_loss, student_loss = self.distillation_loss(
                student_outputs, teacher_outputs, targets
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            total_distillation_loss += distill_loss.item()
            total_student_loss += student_loss.item()

            # Accuracy
            if isinstance(student_outputs, dict):
                predictions = student_outputs["logits"]
            else:
                predictions = student_outputs
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            # Update progress bar
            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"}
            )

        return {
            "loss": total_loss / len(train_dataloader),
            "distillation_loss": total_distillation_loss / len(train_dataloader),
            "student_loss": total_student_loss / len(train_dataloader),
            "accuracy": correct / total,
        }

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate student model."""
        self.student_model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets, _ = batch

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Get predictions
                teacher_outputs = self.teacher_model(inputs)
                student_outputs = self.student_model(inputs)

                # Calculate loss
                loss, _, _ = self.distillation_loss(
                    student_outputs, teacher_outputs, targets
                )
                total_loss += loss.item()

                # Accuracy
                if isinstance(student_outputs, dict):
                    predictions = student_outputs["logits"]
                else:
                    predictions = student_outputs
                _, predicted = torch.max(predictions, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        return {"loss": total_loss / len(val_dataloader), "accuracy": correct / total}


class ONNXOptimizer:
    """ONNX model optimization utilities."""

    def __init__(self):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "ONNX optimization not available - install onnx and onnxruntime"
            )

    def optimize_onnx_model(
        self,
        onnx_model_path: str,
        optimized_model_path: str,
        optimization_level: str = "basic",
    ) -> Dict[str, Any]:
        """Optimize ONNX model with various techniques."""

        print(f"Optimizing ONNX model: {onnx_model_path}")

        # Load model
        session_options = ort.SessionOptions()

        # Set optimization level
        if optimization_level == "basic":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        elif optimization_level == "extended":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
        elif optimization_level == "all":
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        # Set optimized model path
        session_options.optimized_model_filepath = optimized_model_path

        # Create session (this will save the optimized model)
        session = ort.InferenceSession(onnx_model_path, session_options)

        # Get model info
        original_model = onnx.load(onnx_model_path)
        optimized_model = onnx.load(optimized_model_path)

        optimization_info = {
            "original_model_path": onnx_model_path,
            "optimized_model_path": optimized_model_path,
            "optimization_level": optimization_level,
            "original_nodes": len(original_model.graph.node),
            "optimized_nodes": len(optimized_model.graph.node),
            "node_reduction": len(original_model.graph.node)
            - len(optimized_model.graph.node),
            "providers": session.get_providers(),
        }

        print(f"Optimization complete:")
        print(f"  Nodes reduced: {optimization_info['node_reduction']}")
        print(f"  Original nodes: {optimization_info['original_nodes']}")
        print(f"  Optimized nodes: {optimization_info['optimized_nodes']}")

        return optimization_info

    def quantize_onnx_model(
        self,
        onnx_model_path: str,
        quantized_model_path: str,
        quantization_type: str = "dynamic",
    ) -> Dict[str, Any]:
        """Quantize ONNX model."""

        print(f"Quantizing ONNX model: {onnx_model_path}")

        if quantization_type == "dynamic":
            quantize_dynamic(
                onnx_model_path, quantized_model_path, weight_type=QuantType.QInt8
            )
        else:
            raise ValueError(f"Quantization type '{quantization_type}' not implemented")

        # Get model sizes
        original_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(quantized_model_path).stat().st_size / (1024 * 1024)

        quantization_info = {
            "original_model_path": onnx_model_path,
            "quantized_model_path": quantized_model_path,
            "quantization_type": quantization_type,
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": original_size / quantized_size,
        }

        print(f"Quantization complete:")
        print(f"  Compression ratio: {quantization_info['compression_ratio']:.2f}x")
        print(f"  Size reduction: {original_size - quantized_size:.1f} MB")

        return quantization_info

    def benchmark_onnx_model(
        self,
        onnx_model_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        providers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Benchmark ONNX model inference performance."""

        if providers is None:
            providers = ["CPUExecutionProvider"]

        print(f"Benchmarking ONNX model: {onnx_model_path}")

        # Create session
        session = ort.InferenceSession(onnx_model_path, providers=providers)

        # Create dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        import time

        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time

        benchmark_results = {
            "total_time": total_time,
            "average_time": avg_time,
            "fps": fps,
            "num_runs": num_runs,
            "providers": providers,
            "input_shape": input_shape,
        }

        print(f"Benchmark results:")
        print(f"  Average inference time: {avg_time*1000:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")

        return benchmark_results


def create_optimization_pipeline(
    model: nn.Module,
    optimization_config: Dict[str, Any],
    calibration_dataloader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Create complete optimization pipeline based on configuration."""

    optimized_model = model
    optimization_results = {"steps": []}

    # Apply pruning if requested
    if optimization_config.get("pruning", {}).get("enabled", False):
        pruning_config = optimization_config["pruning"]

        pruner = ModelPruner(optimized_model)

        if pruning_config.get("type") == "structured":
            optimized_model = pruner.structured_pruning(
                pruning_config.get("ratio", 0.2), pruning_config.get("dimension", 0)
            )
        else:  # unstructured
            optimized_model = pruner.unstructured_pruning(
                pruning_config.get("ratio", 0.2), pruning_config.get("method", "l1")
            )

        # Remove pruning masks
        optimized_model = pruner.remove_pruning()

        sparsity_stats = pruner.calculate_sparsity()
        optimization_results["steps"].append(
            {
                "type": "pruning",
                "config": pruning_config,
                "sparsity_stats": sparsity_stats,
            }
        )

    # Apply quantization if requested
    if optimization_config.get("quantization", {}).get("enabled", False):
        quantization_config = optimization_config["quantization"]

        quantizer = ModelQuantizer(optimized_model)

        if quantization_config.get("type") == "static" and calibration_dataloader:
            optimized_model = quantizer.static_quantization(calibration_dataloader)
        else:  # dynamic
            optimized_model = quantizer.dynamic_quantization(
                dtype=getattr(torch, quantization_config.get("dtype", "qint8"))
            )

        optimization_results["steps"].append(
            {"type": "quantization", "config": quantization_config}
        )

    optimization_results["success"] = True

    return optimized_model, optimization_results

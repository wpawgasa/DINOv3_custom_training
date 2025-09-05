import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch.distributed as dist


class DeviceManager:
    def __init__(self):
        self._device = None
        self._device_count = 0
        self._device_properties = {}
        self._is_distributed = False
        self._local_rank = 0
        self._world_size = 1

        self._initialize()

    def _initialize(self):
        if torch.cuda.is_available():
            self._device_count = torch.cuda.device_count()
            self._device = torch.device("cuda")
            self._gather_gpu_properties()
        else:
            self._device = torch.device("cpu")

        self._check_distributed_setup()

    def _gather_gpu_properties(self):
        for i in range(self._device_count):
            props = torch.cuda.get_device_properties(i)
            self._device_properties[i] = {
                "name": props.name,
                "total_memory": props.total_memory,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            }

    def _check_distributed_setup(self):
        if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._is_distributed = True
            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cuda_available(self) -> bool:
        return torch.cuda.is_available()

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def get_device_properties(self, device_id: int = 0) -> Dict[str, Any]:
        if device_id in self._device_properties:
            return self._device_properties[device_id].copy()
        return {}

    def get_memory_info(self, device_id: int = 0) -> Dict[str, int]:
        if not self.is_cuda_available:
            return {"total": 0, "free": 0, "used": 0}

        torch.cuda.set_device(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        free, used = torch.cuda.mem_get_info()

        return {
            "total": total,
            "free": free,
            "used": used,
            "allocated": torch.cuda.memory_allocated(device_id),
            "cached": torch.cuda.memory_reserved(device_id),
        }

    def get_optimal_batch_size(
        self, model_memory_mb: float, safety_factor: float = 0.8
    ) -> int:
        if not self.is_cuda_available:
            return 32  # Default CPU batch size

        memory_info = self.get_memory_info()
        available_memory_mb = (memory_info["free"] * safety_factor) / (1024 * 1024)

        if model_memory_mb <= 0:
            return 32

        estimated_batch_size = max(1, int(available_memory_mb / model_memory_mb))
        return min(estimated_batch_size, 128)  # Cap at 128

    def clear_cache(self):
        if self.is_cuda_available:
            torch.cuda.empty_cache()

    def synchronize(self):
        if self.is_cuda_available:
            torch.cuda.synchronize()

    def set_device(self, device_id: int):
        if self.is_cuda_available and 0 <= device_id < self.device_count:
            torch.cuda.set_device(device_id)
            self._device = torch.device(f"cuda:{device_id}")

    def get_system_info(self) -> Dict[str, Any]:
        info = {
            "cpu_count": os.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "torch_version": torch.__version__,
            "cuda_available": self.is_cuda_available,
            "device_count": self.device_count,
        }

        if self.is_cuda_available:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_devices"] = []

            for i in range(self.device_count):
                props = self.get_device_properties(i)
                info["gpu_devices"].append(
                    {
                        "id": i,
                        "name": props.get("name", "Unknown"),
                        "memory_gb": round(
                            props.get("total_memory", 0) / (1024**3), 2
                        ),
                        "compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
                    }
                )

        return info

    def print_device_info(self):
        info = self.get_system_info()
        print("=== Device Information ===")
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")

        if info["cuda_available"]:
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"cuDNN Version: {info['cudnn_version']}")
            print(f"GPU Device Count: {info['device_count']}")

            for gpu in info["gpu_devices"]:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

        print(f"CPU Count: {info['cpu_count']}")
        print(f"System Memory: {info['memory_gb']} GB")
        print("=" * 30)


def setup_distributed_training() -> Tuple[int, int, int]:
    if not dist.is_available():
        raise RuntimeError("Distributed training not available")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    return local_rank, world_size, rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device_for_model(model_size_mb: float = 100.0) -> torch.device:
    device_manager = DeviceManager()

    if not device_manager.is_cuda_available:
        return torch.device("cpu")

    for i in range(device_manager.device_count):
        memory_info = device_manager.get_memory_info(i)
        available_mb = memory_info["free"] / (1024 * 1024)

        if available_mb > model_size_mb * 2:  # 2x safety margin
            return torch.device(f"cuda:{i}")

    return torch.device("cuda:0")  # Fallback to first GPU


def move_to_device(obj, device: torch.device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj


def benchmark_device_performance() -> Dict[str, float]:
    device_manager = DeviceManager()
    results = {}

    # CPU benchmark
    start = (
        torch.cuda.Event(enable_timing=True)
        if device_manager.is_cuda_available
        else None
    )
    end = (
        torch.cuda.Event(enable_timing=True)
        if device_manager.is_cuda_available
        else None
    )

    x = torch.randn(1000, 1000, device="cpu")

    import time

    cpu_start = time.time()
    for _ in range(10):
        torch.mm(x, x)
    cpu_time = (time.time() - cpu_start) / 10
    results["cpu_matmul_time"] = cpu_time

    # GPU benchmark
    if device_manager.is_cuda_available:
        x_gpu = x.to(device_manager.device)

        start.record()
        for _ in range(10):
            torch.mm(x_gpu, x_gpu)
        end.record()

        torch.cuda.synchronize()
        gpu_time = start.elapsed_time(end) / 1000.0 / 10  # Convert to seconds
        results["gpu_matmul_time"] = gpu_time
        results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0

    return results


_global_device_manager = None


def get_device_manager() -> DeviceManager:
    global _global_device_manager
    if _global_device_manager is None:
        _global_device_manager = DeviceManager()
    return _global_device_manager


def get_device() -> torch.device:
    return get_device_manager().device


def is_cuda_available() -> bool:
    return get_device_manager().is_cuda_available

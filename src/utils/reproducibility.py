import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
import json


class ReproducibilityManager:
    def __init__(self, seed: int = 42, strict: bool = True):
        self.seed = seed
        self.strict = strict
        self.initial_states = {}
        
        self._set_seed()
        if strict:
            self._set_deterministic_mode()
    
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        os.environ['PYTHONHASHSEED'] = str(self.seed)
    
    def _set_deterministic_mode(self):
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        if torch.cuda.is_available():
            cudnn.deterministic = True
            cudnn.benchmark = False
        
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    def save_random_states(self, checkpoint_path: Path):
        states = {
            'seed': self.seed,
            'python_random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            states['torch_cuda_random_states'] = torch.cuda.get_rng_state_all()
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            torch.save(states, f)
    
    def load_random_states(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Random state checkpoint not found: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            states = torch.load(f, map_location='cpu')
        
        self.seed = states['seed']
        random.setstate(states['python_random_state'])
        np.random.set_state(states['numpy_random_state'])
        torch.set_rng_state(states['torch_random_state'])
        
        if torch.cuda.is_available() and 'torch_cuda_random_states' in states:
            torch.cuda.set_rng_state_all(states['torch_cuda_random_states'])
    
    def get_reproducibility_info(self) -> Dict[str, Any]:
        info = {
            'seed': self.seed,
            'strict_mode': self.strict,
            'torch_deterministic': torch.are_deterministic_algorithms_enabled(),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'cudnn_deterministic': cudnn.deterministic,
                'cudnn_benchmark': cudnn.benchmark,
            })
        
        return info
    
    def create_experiment_hash(self, config: Dict[str, Any]) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        reproducibility_info = self.get_reproducibility_info()
        
        combined_str = f"{config_str}_{reproducibility_info}"
        return hashlib.sha256(combined_str.encode()).hexdigest()[:16]
    
    def validate_environment(self) -> Dict[str, bool]:
        validation = {
            'seed_set': True,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'pythonhashseed_set': os.environ.get('PYTHONHASHSEED') == str(self.seed),
        }
        
        if torch.cuda.is_available():
            validation.update({
                'cudnn_deterministic': cudnn.deterministic,
                'cudnn_benchmark_disabled': not cudnn.benchmark,
                'cublas_workspace_set': 'CUBLAS_WORKSPACE_CONFIG' in os.environ,
            })
        
        return validation
    
    def reset_to_initial_seed(self):
        self._set_seed()
        if self.strict:
            self._set_deterministic_mode()


def set_seed(seed: int = 42, strict: bool = True) -> ReproducibilityManager:
    return ReproducibilityManager(seed, strict)


def worker_init_fn(worker_id: int, seed: int = 42):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class DeterministicDataLoader:
    @staticmethod
    def create_generator(seed: int = 42) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    
    @staticmethod
    def get_worker_init_fn(seed: int = 42):
        def init_fn(worker_id):
            worker_init_fn(worker_id, seed)
        return init_fn


class ReproducibleTraining:
    def __init__(self, seed: int = 42, strict: bool = True):
        self.repro_manager = ReproducibilityManager(seed, strict)
        self.epoch_seeds = {}
    
    def set_epoch_seed(self, epoch: int, base_seed: Optional[int] = None):
        if base_seed is None:
            base_seed = self.repro_manager.seed
        
        epoch_seed = base_seed + epoch
        self.epoch_seeds[epoch] = epoch_seed
        
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
    
    def get_dataloader_args(self) -> Dict[str, Any]:
        return {
            'generator': DeterministicDataLoader.create_generator(self.repro_manager.seed),
            'worker_init_fn': DeterministicDataLoader.get_worker_init_fn(self.repro_manager.seed),
        }
    
    def save_training_state(self, checkpoint_path: Path, epoch: int):
        state_path = checkpoint_path.parent / f"random_state_epoch_{epoch}.pth"
        self.repro_manager.save_random_states(state_path)
    
    def load_training_state(self, checkpoint_path: Path, epoch: int):
        state_path = checkpoint_path.parent / f"random_state_epoch_{epoch}.pth"
        self.repro_manager.load_random_states(state_path)


def check_reproducibility(model_fn, input_data, seed: int = 42, num_runs: int = 3) -> bool:
    outputs = []
    
    for run in range(num_runs):
        repro_manager = ReproducibilityManager(seed, strict=True)
        
        if hasattr(model_fn, 'reset_parameters'):
            model_fn.reset_parameters()
        
        with torch.no_grad():
            output = model_fn(input_data)
            outputs.append(output.clone())
    
    for i in range(1, len(outputs)):
        if not torch.allclose(outputs[0], outputs[i], atol=1e-6):
            return False
    
    return True


def create_reproducible_split(
    dataset_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    indices = list(range(dataset_size))
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        indices, [train_size, val_size, test_size], generator=generator
    )
    
    return train_indices.indices, val_indices.indices, test_indices.indices


def log_reproducibility_info(logger, repro_manager: ReproducibilityManager):
    info = repro_manager.get_reproducibility_info()
    validation = repro_manager.validate_environment()
    
    logger.info("Reproducibility Configuration:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Environment Validation:")
    for key, value in validation.items():
        status = "✓" if value else "✗"
        logger.info(f"  {status} {key}: {value}")
    
    if not all(validation.values()):
        logger.warning("Some reproducibility checks failed. Results may not be fully reproducible.")


_global_repro_manager = None

def get_reproducibility_manager(seed: int = 42, strict: bool = True) -> ReproducibilityManager:
    global _global_repro_manager
    if _global_repro_manager is None:
        _global_repro_manager = ReproducibilityManager(seed, strict)
    return _global_repro_manager
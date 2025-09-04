"""
Pytest configuration and fixtures for DINOv3 tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import json

import pytest
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.schemas import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig
from models.model_factory import create_model


@pytest.fixture(scope="session")
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    # Create a random RGB image
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(image_array, mode='RGB')


@pytest.fixture(scope="session")
def sample_images(sample_image) -> list:
    """Create a batch of sample images."""
    return [sample_image.copy() for _ in range(5)]


@pytest.fixture(scope="session")
def sample_dataset_dir(temp_dir: Path) -> Path:
    """Create a sample dataset directory structure."""
    dataset_dir = temp_dir / "sample_dataset"
    
    # Create ImageFolder structure
    classes = ["class_0", "class_1", "class_2"]
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        for i in range(5):
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, mode='RGB')
            image.save(class_dir / f"sample_{i}.jpg")
    
    return dataset_dir


@pytest.fixture(scope="session")
def model_config() -> ModelConfig:
    """Create a sample model configuration."""
    return ModelConfig(
        variant="dinov3_vits16",
        task_type="classification",
        num_classes=3,
        pretrained=True,
        dropout=0.1
    )


@pytest.fixture(scope="session")
def training_config() -> TrainingConfig:
    """Create a sample training configuration."""
    return TrainingConfig(
        mode="linear_probe",
        epochs=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_epochs=1,
        freeze_backbone=True,
        mixed_precision=True,
        gradient_clipping=1.0,
        early_stopping={
            "patience": 3,
            "min_delta": 1e-4
        }
    )


@pytest.fixture(scope="session")
def data_config() -> DataConfig:
    """Create a sample data configuration."""
    return DataConfig(
        image_size=[224, 224],
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle_train=True,
        train_data_path="path/to/train",
        val_data_path="path/to/val"
    )


@pytest.fixture(scope="session")
def experiment_config(model_config, training_config, data_config) -> ExperimentConfig:
    """Create a complete experiment configuration."""
    return ExperimentConfig(
        name="test_experiment",
        description="Test experiment configuration",
        seed=42,
        model=model_config,
        training=training_config,
        data=data_config,
        optimizer={
            "name": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1e-8
        },
        scheduler={
            "name": "cosine_annealing",
            "min_lr": 1e-6,
            "warmup_type": "linear"
        },
        augmentation={
            "domain": "natural",
            "train": {
                "horizontal_flip": True,
                "color_jitter": {
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.1
                }
            },
            "val": {
                "resize_shortest": True,
                "center_crop": True,
                "normalize": True
            }
        },
        logging={
            "log_interval": 1,
            "save_interval": 2,
            "use_tensorboard": False,
            "use_wandb": False
        },
        evaluation={
            "metrics": ["accuracy", "top5_accuracy"],
            "save_predictions": False,
            "visualize_predictions": False
        }
    )


@pytest.fixture(scope="session")
def sample_model(model_config, device):
    """Create a sample model for testing."""
    model = create_model(model_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def sample_config_file(temp_dir: Path, experiment_config) -> Path:
    """Create a sample configuration file."""
    config_file = temp_dir / "test_config.yaml"
    
    # Convert to OmegaConf and save
    config_dict = experiment_config.dict()
    omega_config = OmegaConf.create(config_dict)
    
    with open(config_file, 'w') as f:
        OmegaConf.save(omega_config, f)
    
    return config_file


@pytest.fixture(scope="session")
def sample_checkpoint(temp_dir: Path, sample_model, experiment_config) -> Path:
    """Create a sample model checkpoint."""
    checkpoint_path = temp_dir / "sample_checkpoint.pth"
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': sample_model.state_dict(),
        'optimizer_state_dict': {},
        'scheduler_state_dict': {},
        'best_metric': 0.85,
        'training_time': 3600.0,
        'config': experiment_config.dict()
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock W&B for testing."""
    class MockWandB:
        def init(self, **kwargs):
            pass
        
        def log(self, metrics):
            pass
        
        def finish(self):
            pass
        
        def save(self, path):
            pass
    
    mock_wandb = MockWandB()
    monkeypatch.setattr("wandb", mock_wandb)
    return mock_wandb


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow for testing."""
    class MockMLflow:
        def set_experiment(self, name):
            pass
        
        def start_run(self, **kwargs):
            return self
        
        def log_metric(self, key, value, step=None):
            pass
        
        def log_param(self, key, value):
            pass
        
        def log_artifact(self, path):
            pass
        
        def end_run(self):
            pass
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    mock_mlflow = MockMLflow()
    monkeypatch.setattr("mlflow", mock_mlflow)
    return mock_mlflow


@pytest.fixture
def sample_batch(sample_images, device):
    """Create a sample batch tensor."""
    # Convert images to tensors
    batch_tensors = []
    for img in sample_images:
        # Simple normalization
        tensor = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        batch_tensors.append(tensor)
    
    batch = torch.stack(batch_tensors).to(device)
    return batch


@pytest.fixture
def sample_targets(device):
    """Create sample target labels."""
    return torch.tensor([0, 1, 2, 0, 1], device=device)


# Test markers
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or hasattr(item.function, "_gpu_required"):
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "benchmark" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def set_test_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


# Skip conditions
skip_if_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

skip_if_no_transformers = pytest.mark.skipif(
    not hasattr(torch.nn, "Transformer"),
    reason="Transformers not available in this PyTorch version"
)

# Test environment setup
def pytest_sessionstart(session):
    """Set up test environment."""
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "ERROR"  # Reduce log noise during tests
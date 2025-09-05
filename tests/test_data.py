"""
Unit tests for data pipeline functionality.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.augmentations import DomainAugmentations, create_transforms
from data.dataset import CustomDINOv3Dataset, create_dataset
from data.preprocessing import ImagePreprocessor
from data.validation import DatasetValidator


class TestCustomDINOv3Dataset:
    """Test custom dataset functionality."""

    def test_dataset_creation(self, sample_dataset_dir):
        """Test dataset creation from directory."""
        dataset = CustomDINOv3Dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
        )

        assert len(dataset) == 15  # 3 classes * 5 images each
        assert len(dataset.classes) == 3
        assert set(dataset.classes) == {"class_0", "class_1", "class_2"}

    def test_dataset_getitem(self, sample_dataset_dir):
        """Test dataset item retrieval."""
        dataset = CustomDINOv3Dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
        )

        # Get first item
        image, label = dataset[0]

        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert 0 <= label < len(dataset.classes)
        assert image.mode == "RGB"
        assert image.size == (224, 224)

    def test_dataset_with_transform(self, sample_dataset_dir):
        """Test dataset with transforms applied."""
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dataset = CustomDINOv3Dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform,
        )

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_dataset_class_mapping(self, sample_dataset_dir):
        """Test dataset class to index mapping."""
        dataset = CustomDINOv3Dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
        )

        assert hasattr(dataset, "class_to_idx")
        assert len(dataset.class_to_idx) == 3

        for class_name in dataset.classes:
            assert class_name in dataset.class_to_idx
            assert 0 <= dataset.class_to_idx[class_name] < len(dataset.classes)

    def test_create_dataset_function(self, sample_dataset_dir):
        """Test dataset creation function."""
        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
            cache_images=False,
        )

        assert isinstance(dataset, CustomDINOv3Dataset)
        assert len(dataset) == 15


class TestImagePreprocessor:
    """Test image preprocessing functionality."""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = ImagePreprocessor(
            target_size=(224, 224),
            resize_method="resize_shortest",
            quality_threshold=50,
        )

        assert preprocessor.target_size == (224, 224)
        assert preprocessor.resize_method == "resize_shortest"
        assert preprocessor.quality_threshold == 50

    def test_image_processing(self, sample_image):
        """Test basic image processing."""
        preprocessor = ImagePreprocessor(target_size=(256, 256), resize_method="resize")

        processed_image = preprocessor.process_image(sample_image)

        assert isinstance(processed_image, Image.Image)
        assert processed_image.size == (256, 256)
        assert processed_image.mode == "RGB"

    def test_resize_methods(self, sample_image):
        """Test different resize methods."""
        methods = ["resize", "resize_shortest", "center_crop", "resize_crop"]
        target_size = (256, 256)

        for method in methods:
            preprocessor = ImagePreprocessor(
                target_size=target_size, resize_method=method
            )

            processed_image = preprocessor.process_image(sample_image)
            assert isinstance(processed_image, Image.Image)
            assert processed_image.size == target_size

    def test_grayscale_conversion(self):
        """Test grayscale to RGB conversion."""
        # Create grayscale image
        gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        gray_image = Image.fromarray(gray_array, mode="L")

        preprocessor = ImagePreprocessor(
            target_size=(224, 224), convert_grayscale_to_rgb=True
        )

        processed_image = preprocessor.process_image(gray_image)

        assert processed_image.mode == "RGB"
        assert processed_image.size == (224, 224)

    def test_quality_filtering(self):
        """Test image quality filtering."""
        # Create low quality image (very blurry)
        low_quality_array = np.ones((50, 50, 3), dtype=np.uint8) * 128
        low_quality_image = Image.fromarray(low_quality_array, mode="RGB")

        preprocessor = ImagePreprocessor(
            target_size=(224, 224), quality_threshold=80  # High threshold
        )

        # Low quality image should be filtered out
        processed_image = preprocessor.process_image(low_quality_image)
        assert processed_image is None  # Should be filtered out

    def test_batch_processing(self, sample_images):
        """Test batch image processing."""
        preprocessor = ImagePreprocessor(target_size=(256, 256), resize_method="resize")

        processed_images = preprocessor.process_batch(sample_images)

        assert len(processed_images) == len(sample_images)
        for img in processed_images:
            assert isinstance(img, Image.Image)
            assert img.size == (256, 256)


class TestDomainAugmentations:
    """Test domain-specific augmentations."""

    def test_natural_augmentations(self):
        """Test natural image augmentations."""
        aug = DomainAugmentations("natural", image_size=224)

        train_transform = aug.get_train_transforms()
        val_transform = aug.get_val_transforms()

        assert train_transform is not None
        assert val_transform is not None

    def test_medical_augmentations(self):
        """Test medical image augmentations."""
        aug = DomainAugmentations("medical", image_size=224)

        train_transform = aug.get_train_transforms()
        val_transform = aug.get_val_transforms()

        assert train_transform is not None
        assert val_transform is not None

    def test_satellite_augmentations(self):
        """Test satellite image augmentations."""
        aug = DomainAugmentations("satellite", image_size=224)

        train_transform = aug.get_train_transforms()
        val_transform = aug.get_val_transforms()

        assert train_transform is not None
        assert val_transform is not None

    def test_industrial_augmentations(self):
        """Test industrial image augmentations."""
        aug = DomainAugmentations("industrial", image_size=224)

        train_transform = aug.get_train_transforms()
        val_transform = aug.get_val_transforms()

        assert train_transform is not None
        assert val_transform is not None

    def test_augmentation_application(self, sample_image):
        """Test augmentation application to images."""
        aug = DomainAugmentations("natural", image_size=224)
        train_transform = aug.get_train_transforms()

        # Apply multiple times to check randomness
        transformed_1 = train_transform(sample_image)
        transformed_2 = train_transform(sample_image)

        assert isinstance(transformed_1, torch.Tensor)
        assert isinstance(transformed_2, torch.Tensor)
        assert transformed_1.shape == (3, 224, 224)
        assert transformed_2.shape == (3, 224, 224)

        # Transforms should be different due to randomness
        assert not torch.equal(transformed_1, transformed_2)

    def test_create_transforms_function(self):
        """Test create_transforms function."""
        transform_manager = create_transforms(
            domain="natural", image_size=224, train_kwargs={}, val_kwargs={}
        )

        assert hasattr(transform_manager, "get_train_transform")
        assert hasattr(transform_manager, "get_val_transform")

        train_transform = transform_manager.get_train_transform()
        val_transform = transform_manager.get_val_transform()

        assert train_transform is not None
        assert val_transform is not None


class TestDatasetValidator:
    """Test dataset validation functionality."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = DatasetValidator()
        assert validator is not None

    def test_imagefolder_validation(self, sample_dataset_dir):
        """Test ImageFolder dataset validation."""
        validator = DatasetValidator()

        results = validator.validate_imagefolder_dataset(
            sample_dataset_dir, min_samples_per_class=1, validate_images=True
        )

        assert "is_valid" in results
        assert "num_classes" in results
        assert "total_samples" in results
        assert "class_distribution" in results

        assert results["is_valid"] is True
        assert results["num_classes"] == 3
        assert results["total_samples"] == 15

    def test_image_validation(self, sample_dataset_dir):
        """Test individual image validation."""
        validator = DatasetValidator()

        # Get first image path
        first_class_dir = next(sample_dataset_dir.iterdir())
        first_image_path = next(first_class_dir.glob("*.jpg"))

        is_valid = validator.validate_image(first_image_path)
        assert is_valid is True

    def test_class_balance_analysis(self, sample_dataset_dir):
        """Test class balance analysis."""
        validator = DatasetValidator()

        balance_info = validator.analyze_class_balance(sample_dataset_dir)

        assert "class_counts" in balance_info
        assert "balance_ratio" in balance_info
        assert "is_balanced" in balance_info

        # All classes should have 5 samples
        for class_name, count in balance_info["class_counts"].items():
            assert count == 5


class TestDataLoader:
    """Test DataLoader functionality."""

    def test_dataloader_creation(self, sample_dataset_dir):
        """Test DataLoader creation."""
        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
            cache_images=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            drop_last=False,
        )

        assert len(dataloader) == 4  # ceil(15/4) = 4 batches

    def test_dataloader_iteration(self, sample_dataset_dir):
        """Test DataLoader iteration."""
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform,
            cache_images=False,
        )

        dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)

        batch_count = 0
        total_samples = 0

        for batch_images, batch_labels in dataloader:
            batch_count += 1
            current_batch_size = batch_images.shape[0]
            total_samples += current_batch_size

            # Check batch shapes
            assert batch_images.shape == (current_batch_size, 3, 224, 224)
            assert batch_labels.shape == (current_batch_size,)

            # Check data types
            assert batch_images.dtype == torch.float32
            assert batch_labels.dtype == torch.long

            # Check value ranges (after normalization)
            assert (
                batch_images.min() >= -3.0
            )  # Approximate lower bound after normalization
            assert (
                batch_images.max() <= 3.0
            )  # Approximate upper bound after normalization

            # Check label ranges
            assert batch_labels.min() >= 0
            assert batch_labels.max() < 3

        assert total_samples == 15  # All samples processed
        assert batch_count == 5  # ceil(15/3) = 5 batches

    def test_dataloader_with_collate_fn(self, sample_dataset_dir):
        """Test DataLoader with custom collate function."""

        def custom_collate(batch):
            images, labels = zip(*batch)

            # Convert to tensors
            image_tensors = []
            for img in images:
                # Simple tensor conversion
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                image_tensors.append(img_tensor)

            images_tensor = torch.stack(image_tensors)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            return images_tensor, labels_tensor

        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=None,
            cache_images=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate,
        )

        for batch_images, batch_labels in dataloader:
            assert isinstance(batch_images, torch.Tensor)
            assert isinstance(batch_labels, torch.Tensor)
            assert batch_images.shape[1:] == (3, 224, 224)
            break  # Test first batch only


class TestDataIntegration:
    """Integration tests for data pipeline."""

    def test_end_to_end_data_pipeline(self, sample_dataset_dir):
        """Test complete data pipeline from directory to model input."""
        import torchvision.transforms as transforms

        # Create transform
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Create dataset
        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform,
            cache_images=False,
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        # Get one batch
        batch_images, batch_labels = next(iter(dataloader))

        # Verify batch is ready for model
        assert batch_images.shape[1:] == (3, 224, 224)  # Correct input shape
        assert batch_images.dtype == torch.float32  # Correct dtype
        assert batch_labels.dtype == torch.long  # Correct label dtype
        assert len(torch.unique(batch_labels)) <= 3  # Valid class indices

    @pytest.mark.slow
    def test_data_loading_performance(self, sample_dataset_dir):
        """Test data loading performance."""
        import time

        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform,
            cache_images=False,
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        # Time data loading
        start_time = time.time()

        for batch_images, batch_labels in dataloader:
            pass  # Just iterate through all batches

        end_time = time.time()
        loading_time = end_time - start_time

        # Loading should be reasonably fast (less than 10 seconds for small dataset)
        assert loading_time < 10.0

        # Calculate throughput
        samples_per_second = len(dataset) / loading_time
        assert samples_per_second > 1.0  # At least 1 sample per second

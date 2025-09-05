import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms

from ..utils.logging import get_logger


class DomainAugmentationPolicies:
    """Domain-specific augmentation policies optimized for different use cases."""

    @staticmethod
    def medical_imaging_policy(
        image_size: int = 224, conservative: bool = True
    ) -> A.Compose:
        if conservative:
            # Conservative augmentations for medical imaging
            return A.Compose(
                [
                    A.Resize(image_size, image_size, interpolation=1),
                    A.HorizontalFlip(p=0.3),
                    A.Rotate(limit=10, p=0.3, border_mode=0, value=0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.3
                    ),
                    A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            # More aggressive augmentations
            return A.Compose(
                [
                    A.Resize(image_size, image_size, interpolation=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.Rotate(limit=15, p=0.4, border_mode=0, value=0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.4
                    ),
                    A.GaussNoise(var_limit=(0.0, 0.02), p=0.3),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                    A.ElasticTransform(
                        alpha=1,
                        sigma=20,
                        alpha_affine=20,
                        border_mode=0,
                        value=0,
                        p=0.2,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    @staticmethod
    def satellite_imagery_policy(
        image_size: int = 224, multi_scale: bool = True
    ) -> A.Compose:
        augmentations = []

        if multi_scale:
            # Multi-scale training for satellite imagery
            augmentations.extend(
                [
                    A.RandomResizedCrop(
                        height=image_size,
                        width=image_size,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=1,
                        p=0.8,
                    ),
                ]
            )
        else:
            augmentations.append(A.Resize(image_size, image_size, interpolation=1))

        augmentations.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    limit=90, p=0.6, border_mode=0, value=0
                ),  # Satellite images can be rotated freely
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5
                ),
                A.GaussNoise(var_limit=(0.0, 0.03), p=0.3),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=3, p=1.0),
                        A.GaussianBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.3,
                ),
                A.RandomShadow(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        return A.Compose(augmentations)

    @staticmethod
    def industrial_inspection_policy(
        image_size: int = 224, defect_focused: bool = True
    ) -> A.Compose:
        augmentations = [
            A.Resize(image_size, image_size, interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=0.4, border_mode=0, value=0),
        ]

        if defect_focused:
            # Augmentations that preserve defect characteristics
            augmentations.extend(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.3, p=0.5
                    ),
                    A.GaussNoise(var_limit=(0.0, 0.02), p=0.3),
                    A.OneOf(
                        [
                            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
                            A.UnsharpMask(
                                blur_limit=3,
                                sigma_limit=1.0,
                                alpha=0.2,
                                threshold=10,
                                p=1.0,
                            ),
                        ],
                        p=0.4,
                    ),
                    # Preserve fine details important for defect detection
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
                ]
            )
        else:
            # Standard augmentations
            augmentations.extend(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.6
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=15,
                        p=0.4,
                    ),
                    A.GaussNoise(var_limit=(0.0, 0.03), p=0.4),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ]
            )

        augmentations.extend(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

        return A.Compose(augmentations)

    @staticmethod
    def natural_images_policy(image_size: int = 224, strong: bool = False) -> A.Compose:
        if strong:
            # Strong augmentations (similar to AutoAugment/RandAugment)
            return A.Compose(
                [
                    A.RandomResizedCrop(
                        height=image_size,
                        width=image_size,
                        scale=(0.6, 1.0),
                        ratio=(0.75, 1.33),
                        interpolation=1,
                        p=1.0,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf(
                        [
                            A.ColorJitter(
                                brightness=0.4,
                                contrast=0.4,
                                saturation=0.4,
                                hue=0.1,
                                p=1.0,
                            ),
                            A.ToGray(p=1.0),
                            A.ToSepia(p=1.0),
                        ],
                        p=0.6,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=7, p=1.0),
                            A.MotionBlur(blur_limit=7, p=1.0),
                        ],
                        p=0.3,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(var_limit=(0.0, 0.05), p=1.0),
                            A.ISONoise(
                                color_shift=(0.01, 0.1), intensity=(0.1, 0.7), p=1.0
                            ),
                        ],
                        p=0.4,
                    ),
                    A.OneOf(
                        [
                            A.RandomShadow(p=1.0),
                            A.RandomSunFlare(
                                p=1.0, flare_roi=(0, 0, 1, 0.5), angle_lower=0.5
                            ),
                        ],
                        p=0.3,
                    ),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )
        else:
            # Standard augmentations
            return A.Compose(
                [
                    A.RandomResizedCrop(
                        height=image_size,
                        width=image_size,
                        scale=(0.8, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=1,
                        p=1.0,
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6
                    ),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.GaussNoise(var_limit=(0.0, 0.02), p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )


class AugmentationFactory:
    """Factory class for creating augmentation pipelines."""

    DOMAIN_POLICIES = {
        "medical": DomainAugmentationPolicies.medical_imaging_policy,
        "satellite": DomainAugmentationPolicies.satellite_imagery_policy,
        "industrial": DomainAugmentationPolicies.industrial_inspection_policy,
        "natural": DomainAugmentationPolicies.natural_images_policy,
    }

    @classmethod
    def create_train_transforms(
        self, domain: str = "natural", image_size: int = 224, **kwargs
    ) -> A.Compose:
        if domain not in self.DOMAIN_POLICIES:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {list(self.DOMAIN_POLICIES.keys())}"
            )

        return self.DOMAIN_POLICIES[domain](image_size=image_size, **kwargs)

    @classmethod
    def create_val_transforms(
        self, image_size: int = 224, crop_ratio: float = 0.95
    ) -> A.Compose:
        """Create validation transforms (minimal augmentation)."""
        crop_size = int(image_size / crop_ratio)

        return A.Compose(
            [
                A.Resize(crop_size, crop_size, interpolation=1),
                A.CenterCrop(image_size, image_size, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    @classmethod
    def create_test_transforms(self, image_size: int = 224) -> A.Compose:
        """Create test transforms (no augmentation)."""
        return A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class MixAugmentation:
    """Mix-based augmentations like MixUp and CutMix."""

    def __init__(
        self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, prob: float = 0.5
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch

        if random.random() < self.prob:
            if random.random() < 0.5:
                return self.mixup(images, labels)
            else:
                return self.cutmix(images, labels)

        return images, labels

    def mixup(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        mixed_images = lam * images + (1 - lam) * images[indices]

        # Create soft labels
        labels_onehot = torch.zeros(batch_size, labels.max() + 1)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_mixed = lam * labels_onehot + (1 - lam) * labels_onehot[indices]

        return mixed_images, labels_mixed

    def cutmix(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)

        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        # Get bounding box
        W, H = images.size(2), images.size(3)
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_ratio)
        cut_h = np.int(H * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[
            indices, :, bbx1:bbx2, bby1:bby2
        ]

        # Adjust lambda to match the area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        # Create soft labels
        labels_onehot = torch.zeros(batch_size, labels.max() + 1)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_mixed = lam * labels_onehot + (1 - lam) * labels_onehot[indices]

        return mixed_images, labels_mixed


class TransformManager:
    """Manager class for handling training and validation transforms."""

    def __init__(
        self,
        domain: str = "natural",
        image_size: int = 224,
        train_kwargs: Optional[Dict] = None,
        val_kwargs: Optional[Dict] = None,
        use_mixaug: bool = False,
        mixaug_kwargs: Optional[Dict] = None,
        logger=None,
    ):
        self.domain = domain
        self.image_size = image_size
        self.train_kwargs = train_kwargs or {}
        self.val_kwargs = val_kwargs or {}
        self.use_mixaug = use_mixaug
        self.mixaug_kwargs = mixaug_kwargs or {}
        self.logger = logger or get_logger("transforms")

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self.mix_augmentation = None

        self._create_transforms()

    def _create_transforms(self):
        try:
            self.train_transform = AugmentationFactory.create_train_transforms(
                domain=self.domain, image_size=self.image_size, **self.train_kwargs
            )

            self.val_transform = AugmentationFactory.create_val_transforms(
                image_size=self.image_size, **self.val_kwargs
            )

            self.test_transform = AugmentationFactory.create_test_transforms(
                image_size=self.image_size
            )

            if self.use_mixaug:
                self.mix_augmentation = MixAugmentation(**self.mixaug_kwargs)

            self.logger.info(f"Created transforms for domain: {self.domain}")

        except Exception as e:
            self.logger.error(f"Failed to create transforms: {e}")
            raise

    def get_train_transform(self) -> A.Compose:
        return self.train_transform

    def get_val_transform(self) -> A.Compose:
        return self.val_transform

    def get_test_transform(self) -> A.Compose:
        return self.test_transform

    def apply_mix_augmentation(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mix_augmentation is not None:
            return self.mix_augmentation(batch)
        return batch

    def visualize_augmentations(
        self,
        image: Union[np.ndarray, Image.Image],
        num_samples: int = 8,
        save_path: Optional[str] = None,
    ):
        """Visualize augmentations on a sample image."""
        import matplotlib.pyplot as plt

        if isinstance(image, Image.Image):
            image = np.array(image)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(num_samples):
            if i == 0:
                # Show original
                axes[i].imshow(image)
                axes[i].set_title("Original")
            else:
                # Apply augmentation
                augmented = self.train_transform(image=image)["image"]
                if isinstance(augmented, torch.Tensor):
                    # Convert tensor to numpy and denormalize
                    augmented = augmented.permute(1, 2, 0).numpy()
                    # Denormalize using ImageNet stats
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    augmented = std * augmented + mean
                    augmented = np.clip(augmented, 0, 1)

                axes[i].imshow(augmented)
                axes[i].set_title(f"Augmented {i}")

            axes[i].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Augmentation samples saved to {save_path}")

        plt.show()


def create_transforms(
    domain: str = "natural", image_size: int = 224, **kwargs
) -> TransformManager:
    """Convenience function to create a TransformManager."""
    return TransformManager(domain=domain, image_size=image_size, **kwargs)

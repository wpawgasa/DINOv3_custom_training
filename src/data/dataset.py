import hashlib
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.logging import get_logger


class CustomDINOv3Dataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        annotation_format: str = "imagefolder",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_images: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
        logger=None,
    ):
        self.data_path = Path(data_path)
        self.annotation_format = annotation_format.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.cache_images = cache_images
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        self.logger = logger or get_logger("dataset")

        self.extensions = extensions or [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".webp",
        ]
        self.extensions = [ext.lower() for ext in self.extensions]

        self.samples = []
        self.class_to_idx = {}
        self.classes = class_names or []
        self.image_cache = {}

        self._load_dataset()

        if self.cache_images and self.cache_dir:
            self._setup_cache()

    def _load_dataset(self):
        if self.annotation_format == "imagefolder":
            self._load_imagefolder_format()
        elif self.annotation_format == "coco":
            self._load_coco_format()
        elif self.annotation_format == "csv":
            self._load_csv_format()
        elif self.annotation_format == "json":
            self._load_json_format()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")

        if self.max_samples:
            self.samples = self.samples[: self.max_samples]

        self.logger.info(f"Loaded {len(self.samples)} samples from {self.data_path}")
        self.logger.info(f"Number of classes: {len(self.classes)}")

    def _load_imagefolder_format(self):
        if not self.data_path.is_dir():
            raise ValueError(
                f"Data path must be a directory for ImageFolder format: {self.data_path}"
            )

        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        class_dirs.sort()

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), class_idx))

    def _load_coco_format(self):
        annotation_file = self.data_path / "annotations.json"
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {annotation_file}"
            )

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Build category mapping
        categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
        self.classes = list(categories.values())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build image id to filename mapping
        images = {img["id"]: img["file_name"] for img in coco_data["images"]}

        # Process annotations
        image_annotations = defaultdict(list)
        for ann in coco_data["annotations"]:
            image_annotations[ann["image_id"]].append(ann["category_id"])

        for img_id, filename in images.items():
            if img_id in image_annotations:
                # For multi-label, take the first category for now
                category_id = image_annotations[img_id][0]
                category_name = categories[category_id]
                class_idx = self.class_to_idx[category_name]

                img_path = self.data_path / "images" / filename
                if img_path.exists():
                    self.samples.append((str(img_path), class_idx))

    def _load_csv_format(self):
        csv_file = self.data_path / "annotations.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV annotation file not found: {csv_file}")

        df = pd.read_csv(csv_file)
        required_columns = ["image_path", "label"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        # Build class mapping
        unique_labels = sorted(df["label"].unique())
        self.classes = [str(label) for label in unique_labels]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for _, row in df.iterrows():
            img_path = self.data_path / row["image_path"]
            if img_path.exists():
                class_idx = self.class_to_idx[str(row["label"])]
                self.samples.append((str(img_path), class_idx))

    def _load_json_format(self):
        json_file = self.data_path / "annotations.json"
        if not json_file.exists():
            raise FileNotFoundError(f"JSON annotation file not found: {json_file}")

        with open(json_file, "r") as f:
            annotations = json.load(f)

        if isinstance(annotations, dict):
            # Format: {"image_path": "label", ...}
            samples_data = [(path, label) for path, label in annotations.items()]
        elif isinstance(annotations, list):
            # Format: [{"image_path": "path", "label": "label"}, ...]
            samples_data = [(item["image_path"], item["label"]) for item in annotations]
        else:
            raise ValueError("Invalid JSON format")

        # Build class mapping
        unique_labels = sorted(set(label for _, label in samples_data))
        self.classes = [str(label) for label in unique_labels]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for img_path, label in samples_data:
            full_path = self.data_path / img_path
            if full_path.exists():
                class_idx = self.class_to_idx[str(label)]
                self.samples.append((str(full_path), class_idx))

    def _setup_cache(self):
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_info_file = self.cache_dir / "cache_info.json"
        dataset_hash = self._compute_dataset_hash()

        cache_info = {}
        if cache_info_file.exists():
            with open(cache_info_file, "r") as f:
                cache_info = json.load(f)

        if cache_info.get("dataset_hash") != dataset_hash:
            self.logger.info("Dataset changed, rebuilding cache...")
            self._build_image_cache()

            cache_info = {
                "dataset_hash": dataset_hash,
                "num_samples": len(self.samples),
                "cache_dir": str(self.cache_dir),
            }

            with open(cache_info_file, "w") as f:
                json.dump(cache_info, f, indent=2)
        else:
            self.logger.info("Loading existing image cache...")
            self._load_image_cache()

    def _compute_dataset_hash(self) -> str:
        # Create hash based on samples list and modification times
        hash_data = []
        for img_path, label in self.samples[:100]:  # Sample first 100 for efficiency
            stat = os.stat(img_path)
            hash_data.append(f"{img_path}:{label}:{stat.st_mtime}")

        combined_str = "|".join(hash_data)
        return hashlib.md5(combined_str.encode()).hexdigest()

    def _build_image_cache(self):
        self.logger.info("Building image cache...")

        for idx, (img_path, label) in enumerate(self.samples):
            if idx % 1000 == 0:
                self.logger.info(f"Cached {idx}/{len(self.samples)} images")

            cache_file = self.cache_dir / f"image_{idx}.pkl"

            try:
                image = Image.open(img_path).convert("RGB")
                image_array = np.array(image)

                with open(cache_file, "wb") as f:
                    pickle.dump(image_array, f)

                self.image_cache[idx] = cache_file

            except Exception as e:
                self.logger.warning(f"Failed to cache image {img_path}: {e}")

        self.logger.info(f"Image cache built: {len(self.image_cache)} images cached")

    def _load_image_cache(self):
        for idx in range(len(self.samples)):
            cache_file = self.cache_dir / f"image_{idx}.pkl"
            if cache_file.exists():
                self.image_cache[idx] = cache_file

        self.logger.info(f"Loaded cache info for {len(self.image_cache)} images")

    def _load_image(self, idx: int) -> Image.Image:
        if self.cache_images and idx in self.image_cache:
            try:
                with open(self.image_cache[idx], "rb") as f:
                    image_array = pickle.load(f)
                return Image.fromarray(image_array)
            except Exception as e:
                self.logger.warning(f"Failed to load cached image {idx}: {e}")

        # Fallback to loading from disk
        img_path, _ = self.samples[idx]
        return Image.open(img_path).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        try:
            image = self._load_image(idx)
            _, target = self.samples[idx]

            if self.transform is not None:
                image = self.transform(image)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return image, target

        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the training loop
            dummy_image = Image.new("RGB", (224, 224), color="black")
            if self.transform is not None:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

    def get_class_distribution(self) -> Dict[str, int]:
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_name = self.classes[label]
            class_counts[class_name] += 1
        return dict(class_counts)

    def get_dataset_info(self) -> Dict[str, Any]:
        class_dist = self.get_class_distribution()

        return {
            "num_samples": len(self.samples),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "class_distribution": class_dist,
            "annotation_format": self.annotation_format,
            "data_path": str(self.data_path),
            "cached_images": len(self.image_cache) if self.cache_images else 0,
        }

    def cleanup_cache(self):
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("image_*.pkl"):
                cache_file.unlink()

            cache_info_file = self.cache_dir / "cache_info.json"
            if cache_info_file.exists():
                cache_info_file.unlink()

            self.logger.info("Image cache cleaned up")


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)

        if len(self.weights) != len(self.datasets):
            raise ValueError("Number of weights must match number of datasets")

        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = np.cumsum([0] + self.dataset_sizes)

        # Weighted sampling setup
        total_weight = sum(self.weights)
        self.normalized_weights = [w / total_weight for w in self.weights]

        # Calculate effective dataset size based on largest dataset
        self.effective_size = max(self.dataset_sizes)

    def __len__(self) -> int:
        return self.effective_size

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        # Choose dataset based on weights
        dataset_idx = np.random.choice(len(self.datasets), p=self.normalized_weights)
        dataset = self.datasets[dataset_idx]

        # Choose sample from selected dataset
        sample_idx = idx % len(dataset)
        return dataset[sample_idx]


def create_dataset(
    data_path: Union[str, Path],
    annotation_format: str = "imagefolder",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cache_images: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    **kwargs,
) -> CustomDINOv3Dataset:
    return CustomDINOv3Dataset(
        data_path=data_path,
        annotation_format=annotation_format,
        transform=transform,
        target_transform=target_transform,
        cache_images=cache_images,
        cache_dir=cache_dir,
        max_samples=max_samples,
        class_names=class_names,
        **kwargs,
    )

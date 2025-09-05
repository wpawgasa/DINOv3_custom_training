import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
from torchvision import transforms

from ..utils.logging import get_logger


class ImageQualityChecker:
    def __init__(
        self,
        min_size: Tuple[int, int] = (32, 32),
        max_size: Tuple[int, int] = (4096, 4096),
        min_quality_score: float = 0.3,
        check_corruption: bool = True,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_quality_score = min_quality_score
        self.check_corruption = check_corruption

    def check_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        quality_info = {
            "is_valid": True,
            "size_valid": True,
            "quality_score": 1.0,
            "is_corrupted": False,
            "warnings": [],
        }

        # Check size constraints
        width, height = image.size
        if width < self.min_size[0] or height < self.min_size[1]:
            quality_info["size_valid"] = False
            quality_info["is_valid"] = False
            quality_info["warnings"].append(f"Image too small: {width}x{height}")

        if width > self.max_size[0] or height > self.max_size[1]:
            quality_info["size_valid"] = False
            quality_info["is_valid"] = False
            quality_info["warnings"].append(f"Image too large: {width}x{height}")

        # Check for corruption
        if self.check_corruption:
            try:
                image.verify()
            except Exception as e:
                quality_info["is_corrupted"] = True
                quality_info["is_valid"] = False
                quality_info["warnings"].append(f"Image corrupted: {str(e)}")
                return quality_info

        # Calculate quality metrics
        quality_score = self._calculate_quality_score(image)
        quality_info["quality_score"] = quality_score

        if quality_score < self.min_quality_score:
            quality_info["is_valid"] = False
            quality_info["warnings"].append(f"Low quality score: {quality_score:.3f}")

        return quality_info

    def _calculate_quality_score(self, image: Image.Image) -> float:
        # Convert to numpy array
        img_array = np.array(image)

        # Calculate sharpness (variance of Laplacian)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize sharpness score (empirically determined ranges)
        sharpness_score = min(sharpness / 1000.0, 1.0)

        # Calculate brightness and contrast
        stat = ImageStat.Stat(image)

        if len(stat.mean) == 3:  # RGB
            brightness = sum(stat.mean) / 3.0 / 255.0
        else:  # Grayscale
            brightness = stat.mean[0] / 255.0

        # Optimal brightness is around 0.4-0.6
        brightness_score = 1.0 - abs(brightness - 0.5) * 2
        brightness_score = max(0, brightness_score)

        # Combined quality score
        quality_score = sharpness_score * 0.7 + brightness_score * 0.3
        return min(quality_score, 1.0)


class ImagePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        maintain_aspect_ratio: bool = True,
        padding_mode: str = "constant",  # constant, reflect, replicate, circular
        padding_fill: Union[int, Tuple[int, ...]] = 0,
        quality_checker: Optional[ImageQualityChecker] = None,
        logger=None,
    ):
        self.target_size = target_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill
        self.quality_checker = quality_checker or ImageQualityChecker()
        self.logger = logger or get_logger("preprocessor")

        self.stats = {
            "processed_count": 0,
            "filtered_count": 0,
            "converted_count": 0,
            "resized_count": 0,
        }

    def preprocess_image(
        self,
        image: Union[Image.Image, str, Path],
        validate_quality: bool = True,
        convert_grayscale: bool = True,
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        # Load image if path provided
        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except Exception as e:
                return None, {"error": f"Failed to load image: {e}"}

        preprocessing_info = {
            "original_size": image.size,
            "original_mode": image.mode,
            "quality_info": {},
            "operations": [],
        }

        # Quality validation
        if validate_quality:
            quality_info = self.quality_checker.check_image_quality(image)
            preprocessing_info["quality_info"] = quality_info

            if not quality_info["is_valid"]:
                self.stats["filtered_count"] += 1
                return None, preprocessing_info

        # Convert grayscale to RGB
        if convert_grayscale and image.mode in ["L", "LA"]:
            image = self._grayscale_to_rgb(image)
            preprocessing_info["operations"].append("grayscale_to_rgb")
            self.stats["converted_count"] += 1

        # Convert other modes to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            preprocessing_info["operations"].append("convert_to_rgb")

        # Resize image
        if image.size != self.target_size:
            image = self._resize_image(image)
            preprocessing_info["operations"].append("resize")
            self.stats["resized_count"] += 1

        preprocessing_info["final_size"] = image.size
        preprocessing_info["final_mode"] = image.mode

        self.stats["processed_count"] += 1
        return image, preprocessing_info

    def _grayscale_to_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode == "L":
            # Convert grayscale to RGB by duplicating channels
            return Image.merge("RGB", (image, image, image))
        elif image.mode == "LA":
            # Convert grayscale+alpha to RGB
            alpha = image.split()[-1]
            gray = image.convert("L")
            return Image.merge("RGB", (gray, gray, gray))
        else:
            return image.convert("RGB")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if not self.maintain_aspect_ratio:
            return image.resize(self.target_size, Image.LANCZOS)

        # Calculate scaling factor to maintain aspect ratio
        original_width, original_height = image.size
        target_width, target_height = self.target_size

        scale = min(target_width / original_width, target_height / original_height)

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Add padding if necessary
        if (new_width, new_height) != self.target_size:
            resized_image = self._add_padding(resized_image)

        return resized_image

    def _add_padding(self, image: Image.Image) -> Image.Image:
        current_width, current_height = image.size
        target_width, target_height = self.target_size

        pad_width = target_width - current_width
        pad_height = target_height - current_height

        # Calculate padding for each side
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top

        # Apply padding based on mode
        if self.padding_mode == "constant":
            padding = (left, top, right, bottom)
            return ImageOps.expand(image, padding, fill=self.padding_fill)
        else:
            # For other padding modes, use torchvision transforms
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Pad(
                        (left, top, right, bottom), padding_mode=self.padding_mode
                    ),
                    transforms.ToPILImage(),
                ]
            )
            return transform(image)

    def batch_preprocess(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Path] = None,
        validate_quality: bool = True,
        convert_grayscale: bool = True,
        save_stats: bool = True,
    ) -> Dict[str, Any]:
        results = {"successful": [], "failed": [], "filtered": [], "total_processed": 0}

        for i, img_path in enumerate(image_paths):
            if i % 100 == 0:
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}")

            processed_image, info = self.preprocess_image(
                img_path, validate_quality, convert_grayscale
            )

            if processed_image is not None:
                results["successful"].append({"path": str(img_path), "info": info})

                # Save processed image if output directory specified
                if output_dir:
                    output_path = output_dir / Path(img_path).name
                    processed_image.save(output_path)

            elif "error" in info:
                results["failed"].append(
                    {"path": str(img_path), "error": info["error"]}
                )
            else:
                results["filtered"].append(
                    {
                        "path": str(img_path),
                        "quality_info": info.get("quality_info", {}),
                    }
                )

        results["total_processed"] = len(image_paths)
        results["stats"] = self.get_stats()

        if save_stats and output_dir:
            self._save_processing_stats(output_dir, results)

        return results

    def _save_processing_stats(self, output_dir: Path, results: Dict[str, Any]):
        import json

        stats_file = output_dir / "preprocessing_stats.json"
        with open(stats_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Processing statistics saved to {stats_file}")

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()

    def reset_stats(self):
        self.stats = {
            "processed_count": 0,
            "filtered_count": 0,
            "converted_count": 0,
            "resized_count": 0,
        }


class DatasetPreprocessor:
    def __init__(
        self, image_preprocessor: Optional[ImagePreprocessor] = None, logger=None
    ):
        self.image_preprocessor = image_preprocessor or ImagePreprocessor()
        self.logger = logger or get_logger("dataset_preprocessor")

    def preprocess_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        annotation_format: str = "imagefolder",
        validate_quality: bool = True,
        convert_grayscale: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        extensions = [ext.lower() for ext in extensions]

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if annotation_format == "imagefolder":
            return self._preprocess_imagefolder_dataset(
                input_dir, output_dir, extensions, validate_quality, convert_grayscale
            )
        else:
            return self._preprocess_flat_dataset(
                input_dir, output_dir, extensions, validate_quality, convert_grayscale
            )

    def _preprocess_imagefolder_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: List[str],
        validate_quality: bool,
        convert_grayscale: bool,
    ) -> Dict[str, Any]:
        results = {
            "classes_processed": {},
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "filtered": 0,
        }

        # Process each class directory
        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Processing class: {class_name}")

            # Get all images in class directory
            image_paths = []
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    image_paths.append(img_path)

            # Process images
            class_results = self.image_preprocessor.batch_preprocess(
                image_paths,
                output_class_dir,
                validate_quality,
                convert_grayscale,
                save_stats=False,
            )

            results["classes_processed"][class_name] = class_results
            results["total_images"] += len(image_paths)
            results["successful"] += len(class_results["successful"])
            results["failed"] += len(class_results["failed"])
            results["filtered"] += len(class_results["filtered"])

        # Save overall stats
        self._save_dataset_stats(output_dir, results)

        return results

    def _preprocess_flat_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: List[str],
        validate_quality: bool,
        convert_grayscale: bool,
    ) -> Dict[str, Any]:
        # Get all images
        image_paths = []
        for img_path in input_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in extensions:
                image_paths.append(img_path)

        # Process images
        results = self.image_preprocessor.batch_preprocess(
            image_paths,
            output_dir,
            validate_quality,
            convert_grayscale,
            save_stats=False,
        )

        # Save stats
        self._save_dataset_stats(output_dir, results)

        return results

    def _save_dataset_stats(self, output_dir: Path, results: Dict[str, Any]):
        import json

        stats_file = output_dir / "dataset_preprocessing_stats.json"
        with open(stats_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Dataset preprocessing statistics saved to {stats_file}")


def create_preprocessor(
    target_size: Tuple[int, int] = (224, 224),
    maintain_aspect_ratio: bool = True,
    min_quality_score: float = 0.3,
    **kwargs,
) -> ImagePreprocessor:
    quality_checker = ImageQualityChecker(
        min_quality_score=min_quality_score,
        **{k: v for k, v in kwargs.items() if k.startswith("quality_")},
    )

    return ImagePreprocessor(
        target_size=target_size,
        maintain_aspect_ratio=maintain_aspect_ratio,
        quality_checker=quality_checker,
        **{k: v for k, v in kwargs.items() if not k.startswith("quality_")},
    )

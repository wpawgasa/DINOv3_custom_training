import hashlib
import json
import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import Dataset

from ..utils.logging import get_logger
from .dataset import CustomDINOv3Dataset


class DatasetValidator:
    """Comprehensive dataset validation and quality analysis."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("dataset_validator")
        self.validation_results = {}

    def validate_dataset(
        self,
        dataset: Union[Dataset, Path, str],
        annotation_format: str = "imagefolder",
        sample_size: Optional[int] = None,
        check_duplicates: bool = True,
        check_corruption: bool = True,
        generate_report: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if isinstance(dataset, (str, Path)):
            dataset = CustomDINOv3Dataset(
                data_path=dataset, annotation_format=annotation_format
            )

        self.logger.info(f"Validating dataset with {len(dataset)} samples")

        results = {
            "basic_stats": self._get_basic_statistics(dataset),
            "class_distribution": self._analyze_class_distribution(dataset),
            "image_quality": self._analyze_image_quality(dataset, sample_size),
            "corruption_check": {},
            "duplicate_check": {},
            "integrity_check": self._check_data_integrity(dataset),
        }

        if check_corruption:
            results["corruption_check"] = self._check_image_corruption(
                dataset, sample_size
            )

        if check_duplicates:
            results["duplicate_check"] = self._check_duplicates(dataset, sample_size)

        # Overall health score
        results["health_score"] = self._calculate_health_score(results)

        if generate_report and output_dir:
            self._generate_validation_report(results, output_dir)

        self.validation_results = results
        return results

    def _get_basic_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        dataset_info = (
            dataset.get_dataset_info() if hasattr(dataset, "get_dataset_info") else {}
        )

        return {
            "total_samples": len(dataset),
            "num_classes": len(dataset.classes) if hasattr(dataset, "classes") else 0,
            "class_names": dataset.classes if hasattr(dataset, "classes") else [],
            "annotation_format": getattr(dataset, "annotation_format", "unknown"),
            "data_path": str(getattr(dataset, "data_path", "unknown")),
            **dataset_info,
        }

    def _analyze_class_distribution(self, dataset: Dataset) -> Dict[str, Any]:
        if not hasattr(dataset, "get_class_distribution"):
            return {}

        class_dist = dataset.get_class_distribution()

        if not class_dist:
            return {}

        total_samples = sum(class_dist.values())
        class_percentages = {
            k: (v / total_samples) * 100 for k, v in class_dist.items()
        }

        # Calculate balance metrics
        counts = list(class_dist.values())
        balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0

        # Entropy-based balance measure
        probs = np.array(list(class_percentages.values())) / 100
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        max_entropy = np.log2(len(class_dist))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            "distribution": class_dist,
            "percentages": class_percentages,
            "balance_ratio": balance_ratio,
            "entropy_score": normalized_entropy,
            "most_frequent_class": max(class_dist, key=class_dist.get),
            "least_frequent_class": min(class_dist, key=class_dist.get),
            "is_balanced": balance_ratio > 0.5,  # Threshold for "balanced"
        }

    def _analyze_image_quality(
        self, dataset: Dataset, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if sample_size is None:
            sample_size = min(1000, len(dataset))

        # Sample indices
        if sample_size < len(dataset):
            indices = np.random.choice(len(dataset), sample_size, replace=False)
        else:
            indices = range(len(dataset))

        image_stats = {
            "sizes": [],
            "aspect_ratios": [],
            "modes": [],
            "channels": [],
            "file_sizes": [],
        }

        corrupted_count = 0

        for idx in indices:
            try:
                if hasattr(dataset, "samples"):
                    img_path, _ = dataset.samples[idx]
                    image = Image.open(img_path)
                else:
                    image, _ = dataset[idx]
                    if hasattr(image, "size"):
                        pass  # PIL Image
                    else:
                        continue  # Skip tensor images

                width, height = image.size
                image_stats["sizes"].append((width, height))
                image_stats["aspect_ratios"].append(width / height)
                image_stats["modes"].append(image.mode)

                if image.mode == "RGB":
                    image_stats["channels"].append(3)
                elif image.mode == "L":
                    image_stats["channels"].append(1)
                else:
                    image_stats["channels"].append(len(image.getbands()))

                # File size if we have the path
                if hasattr(dataset, "samples"):
                    try:
                        file_size = os.path.getsize(img_path)
                        image_stats["file_sizes"].append(file_size)
                    except:
                        pass

            except Exception as e:
                corrupted_count += 1
                self.logger.warning(f"Error processing image at index {idx}: {e}")

        # Calculate statistics
        sizes = image_stats["sizes"]
        aspect_ratios = image_stats["aspect_ratios"]

        quality_stats = {
            "samples_analyzed": len(sizes),
            "corrupted_samples": corrupted_count,
            "unique_sizes": len(set(sizes)),
            "size_distribution": {
                "min_width": min(w for w, h in sizes) if sizes else 0,
                "max_width": max(w for w, h in sizes) if sizes else 0,
                "min_height": min(h for w, h in sizes) if sizes else 0,
                "max_height": max(h for w, h in sizes) if sizes else 0,
                "mean_width": np.mean([w for w, h in sizes]) if sizes else 0,
                "mean_height": np.mean([h for w, h in sizes]) if sizes else 0,
            },
            "aspect_ratio_stats": {
                "min": min(aspect_ratios) if aspect_ratios else 0,
                "max": max(aspect_ratios) if aspect_ratios else 0,
                "mean": np.mean(aspect_ratios) if aspect_ratios else 0,
                "std": np.std(aspect_ratios) if aspect_ratios else 0,
            },
            "mode_distribution": dict(Counter(image_stats["modes"])),
            "channel_distribution": dict(Counter(image_stats["channels"])),
        }

        if image_stats["file_sizes"]:
            quality_stats["file_size_stats"] = {
                "min_mb": min(image_stats["file_sizes"]) / (1024 * 1024),
                "max_mb": max(image_stats["file_sizes"]) / (1024 * 1024),
                "mean_mb": np.mean(image_stats["file_sizes"]) / (1024 * 1024),
                "total_gb": sum(image_stats["file_sizes"]) / (1024 * 1024 * 1024),
            }

        return quality_stats

    def _check_image_corruption(
        self, dataset: Dataset, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if not hasattr(dataset, "samples"):
            return {"checked": False, "reason": "Dataset does not expose file paths"}

        if sample_size is None:
            sample_size = min(1000, len(dataset.samples))

        if sample_size < len(dataset.samples):
            indices = np.random.choice(len(dataset.samples), sample_size, replace=False)
        else:
            indices = range(len(dataset.samples))

        corrupted_files = []

        for idx in indices:
            img_path, _ = dataset.samples[idx]
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                corrupted_files.append({"path": img_path, "error": str(e)})

        return {
            "checked": True,
            "samples_checked": len(indices),
            "corrupted_count": len(corrupted_files),
            "corruption_rate": len(corrupted_files) / len(indices),
            "corrupted_files": corrupted_files[:10],  # Limit to first 10 for reporting
        }

    def _check_duplicates(
        self, dataset: Dataset, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        if not hasattr(dataset, "samples"):
            return {"checked": False, "reason": "Dataset does not expose file paths"}

        if sample_size is None:
            sample_size = min(1000, len(dataset.samples))

        if sample_size < len(dataset.samples):
            indices = np.random.choice(len(dataset.samples), sample_size, replace=False)
        else:
            indices = range(len(dataset.samples))

        file_hashes = {}
        duplicates = []

        for idx in indices:
            img_path, label = dataset.samples[idx]
            try:
                with open(img_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in file_hashes:
                    duplicates.append(
                        {
                            "original": file_hashes[file_hash],
                            "duplicate": {"path": img_path, "label": label},
                        }
                    )
                else:
                    file_hashes[file_hash] = {"path": img_path, "label": label}

            except Exception as e:
                self.logger.warning(f"Error computing hash for {img_path}: {e}")

        return {
            "checked": True,
            "samples_checked": len(indices),
            "duplicate_count": len(duplicates),
            "duplication_rate": len(duplicates) / len(indices),
            "duplicates": duplicates[:10],  # Limit for reporting
        }

    def _check_data_integrity(self, dataset: Dataset) -> Dict[str, Any]:
        integrity_issues = []

        # Check if all classes have samples
        if hasattr(dataset, "classes") and hasattr(dataset, "get_class_distribution"):
            class_dist = dataset.get_class_distribution()
            empty_classes = [
                cls for cls in dataset.classes if class_dist.get(cls, 0) == 0
            ]
            if empty_classes:
                integrity_issues.append(f"Empty classes found: {empty_classes}")

        # Check for extremely imbalanced classes
        if hasattr(dataset, "get_class_distribution"):
            class_dist = dataset.get_class_distribution()
            if class_dist:
                counts = list(class_dist.values())
                if max(counts) / min(counts) > 100:  # 100:1 ratio threshold
                    integrity_issues.append("Extremely imbalanced dataset detected")

        # Check dataset size
        if len(dataset) < 10:
            integrity_issues.append("Very small dataset (< 10 samples)")

        return {
            "is_valid": len(integrity_issues) == 0,
            "issues": integrity_issues,
            "total_issues": len(integrity_issues),
        }

    def _calculate_health_score(self, results: Dict[str, Any]) -> float:
        score = 100.0

        # Deduct for corruption
        if results["corruption_check"].get("checked"):
            corruption_rate = results["corruption_check"]["corruption_rate"]
            score -= corruption_rate * 30  # Up to -30 points for corruption

        # Deduct for duplicates
        if results["duplicate_check"].get("checked"):
            duplicate_rate = results["duplicate_check"]["duplication_rate"]
            score -= duplicate_rate * 20  # Up to -20 points for duplicates

        # Deduct for class imbalance
        class_dist = results["class_distribution"]
        if class_dist and "balance_ratio" in class_dist:
            balance_penalty = (1 - class_dist["balance_ratio"]) * 25
            score -= balance_penalty

        # Deduct for integrity issues
        integrity_issues = results["integrity_check"]["total_issues"]
        score -= integrity_issues * 10

        return max(0.0, min(100.0, score))

    def _generate_validation_report(self, results: Dict[str, Any], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_report_path = output_dir / "dataset_validation_report.json"
        with open(json_report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate visualizations
        self._create_validation_plots(results, output_dir)

        # Generate text summary
        self._create_text_summary(results, output_dir)

        self.logger.info(f"Validation report saved to {output_dir}")

    def _create_validation_plots(self, results: Dict[str, Any], output_dir: Path):
        # Class distribution plot
        class_dist = results["class_distribution"]
        if class_dist and "distribution" in class_dist:
            plt.figure(figsize=(12, 6))
            classes = list(class_dist["distribution"].keys())
            counts = list(class_dist["distribution"].values())

            plt.subplot(1, 2, 1)
            plt.bar(classes, counts)
            plt.xticks(rotation=45, ha="right")
            plt.title("Class Distribution (Absolute)")
            plt.ylabel("Number of Samples")

            plt.subplot(1, 2, 2)
            percentages = list(class_dist["percentages"].values())
            plt.pie(percentages, labels=classes, autopct="%1.1f%%")
            plt.title("Class Distribution (Percentage)")

            plt.tight_layout()
            plt.savefig(
                output_dir / "class_distribution.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # Image quality plots
        quality_stats = results["image_quality"]
        if quality_stats and "aspect_ratio_stats" in quality_stats:
            plt.figure(figsize=(10, 4))

            # File size distribution (if available)
            if "file_size_stats" in quality_stats:
                plt.subplot(1, 2, 1)
                # This would need the raw data to create a proper histogram
                # For now, just show summary stats
                stats = quality_stats["file_size_stats"]
                plt.text(
                    0.1,
                    0.7,
                    f"Min: {stats['min_mb']:.2f} MB",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    0.1,
                    0.5,
                    f"Max: {stats['max_mb']:.2f} MB",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    0.1,
                    0.3,
                    f"Mean: {stats['mean_mb']:.2f} MB",
                    transform=plt.gca().transAxes,
                )
                plt.text(
                    0.1,
                    0.1,
                    f"Total: {stats['total_gb']:.2f} GB",
                    transform=plt.gca().transAxes,
                )
                plt.title("File Size Statistics")
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.axis("off")

            # Mode distribution
            plt.subplot(1, 2, 2)
            modes = list(quality_stats["mode_distribution"].keys())
            mode_counts = list(quality_stats["mode_distribution"].values())
            plt.bar(modes, mode_counts)
            plt.title("Image Mode Distribution")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.savefig(
                output_dir / "image_quality_stats.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _create_text_summary(self, results: Dict[str, Any], output_dir: Path):
        summary_path = output_dir / "validation_summary.txt"

        with open(summary_path, "w") as f:
            f.write("DATASET VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            # Health Score
            f.write(f"Overall Health Score: {results['health_score']:.1f}/100\n\n")

            # Basic Stats
            basic = results["basic_stats"]
            f.write("BASIC STATISTICS:\n")
            f.write(f"- Total Samples: {basic['total_samples']}\n")
            f.write(f"- Number of Classes: {basic['num_classes']}\n")
            f.write(f"- Annotation Format: {basic['annotation_format']}\n\n")

            # Class Distribution
            class_dist = results["class_distribution"]
            if class_dist:
                f.write("CLASS DISTRIBUTION:\n")
                f.write(f"- Balance Ratio: {class_dist['balance_ratio']:.3f}\n")
                f.write(f"- Entropy Score: {class_dist['entropy_score']:.3f}\n")
                f.write(f"- Is Balanced: {class_dist['is_balanced']}\n")
                f.write(f"- Most Frequent: {class_dist['most_frequent_class']}\n")
                f.write(f"- Least Frequent: {class_dist['least_frequent_class']}\n\n")

            # Quality Issues
            f.write("QUALITY ISSUES:\n")
            if results["corruption_check"].get("checked"):
                corr = results["corruption_check"]
                f.write(f"- Corruption Rate: {corr['corruption_rate']:.3%}\n")

            if results["duplicate_check"].get("checked"):
                dup = results["duplicate_check"]
                f.write(f"- Duplication Rate: {dup['duplication_rate']:.3%}\n")

            integrity = results["integrity_check"]
            if not integrity["is_valid"]:
                f.write(f"- Integrity Issues: {integrity['total_issues']}\n")
                for issue in integrity["issues"]:
                    f.write(f"  * {issue}\n")

            f.write("\n")

        self.logger.info(f"Text summary saved to {summary_path}")


def validate_dataset(
    dataset_path: Union[str, Path],
    annotation_format: str = "imagefolder",
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to validate a dataset."""
    validator = DatasetValidator()

    if output_dir is None:
        output_dir = Path(dataset_path).parent / "validation_report"

    return validator.validate_dataset(
        dataset_path,
        annotation_format=annotation_format,
        output_dir=output_dir,
        **kwargs,
    )

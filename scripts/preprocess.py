#!/usr/bin/env python3
"""
DINOv3 Data Preprocessing Script

Comprehensive data preprocessing and dataset management script with support for
multiple input formats, quality analysis, and dataset conversion utilities.
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from PIL import Image, ImageStat
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf

from data.preprocessing import ImagePreprocessor
from data.validation import DatasetValidator
from utils.logging import setup_logging
from utils.reproducibility import set_seed


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for DINOv3 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input and output paths
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input directory containing raw dataset"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for processed dataset"
    )
    
    # Processing modes
    parser.add_argument(
        "--mode",
        type=str,
        default="preprocess",
        choices=["preprocess", "convert", "validate", "analyze", "split"],
        help="Processing mode"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="imagefolder",
        choices=["imagefolder", "coco", "csv", "json", "raw"],
        help="Input dataset format"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="imagefolder",
        choices=["imagefolder", "coco", "csv", "json"],
        help="Output dataset format"
    )
    
    # Image processing
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Target image size (height width)"
    )
    parser.add_argument(
        "--resize-method",
        type=str,
        default="resize_shortest",
        choices=["resize", "resize_shortest", "center_crop", "resize_crop"],
        help="Resize method"
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=50,
        help="Minimum image quality (0-100)"
    )
    parser.add_argument(
        "--convert-grayscale",
        action="store_true",
        help="Convert grayscale images to RGB"
    )
    
    # Data filtering and validation
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help="Minimum number of samples per class"
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Maximum number of samples per class"
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate image integrity"
    )
    parser.add_argument(
        "--remove-corrupted",
        action="store_true",
        help="Remove corrupted images"
    )
    
    # Dataset splitting
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio for dataset splitting"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify splits by class"
    )
    
    # Performance and processing
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count(),
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Processing batch size"
    )
    parser.add_argument(
        "--cache-preprocessed",
        action="store_true",
        help="Cache preprocessed images"
    )
    
    # Analysis and reporting
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate dataset analysis report"
    )
    parser.add_argument(
        "--sample-visualizations",
        type=int,
        default=0,
        help="Number of sample visualizations to generate"
    )
    
    # Debug and development
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - analyze without processing"
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        help="Limit number of samples for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def discover_dataset_structure(input_dir: Path, input_format: str) -> Dict[str, Any]:
    """Discover and analyze dataset structure."""
    print(f"Discovering dataset structure in: {input_dir}")
    
    structure = {
        'format': input_format,
        'total_files': 0,
        'image_files': 0,
        'classes': [],
        'class_counts': {},
        'file_extensions': {},
        'directory_structure': {}
    }
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    if input_format == "imagefolder":
        # ImageFolder structure: root/class/image.jpg
        for class_dir in input_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                structure['classes'].append(class_name)
                
                class_files = []
                for file_path in class_dir.rglob('*'):
                    if file_path.is_file():
                        structure['total_files'] += 1
                        ext = file_path.suffix.lower()
                        structure['file_extensions'][ext] = structure['file_extensions'].get(ext, 0) + 1
                        
                        if ext in image_extensions:
                            structure['image_files'] += 1
                            class_files.append(file_path)
                
                structure['class_counts'][class_name] = len(class_files)
                structure['directory_structure'][class_name] = class_files
    
    elif input_format == "raw":
        # Raw directory with mixed files
        for file_path in input_dir.rglob('*'):
            if file_path.is_file():
                structure['total_files'] += 1
                ext = file_path.suffix.lower()
                structure['file_extensions'][ext] = structure['file_extensions'].get(ext, 0) + 1
                
                if ext in image_extensions:
                    structure['image_files'] += 1
    
    # TODO: Implement COCO, CSV, JSON format discovery
    elif input_format in ["coco", "csv", "json"]:
        print(f"Format '{input_format}' discovery not yet implemented")
    
    print(f"Discovery complete:")
    print(f"  Total files: {structure['total_files']}")
    print(f"  Image files: {structure['image_files']}")
    print(f"  Classes found: {len(structure['classes'])}")
    
    return structure


def analyze_image_quality(image_path: Path) -> Dict[str, float]:
    """Analyze individual image quality metrics."""
    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Basic statistics
            stat = ImageStat.Stat(img)
            
            # Calculate metrics
            width, height = img.size
            aspect_ratio = width / height
            
            # Brightness (mean of means)
            brightness = np.mean(stat.mean)
            
            # Contrast (standard deviation)
            contrast = np.mean(stat.stddev)
            
            # Sharpness using Laplacian variance
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': laplacian_var,
                'file_size': image_path.stat().st_size,
                'is_valid': True
            }
    
    except Exception as e:
        return {
            'width': 0,
            'height': 0,
            'aspect_ratio': 0,
            'brightness': 0,
            'contrast': 0,
            'sharpness': 0,
            'file_size': 0,
            'is_valid': False,
            'error': str(e)
        }


def analyze_dataset_quality(
    dataset_structure: Dict[str, Any],
    num_workers: int = 4,
    limit_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Analyze dataset quality with parallel processing."""
    print("Analyzing dataset quality...")
    
    if dataset_structure['format'] != 'imagefolder':
        print("Quality analysis currently only supports ImageFolder format")
        return {}
    
    # Collect all image paths
    all_image_paths = []
    for class_name, file_paths in dataset_structure['directory_structure'].items():
        all_image_paths.extend(file_paths)
    
    if limit_samples:
        all_image_paths = all_image_paths[:limit_samples]
    
    print(f"Analyzing {len(all_image_paths)} images with {num_workers} workers...")
    
    # Parallel quality analysis
    quality_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {
            executor.submit(analyze_image_quality, path): path 
            for path in all_image_paths
        }
        
        for future in tqdm(as_completed(future_to_path), total=len(all_image_paths)):
            image_path = future_to_path[future]
            try:
                result = future.result()
                result['path'] = str(image_path)
                result['class'] = image_path.parent.name
                quality_results.append(result)
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(quality_results)
    
    # Calculate summary statistics
    summary_stats = {
        'total_images': len(df),
        'valid_images': df['is_valid'].sum(),
        'corrupted_images': (~df['is_valid']).sum(),
        'resolution_stats': {
            'min_width': df['width'].min(),
            'max_width': df['width'].max(),
            'mean_width': df['width'].mean(),
            'min_height': df['height'].min(),
            'max_height': df['height'].max(),
            'mean_height': df['height'].mean(),
        },
        'quality_stats': {
            'mean_brightness': df['brightness'].mean(),
            'std_brightness': df['brightness'].std(),
            'mean_contrast': df['contrast'].mean(),
            'std_contrast': df['contrast'].std(),
            'mean_sharpness': df['sharpness'].mean(),
            'std_sharpness': df['sharpness'].std(),
        },
        'file_size_stats': {
            'min_size': df['file_size'].min(),
            'max_size': df['file_size'].max(),
            'mean_size': df['file_size'].mean(),
            'total_size_gb': df['file_size'].sum() / (1024**3),
        }
    }
    
    # Per-class statistics
    class_stats = df.groupby('class').agg({
        'is_valid': 'sum',
        'width': ['min', 'max', 'mean'],
        'height': ['min', 'max', 'mean'],
        'brightness': 'mean',
        'contrast': 'mean',
        'sharpness': 'mean'
    }).round(2).to_dict()
    
    return {
        'summary_stats': summary_stats,
        'class_stats': class_stats,
        'detailed_results': quality_results if len(quality_results) < 1000 else None,
        'corrupted_files': df[~df['is_valid']]['path'].tolist()
    }


def preprocess_single_image(args_tuple) -> Tuple[bool, str, Dict]:
    """Process a single image (for multiprocessing)."""
    image_path, output_path, processing_config = args_tuple
    
    try:
        # Create preprocessor
        preprocessor = ImagePreprocessor(
            target_size=processing_config['target_size'],
            resize_method=processing_config['resize_method'],
            quality_threshold=processing_config['quality_threshold']
        )
        
        # Process image
        processed_image = preprocessor.process_image(image_path)
        
        if processed_image is not None:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed image
            processed_image.save(output_path, 'JPEG', quality=95)
            
            return True, str(image_path), {'status': 'success', 'output_path': str(output_path)}
        else:
            return False, str(image_path), {'status': 'failed', 'reason': 'preprocessing_failed'}
    
    except Exception as e:
        return False, str(image_path), {'status': 'error', 'error': str(e)}


def preprocess_dataset(
    dataset_structure: Dict[str, Any],
    output_dir: Path,
    processing_config: Dict[str, Any],
    num_workers: int = 4,
    dry_run: bool = False,
    limit_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Preprocess entire dataset with parallel processing."""
    print("Preprocessing dataset...")
    
    if dataset_structure['format'] != 'imagefolder':
        print("Preprocessing currently only supports ImageFolder format")
        return {}
    
    # Prepare processing tasks
    processing_tasks = []
    for class_name, file_paths in dataset_structure['directory_structure'].items():
        class_output_dir = output_dir / class_name
        
        for input_path in file_paths[:limit_samples] if limit_samples else file_paths:
            output_path = class_output_dir / input_path.name
            processing_tasks.append((input_path, output_path, processing_config))
    
    print(f"Processing {len(processing_tasks)} images with {num_workers} workers...")
    
    if dry_run:
        print("Dry run - would process the following:")
        for task in processing_tasks[:10]:  # Show first 10
            print(f"  {task[0]} -> {task[1]}")
        if len(processing_tasks) > 10:
            print(f"  ... and {len(processing_tasks) - 10} more files")
        return {'total_tasks': len(processing_tasks), 'dry_run': True}
    
    # Process images in parallel
    results = {'successful': 0, 'failed': 0, 'errors': []}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(preprocess_single_image, task): task 
            for task in processing_tasks
        }
        
        for future in tqdm(as_completed(future_to_task), total=len(processing_tasks)):
            task = future_to_task[future]
            try:
                success, image_path, result_info = future.result()
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'path': image_path,
                        'info': result_info
                    })
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'path': str(task[0]),
                    'error': str(e)
                })
    
    print(f"Processing complete: {results['successful']} successful, {results['failed']} failed")
    return results


def split_dataset(
    dataset_structure: Dict[str, Any],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stratify: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """Split dataset into train/val/test sets."""
    print(f"Splitting dataset: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")
    
    set_seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    split_stats = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name, file_paths in dataset_structure['directory_structure'].items():
        # Shuffle files
        file_paths = list(file_paths)
        np.random.shuffle(file_paths)
        
        # Calculate split indices
        n_files = len(file_paths)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        n_test = n_files - n_train - n_val
        
        # Split files
        train_files = file_paths[:n_train]
        val_files = file_paths[n_train:n_train + n_val]
        test_files = file_paths[n_train + n_val:]
        
        # Create class directories in each split
        for split in ['train', 'val', 'test']:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        # Copy files to appropriate splits
        for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file_path in files:
                output_path = output_dir / split / class_name / file_path.name
                shutil.copy2(file_path, output_path)
            
            split_stats[split][class_name] = len(files)
    
    print("Dataset split complete:")
    for split in ['train', 'val', 'test']:
        total = sum(split_stats[split].values())
        print(f"  {split}: {total} files")
    
    return split_stats


def generate_dataset_report(
    dataset_structure: Dict[str, Any],
    quality_analysis: Dict[str, Any],
    output_dir: Path,
    sample_visualizations: int = 0
) -> Dict[str, Any]:
    """Generate comprehensive dataset analysis report."""
    print("Generating dataset report...")
    
    report = {
        'dataset_overview': {
            'format': dataset_structure['format'],
            'total_files': dataset_structure['total_files'],
            'image_files': dataset_structure['image_files'],
            'num_classes': len(dataset_structure['classes']),
            'classes': dataset_structure['classes'],
            'class_distribution': dataset_structure['class_counts']
        },
        'file_types': dataset_structure['file_extensions'],
        'quality_analysis': quality_analysis
    }
    
    # Save report as JSON
    with open(output_dir / 'dataset_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate summary statistics file
    if quality_analysis:
        summary_df = pd.DataFrame([quality_analysis['summary_stats']])
        summary_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    
    # Generate class statistics file
    class_dist_df = pd.DataFrame(list(dataset_structure['class_counts'].items()), 
                                columns=['class', 'count'])
    class_dist_df.to_csv(output_dir / 'class_distribution.csv', index=False)
    
    print(f"Report saved to: {output_dir}")
    return report


def validate_dataset(
    input_dir: Path,
    input_format: str,
    validation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate dataset integrity and format."""
    print("Validating dataset...")
    
    validator = DatasetValidator()
    
    if input_format == "imagefolder":
        results = validator.validate_imagefolder_dataset(
            input_dir,
            min_samples_per_class=validation_config.get('min_samples_per_class', 1),
            validate_images=validation_config.get('validate_images', False)
        )
    else:
        print(f"Validation for format '{input_format}' not yet implemented")
        results = {}
    
    return results


def main():
    """Main preprocessing function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(
        name="dinov3_preprocessing",
        level=log_level,
        log_dir=Path(args.output_dir) / "logs"
    )
    
    try:
        # Set random seed
        set_seed(args.seed)
        
        # Validate paths
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover dataset structure
        dataset_structure = discover_dataset_structure(input_dir, args.input_format)
        
        # Run processing based on mode
        if args.mode == "analyze":
            # Quality analysis
            quality_analysis = analyze_dataset_quality(
                dataset_structure,
                args.num_workers,
                args.limit_samples
            )
            
            # Generate report
            report = generate_dataset_report(
                dataset_structure,
                quality_analysis,
                output_dir,
                args.sample_visualizations
            )
            
            logger.info("Dataset analysis complete")
            
        elif args.mode == "validate":
            # Validation
            validation_config = {
                'min_samples_per_class': args.min_samples_per_class,
                'validate_images': args.validate_images
            }
            
            validation_results = validate_dataset(
                input_dir,
                args.input_format,
                validation_config
            )
            
            # Save validation results
            with open(output_dir / 'validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info("Dataset validation complete")
            
        elif args.mode == "preprocess":
            # Preprocessing configuration
            processing_config = {
                'target_size': tuple(args.image_size),
                'resize_method': args.resize_method,
                'quality_threshold': args.quality_threshold,
                'convert_grayscale': args.convert_grayscale
            }
            
            # Preprocess dataset
            preprocessing_results = preprocess_dataset(
                dataset_structure,
                output_dir,
                processing_config,
                args.num_workers,
                args.dry_run,
                args.limit_samples
            )
            
            # Save processing results
            with open(output_dir / 'preprocessing_results.json', 'w') as f:
                json.dump(preprocessing_results, f, indent=2, default=str)
            
            logger.info("Dataset preprocessing complete")
            
        elif args.mode == "split":
            # Split dataset
            split_results = split_dataset(
                dataset_structure,
                output_dir,
                args.train_ratio,
                args.val_ratio,
                args.test_ratio,
                args.stratify,
                args.seed
            )
            
            # Save split results
            with open(output_dir / 'split_results.json', 'w') as f:
                json.dump(split_results, f, indent=2, default=str)
            
            logger.info("Dataset splitting complete")
        
        # Generate final report if requested
        if args.generate_report:
            quality_analysis = analyze_dataset_quality(
                dataset_structure,
                args.num_workers,
                args.limit_samples
            ) if args.mode != "analyze" else {}
            
            generate_dataset_report(
                dataset_structure,
                quality_analysis,
                output_dir,
                args.sample_visualizations
            )
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
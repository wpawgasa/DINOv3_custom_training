# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DINOv3 custom fine-tuning project designed for implementing comprehensive computer vision pipelines using Meta's DINOv3 foundation models. The project supports multiple training paradigms including linear probing, full fine-tuning, and self-supervised continuation learning across various domains like medical imaging, satellite imagery, and industrial applications. Refer to https://github.com/facebookresearch/dinov3 for original implementation.

## Development Environment

The project uses a containerized development environment with GPU support:
- Base: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- Python 3.10+ with CUDA 12.1 support
- Development container configured for NVIDIA GPU access

## Project Architecture

Based on the comprehensive project specification (`dinov3-project-spec.md`), the intended project structure follows this layout:

```
dinov3_custom_training/
├── configs/
│   ├── models/           # Model configurations (dinov3_vitb16.yaml, etc.)
│   ├── datasets/         # Dataset configurations (medical_xray.yaml, etc.)
│   ├── training/         # Training hyperparameters (linear_probe.yaml, full_fine_tune.yaml)
│   └── evaluation/       # Evaluation settings
├── src/
│   ├── data/            # Data loading and preprocessing (dataset.py, preprocessing.py)
│   ├── models/          # Model implementations (dinov3_classifier.py)
│   ├── training/        # Training loops and utilities (trainer.py)
│   ├── evaluation/      # Evaluation metrics and tools
│   └── utils/           # Utility functions
├── scripts/
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Evaluation script
│   ├── preprocess.py    # Data preprocessing
│   └── deploy.py        # Deployment utilities
├── notebooks/           # Jupyter notebooks for exploration
├── tests/               # Unit and integration tests
├── docker/              # Containerization files
└── docs/                # Documentation
```

## Training Paradigms

The framework supports three main approaches:

1. **Linear Probing** - Freeze DINOv3 backbone, train lightweight classification head
   - Fast training, preserves universal features
   - Configuration: `configs/training/linear_probe.yaml`

2. **Full Fine-tuning** - Update all parameters with differential learning rates
   - Maximum performance on specialized tasks
   - Configuration: `configs/training/full_fine_tune.yaml`

3. **Self-Supervised Continuation** - Continue DINO pretraining on domain-specific data
   - Learn domain-specific features from unlabeled data
   - Configuration: `configs/training/ssl_continue.yaml`

## Model Variants

Supported DINOv3 model variants:
- **ViT-S/16** (21M params) - Edge deployment, 2.1GB memory
- **ViT-B/16** (86M params) - Balanced performance, 3.4GB memory  
- **ViT-L/16** (300M params) - High accuracy, 6.2GB memory
- **ViT-H+/16** (840M params) - Maximum performance, 12.8GB memory
- **ConvNeXt-T** (29M params) - CNN alternative, 2.8GB memory
- **ConvNeXt-B** (89M params) - Production ready, 4.1GB memory

## Common Commands

### Development Environment
```bash
# Start development container (GPU-enabled)
docker build -t dinov3-dev .devcontainer/
docker run --runtime=nvidia --gpus all -it dinov3-dev

# Install dependencies using uv (as configured in Dockerfile)
uv pip install -r requirements.txt
```

### Training Commands
```bash
# Linear probing training
python scripts/train.py --config configs/training/linear_probe.yaml

# Full fine-tuning  
python scripts/train.py --config configs/training/full_fine_tune.yaml

# Self-supervised continuation
python scripts/train.py --config configs/training/ssl_continue.yaml
```

### Data Preprocessing
```bash
# Preprocess custom dataset
python scripts/preprocess.py --data_dir /path/to/dataset --output_dir ./processed_data

# Validate dataset quality
python scripts/preprocess.py --validate --data_dir ./processed_data
```

### Evaluation and Testing
```bash
# Evaluate trained model
python scripts/evaluate.py --model_path ./outputs/best_model.pth --test_dir ./processed_data/test

# Run benchmark evaluation
python scripts/evaluate.py --benchmark --dataset imagenet --model_path ./outputs/best_model.pth

# Unit tests
python -m pytest tests/ -v

# Performance tests
python -m pytest tests/test_performance.py -v
```

### Deployment
```bash
# Export model for deployment
python scripts/deploy.py --model_path ./outputs/best_model.pth --export_format onnx

# Build deployment container
docker build -f docker/Dockerfile.deploy -t dinov3-classifier:latest .

# Run API server
python src/api/server.py --port 8000
```

## Configuration Management

The project uses YAML-based configuration files with hierarchical structure:

- **Model configs** (`configs/models/`): Define architecture, DINOv3 variant, task type
- **Training configs** (`configs/training/`): Specify optimizer, scheduler, hyperparameters  
- **Dataset configs** (`configs/datasets/`): Configure data paths, preprocessing, validation

Key configuration patterns:
- Use `freeze_backbone: true` for linear probing
- Set differential learning rates for full fine-tuning (`backbone_lr` vs `head_lr`)
- Configure domain-specific augmentations in dataset configs

## Domain-Specific Considerations

### Medical Imaging
- Convert grayscale to RGB by channel duplication
- Use conservative augmentations to preserve medical information
- Enable class weighting for handling imbalanced datasets
- Metrics: accuracy, precision, recall, f1, auc

### Satellite Imagery  
- Support high-resolution inputs (512x512+)
- Multi-scale training with resolution scheduling
- Satellite-appropriate augmentations (scale, color jitter, noise)

### Industrial Inspection
- Focus on defect detection and quality assessment
- Handle class imbalance with weighted sampling
- Emphasis on precision and recall metrics

## Performance Optimization

### Memory Management
- Use gradient checkpointing for large models
- Enable mixed precision training (`mixed_precision: true`)
- Configure gradient accumulation for effective larger batch sizes

### Speed Optimization
- Use `uv` package manager for faster dependency installation
- Enable CUDA optimizations in development container
- Consider model quantization for deployment (`quantize_model()`)

## Monitoring and Logging

The framework integrates with:
- **Weights & Biases (wandb)** - Experiment tracking and visualization
- **MLflow** - Model versioning and deployment tracking
- **Prometheus** - Production monitoring metrics

## Key Dependencies

Core dependencies (to be installed via requirements.txt):
- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face transformers for DINOv3 models
- `albumentations` - Advanced image augmentations
- `wandb` - Experiment tracking
- `fastapi` - API deployment framework
- `pytest` - Testing framework

## Testing Strategy

The project follows a comprehensive testing approach:
- **Unit tests** (`tests/test_*.py`) - Test individual components
- **Integration tests** - Test end-to-end pipelines  
- **Performance tests** - Validate latency and memory requirements
- **Deployment tests** - Test API endpoints and containerization

Run all tests before committing changes: `python -m pytest tests/ -v`

## Implementation Plan

This section outlines a comprehensive implementation plan for the DINOv3 custom fine-tuning project, organized into logical phases with dependencies and deliverables.

### Phase 1: Foundation & Infrastructure (Week 1)

**Objective**: Establish project foundation, dependencies, and basic configuration system.

**Tasks**:
1. **Project Structure Setup** (Day 1)
   - Create complete directory structure as specified
   - Initialize `__init__.py` files for all Python packages
   - Setup `.gitignore` and basic repository configuration

2. **Dependency Management** (Day 1-2)
   - Create `requirements.txt` with all core dependencies
   - Setup `pyproject.toml` for package configuration
   - Validate container environment and GPU access

3. **Configuration System** (Day 2-3)
   - Implement `src/utils/config.py` - YAML configuration loader
   - Create base configuration schemas for validation
   - Implement configuration merging and override functionality

4. **Basic Utilities** (Day 3-5)
   - `src/utils/logging.py` - Structured logging setup
   - `src/utils/device.py` - GPU/CPU device management
   - `src/utils/reproducibility.py` - Random seed management

**Deliverables**:
- Complete project structure
- Working configuration system
- Basic utility functions
- Validated development environment

**Dependencies**: None

### Phase 2: Data Pipeline Implementation (Week 2)

**Objective**: Implement robust data loading, preprocessing, and augmentation pipeline.

**Tasks**:
1. **Core Dataset Classes** (Day 1-2)
   - `src/data/dataset.py` - `CustomDINOv3Dataset` class
   - Support for ImageFolder, COCO, and custom annotation formats
   - Efficient data loading with caching

2. **Image Preprocessing Pipeline** (Day 2-3)
   - `src/data/preprocessing.py` - `ImagePreprocessor` class
   - Resize with aspect ratio preservation
   - Quality validation and filtering
   - Grayscale to RGB conversion for medical imaging

3. **Augmentation Framework** (Day 3-4)
   - Domain-specific augmentation policies
   - `src/data/augmentations.py` - Albumentations integration
   - Training vs. validation transform pipelines

4. **Data Validation** (Day 4-5)
   - Dataset quality checks and statistics
   - Class balance analysis
   - Data integrity validation

**Deliverables**:
- Complete data loading pipeline
- Preprocessing and augmentation systems
- Data validation utilities
- Sample dataset configurations

**Dependencies**: Phase 1 (Configuration system)

### Phase 3: Model Framework (Week 3)

**Objective**: Implement DINOv3 model wrapper with task-specific heads and multiple training paradigms.

**Tasks**:
1. **Model Base Classes** (Day 1-2)
   - `src/models/dinov3_classifier.py` - Main model implementation
   - Support for all DINOv3 variants (ViT-S/16, ViT-B/16, etc.)
   - Flexible task head architecture

2. **Task-Specific Heads** (Day 2-3)
   - Classification head with dropout and regularization
   - Segmentation head for dense prediction tasks
   - Multi-task head support

3. **Training Paradigm Support** (Day 3-4)
   - Linear probing (frozen backbone)
   - Full fine-tuning with differential learning rates
   - Self-supervised continuation framework

4. **Model Configuration** (Day 4-5)
   - Model factory pattern for easy instantiation
   - Configuration validation and loading
   - Pretrained weight loading and initialization

**Deliverables**:
- Complete model implementation
- Support for all training paradigms
- Model configuration system
- Model factory utilities

**Dependencies**: Phase 1 (Configuration), Phase 2 (Data pipeline for testing)

### Phase 4: Training System (Week 4-5)

**Objective**: Implement comprehensive training framework with optimization, scheduling, and monitoring.

**Tasks**:
1. **Core Training Loop** (Day 1-2)
   - `src/training/trainer.py` - `DINOv3Trainer` class
   - Training and validation loops
   - Loss computation and backpropagation

2. **Optimization Framework** (Day 2-3)
   - Multi-optimizer support (AdamW, SGD, etc.)
   - Learning rate schedulers (Cosine, Step, Warmup)
   - Differential learning rates for fine-tuning

3. **Advanced Training Features** (Day 3-4)
   - Mixed precision training
   - Gradient accumulation and clipping
   - Early stopping and best model tracking

4. **Checkpointing & Resume** (Day 4-5)
   - Model state saving and loading
   - Training state resumption
   - Automatic backup and recovery

5. **Logging Integration** (Day 5-7)
   - Weights & Biases integration
   - MLflow experiment tracking
   - Real-time metrics visualization

**Deliverables**:
- Complete training framework
- Optimization and scheduling system
- Checkpointing mechanism
- Integrated logging and monitoring

**Dependencies**: Phase 3 (Model framework), Phase 2 (Data pipeline)

### Phase 5: Evaluation Framework (Week 6)

**Objective**: Implement comprehensive evaluation, benchmarking, and robustness testing.

**Tasks**:
1. **Core Evaluation Classes** (Day 1-2)
   - `src/evaluation/evaluators.py` - Classification and segmentation evaluators
   - Comprehensive metrics calculation
   - Per-class and overall performance analysis

2. **Benchmark Protocols** (Day 2-3)
   - Standard benchmark datasets integration
   - Domain-specific benchmark support
   - Automated benchmark runner

3. **Robustness Testing** (Day 3-4)
   - Adversarial robustness evaluation
   - Common corruptions testing
   - Distribution shift analysis

4. **Visualization Tools** (Day 4-5)
   - Confusion matrices and classification reports
   - ROC curves and precision-recall plots
   - Feature visualization and attention maps

**Deliverables**:
- Complete evaluation framework
- Benchmark protocols
- Robustness testing suite
- Visualization utilities

**Dependencies**: Phase 4 (Training system for model loading)

### Phase 6: Command-Line Interface (Week 7)

**Objective**: Create user-friendly CLI scripts and main entry points.

**Tasks**:
1. **Training Script** (Day 1-2)
   - `scripts/train.py` - Main training entry point
   - Argument parsing and configuration loading
   - Multi-GPU and distributed training support

2. **Evaluation Script** (Day 2-3)
   - `scripts/evaluate.py` - Evaluation and benchmarking
   - Model loading and inference
   - Results export and visualization

3. **Data Processing Scripts** (Day 3-4)
   - `scripts/preprocess.py` - Data preprocessing and validation
   - Dataset conversion and formatting utilities
   - Quality analysis and reporting

4. **Utility Scripts** (Day 4-5)
   - Model conversion and export utilities
   - Configuration generation helpers
   - Development and debugging tools

**Deliverables**:
- Complete CLI interface
- User-friendly scripts
- Documentation and help systems
- Example usage patterns

**Dependencies**: Phase 4 (Training), Phase 5 (Evaluation), Phase 2 (Data processing)

### Phase 7: Deployment System (Week 8-9)

**Objective**: Implement production deployment infrastructure with optimization and monitoring.

**Tasks**:
1. **Model Optimization** (Day 1-2)
   - `src/deployment/optimization.py` - Quantization and pruning
   - Knowledge distillation framework
   - ONNX export and optimization

2. **API Server** (Day 2-4)
   - `src/api/server.py` - FastAPI implementation
   - Batch inference support
   - Error handling and validation
   - API documentation with Swagger

3. **Containerization** (Day 4-5)
   - `docker/Dockerfile.deploy` - Production container
   - Multi-stage builds for optimization
   - Security hardening

4. **Kubernetes Deployment** (Day 5-6)
   - K8s manifests and Helm charts
   - Auto-scaling configuration
   - Health checks and monitoring

5. **Monitoring & Observability** (Day 6-7)
   - Prometheus metrics integration
   - Performance monitoring
   - Alert configuration

**Deliverables**:
- Production-ready API server
- Container images
- Kubernetes deployment configs
- Monitoring and alerting setup

**Dependencies**: Phase 4 (Trained models), Phase 5 (Evaluation for validation)

### Phase 8: Testing & Documentation (Week 10)

**Objective**: Comprehensive testing suite and user documentation.

**Tasks**:
1. **Unit Testing** (Day 1-2)
   - Test all core components
   - Mock external dependencies
   - Achieve >90% code coverage

2. **Integration Testing** (Day 2-3)
   - End-to-end pipeline testing
   - Multi-GPU training tests
   - API endpoint testing

3. **Performance Testing** (Day 3-4)
   - Memory usage profiling
   - Inference speed benchmarks
   - Scalability testing

4. **Documentation** (Day 4-5)
   - API documentation
   - User guides and tutorials
   - Deployment guides
   - Troubleshooting documentation

**Deliverables**:
- Comprehensive test suite
- Performance benchmarks
- Complete documentation
- User tutorials and examples

**Dependencies**: All previous phases

### Implementation Guidelines

**Development Principles**:
- **Incremental Development**: Each phase builds on previous phases
- **Test-Driven Development**: Write tests alongside implementation
- **Configuration-First**: All behavior controlled via YAML configs
- **Modular Design**: Loosely coupled, highly cohesive components
- **Performance-Conscious**: Profile and optimize critical paths

**Quality Gates**:
- All unit tests must pass before phase completion
- Code coverage must exceed 85% for core components
- Performance benchmarks must meet specified requirements
- All configurations must be validated and documented

**Risk Mitigation**:
- **Technical Risks**: Extensive testing, gradual rollout, fallback mechanisms
- **Timeline Risks**: Parallel development where possible, critical path focus
- **Quality Risks**: Code reviews, automated testing, performance monitoring

**Dependencies Management**:
- Pin all dependency versions in requirements.txt
- Regular security updates and compatibility testing
- Container-based development for consistency

This implementation plan provides a structured approach to building the comprehensive DINOv3 fine-tuning framework while maintaining quality and enabling incremental progress validation.
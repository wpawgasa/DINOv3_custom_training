# Quick Start Guide

Get up and running with the DINOv3 Custom Fine-tuning Framework in just a few minutes.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 50GB free disk space

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install uv (fast Python package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-org/dinov3-custom-training.git
cd dinov3-custom-training

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-org/dinov3-custom-training.git
cd dinov3-custom-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Clone the repository
git clone https://github.com/your-org/dinov3-custom-training.git
cd dinov3-custom-training

# Build development container
docker build -f .devcontainer/Dockerfile -t dinov3-dev .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace dinov3-dev
```

## Verify Installation

```bash
# Test the installation
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"

# Run basic tests
python -m pytest tests/test_models.py::TestModelFactory::test_get_model_factory -v
```

## 🚀 5-Minute Example

Let's train a DINOv3 model on a sample dataset:

### Step 1: Prepare Sample Data

```bash
# Create sample dataset structure
mkdir -p data/sample_dataset/{cats,dogs,birds}

# Add some sample images to each category
# (Replace with your actual images)
cp /path/to/cat/images/* data/sample_dataset/cats/
cp /path/to/dog/images/* data/sample_dataset/dogs/
cp /path/to/bird/images/* data/sample_dataset/birds/
```

### Step 2: Generate Configuration

```bash
# Generate a configuration for your dataset
python scripts/generate_config.py \
  --mode interactive \
  --config-name animal_classification \
  --output-dir configs/
```

Answer the interactive prompts:
- Task type: `1` (Classification)
- Domain: `1` (Natural)
- Number of classes: `3`
- Training mode: `1` (Linear Probe)
- GPU memory: `8` (GB)
- Performance priority: `2` (Balanced)

### Step 3: Preprocess Data

```bash
# Preprocess and validate your dataset
python scripts/preprocess.py \
  --input-dir data/sample_dataset \
  --output-dir data/processed_dataset \
  --mode preprocess \
  --image-size 224 224 \
  --validate-images
```

### Step 4: Train the Model

```bash
# Start training
python scripts/train.py \
  --config configs/animal_classification.yaml \
  --output-dir outputs/animal_experiment \
  --experiment-name "my_first_experiment"
```

### Step 5: Evaluate the Model

```bash
# Evaluate the trained model
python scripts/evaluate.py \
  --model-path outputs/animal_experiment/best_model.pth \
  --data-path data/processed_dataset \
  --visualize \
  --export-format json csv
```

### Step 6: Serve the Model

```bash
# Start the API server
MODEL_PATH=outputs/animal_experiment/best_model.pth \
CONFIG_PATH=configs/animal_classification.yaml \
python src/api/server.py
```

Test the API:
```bash
# Test single prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/test_image.jpg"

# View API documentation
open http://localhost:8000/docs
```

## 🎯 Domain-Specific Quick Starts

### Medical Imaging

```bash
# Generate medical imaging configuration
python scripts/generate_config.py \
  --template medical_linear_probe \
  --num-classes 5 \
  --output-dir configs/

# Train with medical-specific augmentations
python scripts/train.py \
  --config configs/medical_linear_probe.yaml \
  --training-config configs/training/medical_conservative.yaml \
  --data-path /path/to/medical/dataset
```

### Satellite Imagery

```bash
# Generate satellite imaging configuration
python scripts/generate_config.py \
  --template satellite_fine_tune \
  --num-classes 10 \
  --image-size 256 256 \
  --output-dir configs/

# Train with satellite-specific settings
python scripts/train.py \
  --config configs/satellite_fine_tune.yaml \
  --data-path /path/to/satellite/dataset
```

### Industrial Inspection

```bash
# Generate industrial configuration
python scripts/generate_config.py \
  --template industrial_ssl \
  --num-classes 2 \
  --output-dir configs/

# Train for defect detection
python scripts/train.py \
  --config configs/industrial_ssl.yaml \
  --data-path /path/to/inspection/dataset
```

## 🔧 Common Configuration Tweaks

### Adjust for Limited GPU Memory

```yaml
# configs/low_memory.yaml
data:
  batch_size: 8          # Reduce from default 32
  
training:
  mixed_precision: true  # Enable to save memory
  gradient_accumulation_steps: 4  # Simulate larger batch

model:
  variant: "dinov3_vits16"  # Use smaller model
```

### Speed up Training

```yaml
# configs/fast_training.yaml
training:
  epochs: 20             # Reduce epochs
  mixed_precision: true  # Faster training
  
data:
  batch_size: 64         # Larger batches (if memory allows)
  num_workers: 8         # More data loading workers
  pin_memory: true       # Faster GPU transfer
```

### Maximum Accuracy

```yaml
# configs/high_accuracy.yaml
model:
  variant: "dinov3_vitl16"  # Larger model

training:
  mode: "full_fine_tune"    # Fine-tune everything
  epochs: 100
  learning_rate: 2e-5       # Lower learning rate
  
data:
  image_size: [384, 384]    # Higher resolution
```

## 📊 Monitoring Your Training

### Enable Weights & Biases

```bash
# Install wandb
pip install wandb

# Login to wandb
wandb login

# Add to your config
python scripts/train.py \
  --config configs/your_config.yaml \
  --use-wandb \
  --wandb-project "dinov3-experiments" \
  --wandb-entity "your-username"
```

### Enable TensorBoard

```bash
# Training automatically logs to tensorboard
python scripts/train.py --config configs/your_config.yaml

# View in another terminal
tensorboard --logdir outputs/your_experiment/logs
```

### Real-time Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f outputs/your_experiment/logs/training.log

# Monitor system resources
htop
```

## 🚨 Troubleshooting

### Out of Memory Errors

```bash
# Reduce batch size
python scripts/train.py --config configs/your_config.yaml --override "data.batch_size=8"

# Use gradient accumulation
python scripts/train.py --config configs/your_config.yaml \
  --override "training.gradient_accumulation_steps=4" \
  --override "data.batch_size=8"

# Use smaller model
python scripts/train.py --config configs/your_config.yaml \
  --override "model.variant=dinov3_vits16"
```

### Slow Data Loading

```bash
# Increase number of workers
python scripts/train.py --config configs/your_config.yaml \
  --override "data.num_workers=8"

# Cache preprocessed images
python scripts/preprocess.py --cache-preprocessed
```

### Model Not Learning

```bash
# Check learning rate
python scripts/train.py --config configs/your_config.yaml \
  --override "training.learning_rate=1e-3"

# Disable backbone freezing
python scripts/train.py --config configs/your_config.yaml \
  --override "training.freeze_backbone=false"

# Enable debugging
python scripts/debug_tools.py --tool gradient_analysis \
  --config configs/your_config.yaml \
  --model-path outputs/your_model.pth
```

## 📁 Project Structure After Quick Start

```
dinov3-custom-training/
├── configs/
│   └── animal_classification.yaml    # Your generated config
├── data/
│   ├── sample_dataset/              # Original data
│   └── processed_dataset/           # Preprocessed data
├── outputs/
│   └── animal_experiment/           # Training outputs
│       ├── best_model.pth          # Best model checkpoint
│       ├── config.yaml             # Training configuration
│       ├── metrics.json            # Training metrics
│       └── logs/                   # Training logs
└── eval_results/
    ├── animal_classification_results.json  # Evaluation results
    └── visualizations/             # Generated plots
```

## 🎯 Next Steps

After completing the quick start:

1. **Read the [Training Guide](training.md)** for advanced training techniques
2. **Explore [Configuration Reference](configuration.md)** for all options
3. **Check out [Deployment Guide](deployment.md)** for production deployment
4. **Review [API Documentation](api.md)** for integration details
5. **Try [Example Notebooks](../notebooks/)** for interactive tutorials

## 💡 Tips for Success

1. **Start Small**: Begin with a subset of your data and a smaller model
2. **Monitor Closely**: Use wandb or tensorboard to track training progress
3. **Validate Early**: Run evaluation after a few epochs to catch issues
4. **Save Frequently**: Enable checkpointing to avoid losing progress
5. **Experiment**: Try different configurations to find what works best

## 🆘 Getting Help

- **Check [Troubleshooting](troubleshooting.md)** for common issues
- **Browse [GitHub Issues](https://github.com/your-org/dinov3-custom-training/issues)** for known problems
- **Join [Discussions](https://github.com/your-org/dinov3-custom-training/discussions)** for community support
- **Read the [FAQ](faq.md)** for frequently asked questions

---

**Congratulations! 🎉 You've successfully set up and run your first DINOv3 experiment!**
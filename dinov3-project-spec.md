# DINOv3 Project Specification for Custom Dataset Fine-tuning

## Executive Summary

This specification outlines a comprehensive project for implementing DINOv3 fine-tuning with custom datasets. The project addresses three primary approaches: linear probing, full fine-tuning, and self-supervised continued training. It provides technical requirements, implementation guidelines, evaluation protocols, and deployment considerations for production-ready computer vision systems.

## 1. Project Overview

### 1.1 Objectives
- Implement robust DINOv3 fine-tuning pipeline for custom datasets
- Support multiple training paradigms (linear probe, full fine-tuning, SSL continuation)
- Ensure reproducible and scalable training across different domains
- Provide comprehensive evaluation and deployment frameworks
- Enable efficient transfer learning for specialized visual tasks

### 1.2 Scope and Deliverables

**Core Components:**
- DINOv3 model adaptation framework
- Custom dataset preprocessing pipeline  
- Training orchestration system
- Evaluation and benchmarking suite
- Model deployment infrastructure
- Documentation and testing framework

**Target Domains:**
- Medical imaging (radiology, pathology, dermatology)
- Satellite and aerial imagery analysis
- Industrial quality inspection
- Scientific imaging (microscopy, astronomy)
- General computer vision applications

### 1.3 Success Criteria
- Achieve >90% of specialized model performance with frozen features
- Reduce training time by 70% compared to training from scratch
- Enable deployment across edge to cloud environments
- Maintain model robustness across diverse visual domains
- Support batch inference for production workloads

## 2. Technical Architecture

### 2.1 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Training Engine │    │ Evaluation Suite│
│                 │    │                  │    │                 │
│ • Preprocessing │────│ • DINOv3 Backbone│────│ • Benchmarking  │
│ • Augmentation  │    │ • Task Heads     │    │ • Visualization │
│ • Validation    │    │ • Loss Functions │    │ • Reporting     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌──────────────────┐             │
         └──────────────│ Configuration    │─────────────┘
                        │ Management       │
                        │ • Hyperparams    │
                        │ • Model Configs  │
                        │ • Experiment     │
                        │   Tracking       │
                        └──────────────────┘
```

### 2.2 Model Architecture Options

| Model Variant | Parameters | Use Case | Memory (GB) | Speed (img/s) |
|---------------|------------|----------|-------------|---------------|
| ViT-S/16      | 21M        | Edge deployment | 2.1 | 450 |
| ViT-B/16      | 86M        | Balanced performance | 3.4 | 280 |
| ViT-L/16      | 300M       | High accuracy | 6.2 | 120 |
| ViT-H+/16     | 840M       | Maximum performance | 12.8 | 45 |
| ConvNeXt-T    | 29M        | CNN alternative | 2.8 | 380 |
| ConvNeXt-B    | 89M        | Production ready | 4.1 | 220 |

### 2.3 Training Paradigms

#### 2.3.1 Linear Probing
**Approach:** Freeze DINOv3 backbone, train lightweight classification head
**Benefits:** Fast training, preserves universal features, excellent generalization
**Implementation:**
```python
class DINOv3LinearProbe(nn.Module):
    def __init__(self, backbone_name, num_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
```

#### 2.3.2 Full Fine-tuning  
**Approach:** Update all parameters with differential learning rates
**Benefits:** Maximum performance on specialized tasks, domain adaptation
**Considerations:** Requires careful learning rate scheduling, risk of overfitting

#### 2.3.3 Self-Supervised Continuation
**Approach:** Continue DINO pretraining on domain-specific unlabeled data
**Benefits:** Learn domain-specific features, leverage large unlabeled datasets
**Use Cases:** Medical imaging, satellite imagery, scientific applications

## 3. Implementation Framework

### 3.1 Project Structure

```
dinov3_custom_training/
├── configs/
│   ├── models/           # Model configurations
│   ├── datasets/         # Dataset configurations  
│   ├── training/         # Training hyperparameters
│   └── evaluation/       # Evaluation settings
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model implementations
│   ├── training/        # Training loops and utilities
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

### 3.2 Configuration Management

**Model Configuration (configs/models/dinov3_vitb16.yaml):**
```yaml
model:
  name: "facebook/dinov3-vitb16-pretrain-lvd1689m"
  architecture: "vit"
  patch_size: 16
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  
fine_tuning:
  approach: "linear_probe"  # linear_probe | full_fine_tune | ssl_continue
  freeze_backbone: true
  dropout: 0.1
  
task:
  type: "classification"  # classification | segmentation | detection
  num_classes: 10
```

**Training Configuration (configs/training/linear_probe.yaml):**
```yaml
training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 50
  warmup_epochs: 5
  
optimizer:
  name: "AdamW"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  
scheduler:
  name: "CosineAnnealingLR"
  T_max: 50
  eta_min: 1e-6
  
augmentation:
  horizontal_flip: 0.5
  rotation_degrees: 15
  color_jitter: 0.4
  normalization: "imagenet"
```

**Dataset Configuration (configs/datasets/medical_xray.yaml):**
```yaml
dataset:
  name: "medical_xray"
  root_dir: "/data/medical_xray"
  split_ratio: [0.7, 0.15, 0.15]  # train/val/test
  
preprocessing:
  image_size: [224, 224]
  channels: 3  # Convert grayscale to RGB
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  
validation:
  min_samples_per_class: 10
  max_class_imbalance: 10.0
  quality_checks: true
```

### 3.3 Core Implementation Components

#### 3.3.1 Data Pipeline (src/data/dataset.py)
```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from PIL import Image
import albumentations as A

class CustomDINOv3Dataset(Dataset):
    def __init__(self, data_dir, processor, split='train', config=None):
        self.data_dir = data_dir
        self.processor = processor
        self.split = split
        self.config = config
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Setup augmentations
        self.transform = self._create_transforms()
    
    def _load_samples(self):
        # Implementation for loading dataset samples
        # Support various formats: ImageFolder, COCO, custom annotations
        pass
    
    def _create_transforms(self):
        if self.split == 'train':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = np.array(image)
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Process for DINOv3
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'image_path': image_path
        }
    
    def __len__(self):
        return len(self.samples)
```

#### 3.3.2 Model Implementation (src/models/dinov3_classifier.py)
```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class DINOv3Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load DINOv3 backbone
        self.backbone = AutoModel.from_pretrained(config.model.name)
        
        # Configure fine-tuning approach
        if config.fine_tuning.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Task-specific head
        hidden_size = self.backbone.config.hidden_size
        
        if config.task.type == "classification":
            self.head = self._create_classification_head(hidden_size, config.task.num_classes)
        elif config.task.type == "segmentation":
            self.head = self._create_segmentation_head(hidden_size, config.task.num_classes)
        
    def _create_classification_head(self, hidden_size, num_classes):
        return nn.Sequential(
            nn.Dropout(self.config.fine_tuning.dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def _create_segmentation_head(self, hidden_size, num_classes):
        # Implementation for dense prediction head
        pass
    
    def forward(self, pixel_values, return_features=False):
        outputs = self.backbone(pixel_values=pixel_values)
        
        if self.config.task.type == "classification":
            # Use pooled output (CLS token)
            features = outputs.pooler_output
            logits = self.head(features)
        elif self.config.task.type == "segmentation":
            # Use patch features for dense prediction
            features = outputs.last_hidden_state
            logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits
```

#### 3.3.3 Training Engine (src/training/trainer.py)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

class DINOv3Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Initialize tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def _create_optimizer(self):
        if self.config.fine_tuning.approach == "full_fine_tune":
            # Differential learning rates
            backbone_params = self.model.backbone.parameters()
            head_params = self.model.head.parameters()
            
            param_groups = [
                {'params': backbone_params, 'lr': self.config.training.backbone_lr},
                {'params': head_params, 'lr': self.config.training.head_lr}
            ]
        else:
            # Linear probe - only train head
            param_groups = [{'params': self.model.head.parameters()}]
        
        return optim.AdamW(
            param_groups,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.optimizer.weight_decay
        )
    
    def _create_scheduler(self):
        if self.config.scheduler.name == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler.T_max,
                eta_min=self.config.scheduler.eta_min
            )
        elif self.config.scheduler.name == "StepLR":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler.step_size,
                gamma=self.config.scheduler.gamma
            )
    
    def _create_criterion(self):
        if self.config.task.type == "classification":
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        elif self.config.task.type == "segmentation":
            return nn.CrossEntropyLoss(ignore_index=255)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(pixel_values)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(pixel_values)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(self.val_loader), 100 * correct / total
    
    def train(self):
        print(f"Starting training for {self.config.training.epochs} epochs...")
        
        for epoch in range(self.config.training.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.training.epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_accuracies.append(val_acc)
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Best Val Acc: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, filename)
```

## 4. Dataset Requirements and Preprocessing

### 4.1 Dataset Structure

**Recommended Format:**
```
dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
├── val/
│   └── [same structure]
├── test/
│   └── [same structure]
└── annotations/
    ├── train_annotations.json
    ├── val_annotations.json
    └── test_annotations.json
```

### 4.2 Data Quality Requirements

| Metric | Minimum | Recommended | Notes |
|--------|---------|-------------|-------|
| Images per class | 50 | 500+ | For linear probing |
| Image resolution | 224x224 | 512x512+ | Higher is better for detail |
| Format | JPEG, PNG | Lossless PNG | Avoid compression artifacts |
| Class balance | 1:10 ratio | 1:3 ratio | Use weighted sampling if imbalanced |
| Total dataset size | 1,000 | 10,000+ | More data improves generalization |

### 4.3 Preprocessing Pipeline

#### 4.3.1 Image Preprocessing (src/data/preprocessing.py)
```python
import cv2
import numpy as np
from PIL import Image
import albumentations as A

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config
        self.target_size = config.preprocessing.image_size
        self.channels = config.preprocessing.channels
    
    def preprocess_image(self, image_path):
        """Standard preprocessing pipeline for DINOv3"""
        # Load image
        image = Image.open(image_path)
        
        # Convert grayscale to RGB if needed
        if image.mode != 'RGB':
            if self.channels == 3:
                image = image.convert('RGB')
        
        # Resize maintaining aspect ratio
        image = self.resize_with_padding(image, self.target_size)
        
        # Quality checks
        if not self.quality_check(image):
            raise ValueError(f"Image quality check failed: {image_path}")
        
        return image
    
    def resize_with_padding(self, image, target_size):
        """Resize image with padding to maintain aspect ratio"""
        old_size = image.size
        ratio = float(target_size[0]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new("RGB", target_size, (0, 0, 0))
        new_image.paste(image, ((target_size[0] - new_size[0]) // 2,
                               (target_size[1] - new_size[1]) // 2))
        
        return new_image
    
    def quality_check(self, image):
        """Basic quality checks for image"""
        img_array = np.array(image)
        
        # Check if image is too dark or too bright
        mean_brightness = np.mean(img_array)
        if mean_brightness < 10 or mean_brightness > 245:
            return False
        
        # Check for sufficient contrast
        if np.std(img_array) < 5:
            return False
        
        return True
```

#### 4.3.2 Data Augmentation Strategy

**Training Augmentations:**
```python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.5),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Domain-Specific Augmentations:**

*Medical Imaging:*
```python
medical_transforms = A.Compose([
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    # Preserve medical image characteristics
])
```

*Satellite Imagery:*
```python
satellite_transforms = A.Compose([
    A.RandomScale(scale_limit=0.1, p=0.5),
    A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
    A.RandomCrop(height=224, width=224, p=1.0),
    A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
])
```

## 5. Training Configuration and Hyperparameters

### 5.1 Training Paradigm Selection

#### 5.1.1 Linear Probing Configuration
**Use Case:** Fast deployment, limited compute, preserving universal features

```yaml
linear_probe:
  approach: "freeze_backbone"
  learning_rate: 1e-3
  batch_size: 64
  epochs: 50
  warmup_epochs: 5
  
  optimizer:
    name: "AdamW"
    weight_decay: 0.01
    
  scheduler:
    name: "CosineAnnealingLR"
    T_max: 50
    
  early_stopping:
    patience: 10
    monitor: "val_accuracy"
    min_delta: 0.001
```

#### 5.1.2 Full Fine-tuning Configuration  
**Use Case:** Maximum performance, domain adaptation, sufficient compute

```yaml
full_fine_tune:
  approach: "differential_lr"
  backbone_lr: 1e-5
  head_lr: 1e-3
  batch_size: 32
  epochs: 100
  warmup_epochs: 10
  
  optimizer:
    name: "AdamW"
    weight_decay: 0.05
    betas: [0.9, 0.999]
    
  scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 20
    T_mult: 2
    
  gradient_clipping:
    max_norm: 1.0
    
  mixed_precision: true
```

#### 5.1.3 Self-Supervised Continuation
**Use Case:** Domain-specific feature learning, large unlabeled datasets

```yaml
ssl_continuation:
  approach: "domain_adaptation"
  learning_rate: 1e-4
  batch_size: 256
  epochs: 200
  
  dino_loss:
    temperature: 0.1
    student_temp: 0.1
    teacher_temp: 0.04
    
  distillation:
    teacher_momentum: 0.996
    center_momentum: 0.9
    
  multicrop:
    global_crops_scale: [0.4, 1.0]
    local_crops_scale: [0.05, 0.4]
    local_crops_number: 8
```

### 5.2 Model-Specific Configurations

#### 5.2.1 ViT-S/16 (Edge Deployment)
```yaml
model_config:
  name: "facebook/dinov3-vits16-pretrain-lvd1689m"
  patch_size: 16
  hidden_size: 384
  num_layers: 12
  num_heads: 6
  
training_config:
  batch_size: 128  # Can handle larger batches
  gradient_accumulation: 1
  mixed_precision: true
  
optimization:
  learning_rate: 2e-3  # Can use higher LR for smaller model
  weight_decay: 0.01
```

#### 5.2.2 ViT-L/16 (High Performance)
```yaml  
model_config:
  name: "facebook/dinov3-vitl16-pretrain-lvd1689m"
  patch_size: 16
  hidden_size: 1024
  num_layers: 24
  num_heads: 16
  
training_config:
  batch_size: 16  # Reduced due to memory constraints
  gradient_accumulation: 4  # Effective batch size 64
  mixed_precision: true
  
optimization:
  learning_rate: 5e-4  # Lower LR for larger model
  weight_decay: 0.05
```

### 5.3 Domain-Specific Configurations

#### 5.3.1 Medical Imaging Configuration
```yaml
medical_config:
  preprocessing:
    # Convert grayscale to RGB by channel duplication
    grayscale_to_rgb: true
    normalization: "imagenet"  # Use ImageNet stats
    
  augmentation:
    # Conservative augmentations to preserve medical information
    horizontal_flip: 0.5
    rotation_degrees: 5  # Limited rotation
    brightness_contrast: 0.1  # Subtle adjustments
    
  training:
    class_weights: true  # Handle class imbalance
    label_smoothing: 0.05  # Reduced for medical accuracy
    
  evaluation:
    metrics: ["accuracy", "precision", "recall", "f1", "auc"]
    stratified_splits: true
```

#### 5.3.2 Satellite Imagery Configuration
```yaml
satellite_config:
  preprocessing:
    # Use satellite-specific normalization if available
    normalization: "satellite"  # or fallback to imagenet
    high_resolution: true
    
  augmentation:
    # Satellite-appropriate augmentations
    random_scale: 0.1
    color_jitter: 0.1
    gaussian_noise: 0.02
    
  training:
    multi_scale: true  # Train on multiple resolutions
    resolution_schedule:
      - epochs: [0, 50]
        size: [224, 224]
      - epochs: [50, 100]  
        size: [512, 512]
```

## 6. Evaluation Framework

### 6.1 Evaluation Metrics

#### 6.1.1 Classification Metrics
```python
class ClassificationEvaluator:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def evaluate(self, model, test_loader, device):
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        class_report = classification_report(y_true, y_pred, 
                                           target_names=self.class_names,
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'overall_metrics': metrics,
            'class_report': class_report,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
```

#### 6.1.2 Dense Prediction Metrics
```python
class SegmentationEvaluator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def evaluate(self, model, test_loader, device):
        model.eval()
        total_iou = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Segmentation"):
                pixel_values = batch['pixel_values'].to(device)
                masks = batch['masks'].to(device)
                
                outputs = model(pixel_values)
                predictions = torch.argmax(outputs, dim=1)
                
                # Calculate IoU per image
                iou = self.calculate_iou(predictions, masks)
                total_iou += iou
                total_samples += 1
        
        return {'mean_iou': total_iou / total_samples}
    
    def calculate_iou(self, pred, target):
        # Implementation of IoU calculation
        pass
```

### 6.2 Benchmark Protocols

#### 6.2.1 Standard Benchmarks
```yaml
benchmark_config:
  datasets:
    - name: "ImageNet-1K"
      split: "validation"
      metric: "top1_accuracy"
      
    - name: "CIFAR-10"
      split: "test" 
      metric: "accuracy"
      
    - name: "Places365"
      split: "validation"
      metric: "top5_accuracy"
      
  evaluation:
    batch_size: 64
    num_workers: 4
    pin_memory: true
```

#### 6.2.2 Domain-Specific Benchmarks
```yaml
medical_benchmarks:
  - dataset: "ChestX-ray14"
    task: "multi_label_classification"
    metric: "auc_roc"
    
  - dataset: "ISIC2019"
    task: "skin_lesion_classification"
    metric: "balanced_accuracy"
    
  - dataset: "Montgomery_County"
    task: "tuberculosis_detection"
    metric: "sensitivity_specificity"
```

### 6.3 Robustness Evaluation

#### 6.3.1 Adversarial Robustness
```python
class AdversarialEvaluator:
    def __init__(self, model, epsilon=0.01):
        self.model = model
        self.epsilon = epsilon
    
    def evaluate_pgd_robustness(self, test_loader, device):
        # PGD attack evaluation
        pass
    
    def evaluate_common_corruptions(self, test_loader, device):
        # Common corruptions (brightness, contrast, noise, etc.)
        pass
```

#### 6.3.2 Distribution Shift Evaluation  
```python
class DistributionShiftEvaluator:
    def evaluate_ood_detection(self, model, in_dist_loader, ood_loader):
        # Out-of-distribution detection
        pass
    
    def evaluate_domain_transfer(self, model, source_loader, target_loader):
        # Cross-domain performance evaluation
        pass
```

## 7. Deployment and Production

### 7.1 Model Optimization

#### 7.1.1 Model Quantization
```python
import torch.quantization as quantization

def quantize_model(model, calibration_loader):
    """Post-training quantization for deployment"""
    model.eval()
    
    # Prepare model for quantization
    model_fp32 = model
    model_fp32.qconfig = quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = quantization.prepare(model_fp32)
    
    # Calibrate with representative data
    with torch.no_grad():
        for batch in calibration_loader:
            pixel_values = batch['pixel_values']
            _ = model_fp32_prepared(pixel_values)
    
    # Convert to quantized model
    model_int8 = quantization.convert(model_fp32_prepared)
    
    return model_int8
```

#### 7.1.2 Knowledge Distillation
```python
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        # Hard loss (student predictions vs true labels)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Soft loss (student vs teacher predictions)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        )
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

### 7.2 Deployment Infrastructure

#### 7.2.1 Docker Configuration
```dockerfile
# Dockerfile for DINOv3 deployment
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start API server
CMD ["python", "src/api/server.py"]
```

#### 7.2.2 FastAPI Deployment
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import io
import numpy as np

app = FastAPI(title="DINOv3 Classification API")

# Global model instance
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    # Load model and processor
    model = torch.load('models/best_model.pth')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m')
    model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Process uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess and predict
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        return {
            "predicted_class": int(predicted_class.item()),
            "confidence": float(torch.max(probabilities).item()),
            "all_probabilities": probabilities.squeeze().tolist()
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

#### 7.2.3 Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dinov3-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dinov3-classifier
  template:
    metadata:
      labels:
        app: dinov3-classifier
    spec:
      containers:
      - name: dinov3-classifier
        image: dinov3-classifier:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: "/app/models/best_model.pth"
        - name: BATCH_SIZE
          value: "32"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: dinov3-classifier-service
spec:
  selector:
    app: dinov3-classifier
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 7.3 Monitoring and Logging

#### 7.3.1 Performance Monitoring
```python
import time
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('predictions_total', 'Total predictions')
PREDICTION_TIME = Histogram('prediction_duration_seconds', 'Prediction time')
GPU_MEMORY = Gauge('gpu_memory_usage_bytes', 'GPU memory usage')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

def monitor_prediction(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.inc()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            PREDICTION_TIME.observe(duration)
            
            # Update system metrics
            CPU_USAGE.set(psutil.cpu_percent())
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    GPU_MEMORY.set(gpus[0].memoryUsed * 1024 * 1024)
            except:
                pass
    
    return wrapper
```

#### 7.3.2 Model Performance Tracking
```python
import mlflow
import wandb
from datetime import datetime

class ModelMonitor:
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.predictions = []
        self.start_time = datetime.now()
    
    def log_prediction(self, input_hash, prediction, confidence, latency):
        self.predictions.append({
            'timestamp': datetime.now(),
            'input_hash': input_hash,
            'prediction': prediction,
            'confidence': confidence,
            'latency': latency
        })
        
        # Log to MLflow
        mlflow.log_metric('prediction_confidence', confidence)
        mlflow.log_metric('prediction_latency', latency)
        
        # Detect drift or anomalies
        self.detect_drift()
    
    def detect_drift(self):
        # Implement drift detection logic
        if len(self.predictions) > 100:
            recent_confidences = [p['confidence'] for p in self.predictions[-100:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            
            if avg_confidence < 0.7:  # Threshold for retraining alert
                self.alert_low_confidence()
    
    def alert_low_confidence(self):
        # Send alert for potential model degradation
        pass
```

## 8. Testing and Validation

### 8.1 Unit Testing Framework
```python
import unittest
import torch
from src.models.dinov3_classifier import DINOv3Classifier
from src.data.dataset import CustomDINOv3Dataset

class TestDINOv3Classifier(unittest.TestCase):
    def setUp(self):
        self.config = self.load_test_config()
        self.model = DINOv3Classifier(self.config)
    
    def test_model_initialization(self):
        """Test model initializes correctly"""
        self.assertIsInstance(self.model.backbone, AutoModel)
        self.assertEqual(self.model.head[-1].out_features, self.config.task.num_classes)
    
    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        batch_size = 2
        channels = 3
        height = 224
        width = 224
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        outputs = self.model(dummy_input)
        
        self.assertEqual(outputs.shape, (batch_size, self.config.task.num_classes))
    
    def test_frozen_backbone(self):
        """Test backbone is properly frozen for linear probing"""
        if self.config.fine_tuning.freeze_backbone:
            for param in self.model.backbone.parameters():
                self.assertFalse(param.requires_grad)
    
    def load_test_config(self):
        # Load test configuration
        pass

class TestDataProcessing(unittest.TestCase):
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        # Test with various image formats and sizes
        pass
    
    def test_data_augmentation(self):
        """Test data augmentation transforms"""
        pass

if __name__ == '__main__':
    unittest.main()
```

### 8.2 Integration Testing
```python
class IntegrationTests(unittest.TestCase):
    def test_end_to_end_training(self):
        """Test complete training pipeline"""
        # Create small test dataset
        # Run training for few epochs
        # Verify model improves
        pass
    
    def test_model_save_load(self):
        """Test model serialization"""
        # Save model
        # Load model
        # Verify outputs are identical
        pass
    
    def test_deployment_pipeline(self):
        """Test deployment readiness"""
        # Test API endpoints
        # Test batch inference
        # Test error handling
        pass
```

### 8.3 Performance Testing
```python
import time
import memory_profiler

class PerformanceTests(unittest.TestCase):
    def test_inference_speed(self):
        """Test inference latency requirements"""
        model = self.load_model()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Measure inference time
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        avg_time = (time.time() - start_time) / 100
        
        # Assert latency requirement (e.g., < 50ms)
        self.assertLess(avg_time, 0.05)
    
    @memory_profiler.profile
    def test_memory_usage(self):
        """Test memory consumption"""
        model = self.load_model()
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            _ = model(dummy_input)
            
            # Check GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                print(f"Batch size {batch_size}: {memory_used:.2f} MB")
```

## 9. Documentation and Maintenance

### 9.1 API Documentation
```python
"""
DINOv3 Custom Fine-tuning Framework

This module provides a comprehensive framework for fine-tuning DINOv3 models
on custom datasets using various approaches including linear probing,
full fine-tuning, and self-supervised continuation.

Example:
    Basic usage for linear probing:
    
    ```python
    from dinov3_custom import DINOv3Trainer, DINOv3Classifier
    
    # Load configuration
    config = load_config('configs/linear_probe.yaml')
    
    # Initialize model
    model = DINOv3Classifier(config)
    
    # Setup data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Train model
    trainer = DINOv3Trainer(model, train_loader, val_loader, config)
    trainer.train()
    ```

Classes:
    DINOv3Classifier: Main model class with task-specific heads
    DINOv3Trainer: Training orchestration and optimization
    CustomDINOv3Dataset: Dataset loader with preprocessing
    ModelEvaluator: Comprehensive evaluation framework
"""
```

### 9.2 User Guide
```markdown
# DINOv3 Custom Fine-tuning User Guide

## Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/your-org/dinov3-custom-training.git
cd dinov3-custom-training
conda env create -f environment.yml
conda activate dinov3-training
```

### 2. Data Preparation
```bash
python scripts/preprocess.py --data_dir /path/to/your/dataset --output_dir ./processed_data
```

### 3. Configure Training
Edit `configs/training/your_config.yaml` with your dataset and model settings.

### 4. Start Training
```bash
python scripts/train.py --config configs/training/your_config.yaml
```

### 5. Evaluate Model
```bash
python scripts/evaluate.py --model_path ./outputs/best_model.pth --test_dir ./processed_data/test
```

## Advanced Configuration

### Custom Dataset Integration
[Detailed instructions for dataset integration]

### Hyperparameter Tuning
[Guidelines for optimization]

### Production Deployment
[Deployment best practices]
```

### 9.3 Maintenance Schedule

| Task | Frequency | Responsibility |
|------|-----------|---------------|
| Dependency updates | Monthly | DevOps Team |
| Security patches | As needed | Security Team |
| Model retraining | Quarterly | ML Team |
| Performance monitoring | Continuous | MLOps Team |
| Documentation updates | Bi-weekly | Development Team |

## 10. Risk Assessment and Mitigation

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| Model overfitting | High | Medium | Cross-validation, early stopping, regularization |
| Data drift in production | High | Medium | Continuous monitoring, automated retraining triggers |
| CUDA/GPU compatibility issues | Medium | Low | Docker containerization, version pinning |
| Memory overflow with large models | Medium | Medium | Gradient checkpointing, model sharding |
| Poor generalization to new domains | High | Medium | Diverse training data, robust evaluation protocols |

### 10.2 Operational Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| Model serving downtime | High | Low | Kubernetes auto-scaling, health checks |
| Data privacy violations | Very High | Low | Data anonymization, secure data handling |
| Model bias and fairness issues | High | Medium | Bias detection tools, diverse datasets |
| Intellectual property concerns | Medium | Low | Proper licensing, legal review |

### 10.3 Contingency Plans

**Model Performance Degradation:**
1. Automated alert system for accuracy drops
2. Rollback to previous stable model version
3. Emergency retraining pipeline activation
4. Root cause analysis and remediation

**Infrastructure Failure:**
1. Multi-region deployment setup
2. Automated failover mechanisms
3. Regular backup and recovery testing
4. Incident response procedures

## 11. Budget and Resource Planning

### 11.1 Compute Resources

| Phase | Duration | GPU Hours | Estimated Cost |
|-------|----------|-----------|---------------|
| Data preparation | 1 week | 40 | $200 |
| Linear probing | 2 days | 16 | $80 |
| Full fine-tuning | 1 week | 168 | $840 |
| Evaluation | 1 day | 8 | $40 |
| **Total Development** | **3 weeks** | **232** | **$1,160** |

### 11.2 Production Costs (Monthly)

| Resource | Quantity | Unit Cost | Monthly Cost |
|----------|----------|-----------|--------------|
| GPU instances (V100) | 2 | $1,000 | $2,000 |
| Storage (1TB) | 1 | $50 | $50 |
| Load balancer | 1 | $25 | $25 |
| Monitoring tools | 1 | $100 | $100 |
| **Total Monthly** | | | **$2,175** |

### 11.3 Human Resources

| Role | Time Allocation | Duration | Cost |
|------|----------------|----------|------|
| ML Engineer | 100% | 4 weeks | $8,000 |
| Data Engineer | 50% | 2 weeks | $2,000 |
| DevOps Engineer | 25% | 4 weeks | $2,000 |
| **Total Human Cost** | | | **$12,000** |

## Conclusion

This comprehensive specification provides a robust framework for implementing DINOv3 fine-tuning on custom datasets. The modular design allows for flexible adaptation to various domains while maintaining best practices for reproducibility, scalability, and production readiness.

Key advantages of this approach:
- **Flexibility:** Supports multiple training paradigms and model variants
- **Scalability:** Designed for both research and production environments  
- **Robustness:** Comprehensive testing and monitoring frameworks
- **Maintainability:** Clear documentation and structured codebase
- **Cost-effectiveness:** Optimized resource utilization and deployment strategies

The framework has been designed to minimize technical debt while maximizing adaptability to emerging requirements and domain-specific needs. Regular updates and community contributions will ensure continued relevance and performance improvements.
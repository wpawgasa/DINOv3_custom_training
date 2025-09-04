"""
Integration tests for DINOv3 framework.
Tests end-to-end functionality across multiple components.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
from pathlib import Path
import json
import time

from models.model_factory import create_model
from data.dataset import create_dataset
from data.augmentations import create_transforms
from training.trainer import DINOv3Trainer
from evaluation.evaluators import create_evaluator
from utils.config import load_hierarchical_config
from utils.schemas import ExperimentConfig


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    @pytest.mark.slow
    def test_end_to_end_training(self, experiment_config, sample_dataset_dir, temp_dir):
        """Test complete training pipeline."""
        # Update config with actual paths
        experiment_config.data.train_data_path = str(sample_dataset_dir)
        experiment_config.data.val_data_path = str(sample_dataset_dir)
        experiment_config.training.epochs = 2  # Short training for test
        experiment_config.data.batch_size = 2   # Small batch for test
        
        # Create transforms
        transform_manager = create_transforms(
            domain=experiment_config.augmentation.domain,
            image_size=experiment_config.data.image_size
        )
        
        # Create datasets
        train_dataset = create_dataset(
            data_path=experiment_config.data.train_data_path,
            annotation_format="imagefolder",
            transform=transform_manager.get_train_transform(),
            cache_images=False
        )
        
        val_dataset = create_dataset(
            data_path=experiment_config.data.val_data_path,
            annotation_format="imagefolder",
            transform=transform_manager.get_val_transform(),
            cache_images=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=experiment_config.data.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing in tests
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=experiment_config.data.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        with pytest.mock.patch('transformers.AutoModel.from_pretrained') as mock_pretrained:
            mock_model = pytest.mock.Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model
            
            model = create_model(experiment_config.model)
            model.set_training_mode(experiment_config.training.mode)
        
        # Create trainer
        trainer = DINOv3Trainer(
            model=model,
            config=experiment_config.training,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            logger=None,
            experiment_tracker=None
        )
        
        # Train model
        output_dir = temp_dir / "training_output"
        output_dir.mkdir(exist_ok=True)
        
        results = trainer.train(output_dir)
        
        # Verify training results
        assert 'training_time' in results
        assert 'final_epoch' in results
        assert results['final_epoch'] == experiment_config.training.epochs
        assert results['training_time'] > 0
        
        # Verify checkpoint was saved
        checkpoints = list(output_dir.glob("*.pth"))
        assert len(checkpoints) > 0
        
        # Verify checkpoint can be loaded
        checkpoint_path = checkpoints[0]
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
    
    @pytest.mark.slow
    def test_linear_probe_training(self, experiment_config, sample_dataset_dir, temp_dir):
        """Test linear probe training specifically."""
        # Configure for linear probing
        experiment_config.training.mode = "linear_probe"
        experiment_config.training.freeze_backbone = True
        experiment_config.training.epochs = 2
        experiment_config.data.batch_size = 2
        experiment_config.data.train_data_path = str(sample_dataset_dir)
        experiment_config.data.val_data_path = str(sample_dataset_dir)
        
        # Create model
        with pytest.mock.patch('transformers.AutoModel.from_pretrained') as mock_pretrained:
            mock_backbone = pytest.mock.Mock()
            mock_backbone.config.hidden_size = 384
            mock_backbone.parameters.return_value = [nn.Parameter(torch.randn(10, 10))]
            mock_pretrained.return_value = mock_backbone
            
            model = create_model(experiment_config.model)
            model.set_training_mode("linear_probe")
            
            # Verify backbone is frozen
            backbone_params_frozen = all(
                not param.requires_grad 
                for param in model.backbone.parameters()
            )
            assert backbone_params_frozen
            
            # Verify classifier is trainable
            classifier_params_trainable = all(
                param.requires_grad 
                for param in model.classifier.parameters()
            )
            assert classifier_params_trainable
    
    @pytest.mark.slow 
    def test_resume_training(self, experiment_config, sample_dataset_dir, temp_dir):
        """Test training resumption from checkpoint."""
        experiment_config.data.train_data_path = str(sample_dataset_dir)
        experiment_config.data.val_data_path = str(sample_dataset_dir)
        experiment_config.training.epochs = 3
        experiment_config.data.batch_size = 2
        
        output_dir = temp_dir / "resume_training"
        output_dir.mkdir(exist_ok=True)
        
        # First training session (2 epochs)
        with pytest.mock.patch('transformers.AutoModel.from_pretrained') as mock_pretrained:
            mock_model = pytest.mock.Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model
            
            # Create initial setup
            transform_manager = create_transforms(
                domain=experiment_config.augmentation.domain,
                image_size=experiment_config.data.image_size
            )
            
            train_dataset = create_dataset(
                data_path=experiment_config.data.train_data_path,
                annotation_format="imagefolder",
                transform=transform_manager.get_train_transform(),
                cache_images=False
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=experiment_config.data.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            model = create_model(experiment_config.model)
            
            # Train for 2 epochs
            experiment_config.training.epochs = 2
            trainer = DINOv3Trainer(
                model=model,
                config=experiment_config.training,
                train_dataloader=train_loader,
                val_dataloader=train_loader,  # Use same for simplicity
                logger=None,
                experiment_tracker=None
            )
            
            trainer.train(output_dir)
            
            # Find saved checkpoint
            checkpoints = list(output_dir.glob("*.pth"))
            assert len(checkpoints) > 0
            checkpoint_path = checkpoints[0]
            
            # Create new trainer and resume
            model2 = create_model(experiment_config.model)
            experiment_config.training.epochs = 3  # Resume to epoch 3
            
            trainer2 = DINOv3Trainer(
                model=model2,
                config=experiment_config.training,
                train_dataloader=train_loader,
                val_dataloader=train_loader,
                logger=None,
                experiment_tracker=None
            )
            
            # Load checkpoint and resume
            trainer2.load_checkpoint(str(checkpoint_path))
            results = trainer2.train(output_dir)
            
            # Should complete 3 epochs total
            assert results['final_epoch'] == 3


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""
    
    def test_end_to_end_evaluation(self, experiment_config, sample_dataset_dir, sample_model):
        """Test complete evaluation pipeline."""
        # Create evaluation dataset
        transform_manager = create_transforms(
            domain=experiment_config.augmentation.domain,
            image_size=experiment_config.data.image_size
        )
        
        eval_dataset = create_dataset(
            data_path=str(sample_dataset_dir),
            annotation_format="imagefolder",
            transform=transform_manager.get_val_transform(),
            cache_images=False
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        
        # Create evaluator
        evaluator = create_evaluator(
            task_type=experiment_config.model.task_type,
            num_classes=experiment_config.model.num_classes,
            device=torch.device('cpu')
        )
        
        sample_model.eval()
        
        # Run evaluation
        with torch.no_grad():
            for batch in eval_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets, _ = batch
                
                outputs = sample_model(inputs)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                evaluator.update(logits, targets)
        
        # Compute metrics
        metrics = evaluator.compute()
        
        # Verify metrics
        assert 'accuracy' in metrics
        assert 'top5_accuracy' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0.0 <= metrics['accuracy'] <= 1.0
    
    def test_model_inference_pipeline(self, sample_model, sample_images):
        """Test model inference on batch of images."""
        sample_model.eval()
        
        # Convert images to tensors manually (simulate preprocessing)
        tensors = []
        for img in sample_images:
            img_array = torch.from_numpy(np.array(img)).float() / 255.0
            img_array = img_array.permute(2, 0, 1)  # HWC to CHW
            tensors.append(img_array)
        
        batch = torch.stack(tensors)
        
        # Run inference
        with torch.no_grad():
            outputs = sample_model(batch)
        
        # Verify outputs
        assert isinstance(outputs, dict)
        assert 'logits' in outputs
        assert 'features' in outputs
        
        logits = outputs['logits']
        features = outputs['features']
        
        assert logits.shape == (len(sample_images), sample_model.num_classes)
        assert features.shape[0] == len(sample_images)
        
        # Test prediction methods
        predictions, confidences = sample_model.predict_with_confidence(batch)
        assert predictions.shape == (len(sample_images),)
        assert confidences.shape == (len(sample_images),)


class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_config_loading_and_validation(self, sample_config_file):
        """Test configuration loading and validation."""
        # Load configuration
        config_dict = load_hierarchical_config(base_config=str(sample_config_file))
        
        # Validate configuration
        config = ExperimentConfig(**config_dict)
        
        assert config.name == "test_experiment"
        assert config.model.variant == "dinov3_vits16"
        assert config.training.mode == "linear_probe"
        assert config.data.batch_size == 4
    
    def test_config_override_system(self, temp_dir):
        """Test configuration override system."""
        # Create base config
        base_config = {
            "name": "base_config",
            "model": {
                "variant": "dinov3_vits16",
                "num_classes": 10
            },
            "training": {
                "epochs": 100,
                "learning_rate": 1e-4
            }
        }
        
        base_config_file = temp_dir / "base.yaml"
        with open(base_config_file, 'w') as f:
            import yaml
            yaml.dump(base_config, f)
        
        # Test override
        overrides = ["training.epochs=50", "training.learning_rate=2e-4", "name=overridden_config"]
        
        config_dict = load_hierarchical_config(
            base_config=str(base_config_file),
            overrides=overrides
        )
        
        assert config_dict["name"] == "overridden_config"
        assert config_dict["training"]["epochs"] == 50
        assert config_dict["training"]["learning_rate"] == 2e-4
        assert config_dict["model"]["num_classes"] == 10  # Unchanged


class TestDeploymentIntegration:
    """Integration tests for deployment functionality."""
    
    @pytest.mark.slow
    def test_model_optimization_pipeline(self, sample_model, temp_dir):
        """Test model optimization pipeline."""
        from deployment.optimization import ModelQuantizer, create_optimization_pipeline
        
        # Test quantization
        quantizer = ModelQuantizer(sample_model)
        
        try:
            quantized_model = quantizer.dynamic_quantization()
            
            # Test that quantized model still works
            test_input = torch.randn(1, 3, 224, 224)
            
            with torch.no_grad():
                original_output = sample_model(test_input)
                quantized_output = quantized_model(test_input)
            
            # Outputs should have same shape
            assert original_output['logits'].shape == quantized_output['logits'].shape
            
        except RuntimeError as e:
            if "quantization not available" in str(e).lower():
                pytest.skip("Quantization not available in this PyTorch version")
            else:
                raise
    
    def test_model_checkpoint_compatibility(self, sample_model, temp_dir):
        """Test model checkpoint saving and loading compatibility."""
        checkpoint_path = temp_dir / "test_checkpoint.pth"
        
        # Save checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': sample_model.state_dict(),
            'optimizer_state_dict': {},
            'best_metric': 0.85,
            'config': sample_model.config.dict()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Load and verify
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        assert loaded_checkpoint['epoch'] == 5
        assert loaded_checkpoint['best_metric'] == 0.85
        assert 'model_state_dict' in loaded_checkpoint
        assert 'config' in loaded_checkpoint
        
        # Test model loading
        new_model = create_model(sample_model.config)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        
        # Test identical outputs
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = sample_model(test_input)
            loaded_output = new_model(test_input)
        
        torch.testing.assert_close(
            original_output['logits'],
            loaded_output['logits'],
            rtol=1e-5,
            atol=1e-5
        )


class TestPerformanceIntegration:
    """Performance integration tests."""
    
    @pytest.mark.slow
    def test_training_performance(self, experiment_config, sample_dataset_dir):
        """Test training performance benchmarks."""
        experiment_config.data.train_data_path = str(sample_dataset_dir)
        experiment_config.data.val_data_path = str(sample_dataset_dir)
        experiment_config.training.epochs = 1
        experiment_config.data.batch_size = 4
        
        # Create minimal training setup
        transform_manager = create_transforms(
            domain=experiment_config.augmentation.domain,
            image_size=experiment_config.data.image_size
        )
        
        train_dataset = create_dataset(
            data_path=experiment_config.data.train_data_path,
            annotation_format="imagefolder",
            transform=transform_manager.get_train_transform(),
            cache_images=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=experiment_config.data.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        with pytest.mock.patch('transformers.AutoModel.from_pretrained') as mock_pretrained:
            mock_model = pytest.mock.Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model
            
            model = create_model(experiment_config.model)
            
            # Time training step
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx >= 5:  # Test just a few batches
                    break
                    
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets, _ = batch
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs['logits'], targets)
                loss.backward()
                optimizer.step()
            
            end_time = time.time()
            batch_time = (end_time - start_time) / 5
            
            # Should process batches reasonably quickly
            assert batch_time < 2.0  # Less than 2 seconds per batch
    
    @pytest.mark.slow
    def test_inference_performance(self, sample_model):
        """Test inference performance benchmarks."""
        sample_model.eval()
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 3, 224, 224)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = sample_model(test_input)
            
            # Time inference
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    outputs = sample_model(test_input)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            throughput = batch_size / avg_time
            
            # Should achieve reasonable throughput
            assert throughput > 1.0  # At least 1 sample per second
            assert avg_time < 5.0    # Less than 5 seconds per batch
    
    @pytest.mark.gpu
    def test_gpu_memory_usage(self, sample_model):
        """Test GPU memory usage patterns."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device('cuda')
        model = sample_model.to(device)
        
        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8]
        memory_usage = {}
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            test_input = torch.randn(batch_size, 3, 224, 224, device=device)
            
            with torch.no_grad():
                outputs = model(test_input)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memory_usage[batch_size] = peak_memory / 1024 / 1024  # MB
        
        # Memory usage should scale reasonably with batch size
        assert memory_usage[4] > memory_usage[1]
        assert memory_usage[8] > memory_usage[4]
        
        # Should not exceed reasonable limits
        assert memory_usage[8] < 8192  # Less than 8GB for batch size 8
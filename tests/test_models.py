"""
Unit tests for DINOv3 model implementations.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from models.dinov3_classifier import DINOv3Classifier
from models.model_factory import create_model, get_model_factory
from utils.schemas import ModelConfig


class TestModelFactory:
    """Test model factory functionality."""

    def test_get_model_factory(self):
        """Test model factory retrieval."""
        factory = get_model_factory()
        assert factory is not None
        assert hasattr(factory, "create_model")

    def test_create_model_with_config(self, model_config):
        """Test model creation with configuration."""
        model = create_model(model_config)

        assert isinstance(model, DINOv3Classifier)
        assert model.num_classes == model_config.num_classes
        assert model.variant == model_config.variant

    def test_create_model_all_variants(self):
        """Test model creation for all supported variants."""
        variants = [
            "dinov3_vits16",
            "dinov3_vitb16",
            "dinov3_vitl16",
            "dinov3_convnext_tiny",
        ]

        for variant in variants:
            config = ModelConfig(
                variant=variant,
                task_type="classification",
                num_classes=10,
                pretrained=False,  # Skip downloading for tests
            )

            with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
                # Mock the pretrained model
                mock_model = Mock()
                mock_model.config.hidden_size = 384 if "vits" in variant else 768
                mock_pretrained.return_value = mock_model

                model = create_model(config)
                assert isinstance(model, DINOv3Classifier)
                assert model.variant == variant


class TestDINOv3Classifier:
    """Test DINOv3Classifier functionality."""

    def test_model_initialization(self, model_config):
        """Test model initialization."""
        with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
            mock_model = Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model

            model = DINOv3Classifier(model_config)

            assert model.num_classes == model_config.num_classes
            assert model.variant == model_config.variant
            assert model.task_type == model_config.task_type
            assert hasattr(model, "backbone")
            assert hasattr(model, "classifier")

    def test_forward_pass(self, sample_model, sample_batch):
        """Test forward pass through model."""
        with torch.no_grad():
            outputs = sample_model(sample_batch)

        # Check output structure
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "features" in outputs

        # Check output shapes
        batch_size = sample_batch.shape[0]
        num_classes = sample_model.num_classes

        assert outputs["logits"].shape == (batch_size, num_classes)
        assert len(outputs["features"].shape) == 2  # (batch_size, feature_dim)

    def test_training_mode_linear_probe(self, model_config):
        """Test linear probe training mode."""
        model_config.task_type = "classification"

        with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
            mock_model = Mock()
            mock_model.config.hidden_size = 384
            mock_model.parameters.return_value = [nn.Parameter(torch.randn(10, 10))]
            mock_pretrained.return_value = mock_model

            model = DINOv3Classifier(model_config)
            model.set_training_mode("linear_probe")

            # Check that backbone parameters are frozen
            for param in model.backbone.parameters():
                assert param.requires_grad is False

            # Check that classifier parameters are trainable
            for param in model.classifier.parameters():
                assert param.requires_grad is True

    def test_training_mode_full_fine_tune(self, model_config):
        """Test full fine-tuning mode."""
        with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
            mock_model = Mock()
            mock_model.config.hidden_size = 384
            mock_model.parameters.return_value = [nn.Parameter(torch.randn(10, 10))]
            mock_pretrained.return_value = mock_model

            model = DINOv3Classifier(model_config)
            model.set_training_mode("full_fine_tune")

            # Check that all parameters are trainable
            for param in model.parameters():
                assert param.requires_grad is True

    def test_get_model_info(self, sample_model):
        """Test model information retrieval."""
        info = sample_model.get_model_info()

        assert isinstance(info, dict)
        assert "variant" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info

        assert info["variant"] == sample_model.variant
        assert info["total_parameters"] > 0
        assert info["trainable_parameters"] >= 0
        assert info["model_size_mb"] > 0

    def test_feature_extraction(self, sample_model, sample_batch):
        """Test feature extraction functionality."""
        with torch.no_grad():
            features = sample_model.extract_features(sample_batch)

        batch_size = sample_batch.shape[0]
        assert features.shape[0] == batch_size
        assert len(features.shape) == 2  # (batch_size, feature_dim)

    def test_prediction_with_confidence(self, sample_model, sample_batch):
        """Test prediction with confidence scores."""
        with torch.no_grad():
            predictions, confidences = sample_model.predict_with_confidence(
                sample_batch
            )

        batch_size = sample_batch.shape[0]
        assert predictions.shape == (batch_size,)
        assert confidences.shape == (batch_size,)

        # Check that predictions are valid class indices
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < sample_model.num_classes)

        # Check that confidences are probabilities
        assert torch.all(confidences >= 0.0)
        assert torch.all(confidences <= 1.0)

    @pytest.mark.gpu
    def test_model_on_gpu(self, model_config, device):
        """Test model functionality on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
            mock_model = Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model

            model = DINOv3Classifier(model_config)
            model = model.to(device)

            # Test forward pass on GPU
            batch = torch.randn(2, 3, 224, 224, device=device)
            with torch.no_grad():
                outputs = model(batch)

            assert outputs["logits"].device == device
            assert outputs["features"].device == device

    def test_model_state_dict_save_load(self, sample_model, temp_dir):
        """Test model state dict saving and loading."""
        # Save state dict
        state_dict_path = temp_dir / "model_state.pth"
        torch.save(sample_model.state_dict(), state_dict_path)

        # Create new model and load state dict
        new_model = create_model(sample_model.config)
        new_model.load_state_dict(torch.load(state_dict_path))

        # Compare model outputs
        sample_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            original_output = sample_model(sample_input)
            loaded_output = new_model(sample_input)

        # Outputs should be identical
        torch.testing.assert_close(
            original_output["logits"], loaded_output["logits"], rtol=1e-5, atol=1e-5
        )

    def test_model_parameter_count(self, sample_model):
        """Test parameter counting functionality."""
        info = sample_model.get_model_info()

        # Manual parameter count
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(
            p.numel() for p in sample_model.parameters() if p.requires_grad
        )

        assert info["total_parameters"] == total_params
        assert info["trainable_parameters"] == trainable_params

    def test_model_memory_usage(self, sample_model, sample_batch):
        """Test model memory usage calculation."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = sample_model.to(device)
            batch = sample_batch.to(device)

            # Clear cache and reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                outputs = model(batch)

            peak_memory = torch.cuda.max_memory_allocated()
            assert peak_memory > 0

    def test_model_with_different_input_sizes(self, sample_model):
        """Test model with different input sizes."""
        input_sizes = [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)]

        for batch_size, channels, height, width in input_sizes:
            input_tensor = torch.randn(batch_size, channels, height, width)

            with torch.no_grad():
                outputs = sample_model(input_tensor)

            assert outputs["logits"].shape[0] == batch_size
            assert outputs["features"].shape[0] == batch_size

    def test_model_evaluation_mode(self, sample_model):
        """Test model evaluation mode settings."""
        # Test training mode
        sample_model.train()
        assert sample_model.training is True

        # Test evaluation mode
        sample_model.eval()
        assert sample_model.training is False

        # Test that dropout behaves differently
        input_tensor = torch.randn(1, 3, 224, 224)

        # Multiple forward passes in eval mode should be identical
        sample_model.eval()
        with torch.no_grad():
            output1 = sample_model(input_tensor)
            output2 = sample_model(input_tensor)

        torch.testing.assert_close(output1["logits"], output2["logits"])


class TestModelIntegration:
    """Integration tests for model functionality."""

    def test_model_training_step(self, sample_model, sample_batch, sample_targets):
        """Test a complete training step."""
        sample_model.train()

        # Forward pass
        outputs = sample_model(sample_batch)
        logits = outputs["logits"]

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, sample_targets)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for param in sample_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.slow
    def test_model_overfitting_small_batch(
        self, sample_model, sample_batch, sample_targets
    ):
        """Test model can overfit to a small batch."""
        sample_model.train()
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None

        for epoch in range(10):
            optimizer.zero_grad()

            outputs = sample_model(sample_batch)
            loss = criterion(outputs["logits"], sample_targets)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should decrease
        assert loss.item() < initial_loss

    def test_model_gradient_flow(self, sample_model, sample_batch, sample_targets):
        """Test gradient flow through the model."""
        sample_model.train()

        # Forward and backward pass
        outputs = sample_model(sample_batch)
        loss = nn.CrossEntropyLoss()(outputs["logits"], sample_targets)
        loss.backward()

        # Check gradient flow
        has_gradients = []
        for name, param in sample_model.named_parameters():
            if param.requires_grad:
                has_gradients.append(
                    param.grad is not None and torch.any(param.grad != 0)
                )

        # At least some parameters should have non-zero gradients
        assert any(has_gradients), "No parameters received gradients"

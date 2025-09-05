"""
Unit tests for FastAPI server functionality.
"""

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Import after setting up path in conftest
from api.server import app, model_manager


class TestAPIEndpoints:
    """Test API endpoint functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_model_manager(self, sample_model, experiment_config):
        """Mock model manager for testing."""
        mock_manager = Mock()
        mock_manager.model = sample_model
        mock_manager.config = experiment_config
        mock_manager.device = torch.device("cpu")
        mock_manager.class_names = ["class_0", "class_1", "class_2"]
        mock_manager.model_info = {
            "variant": "dinov3_vits16",
            "total_parameters": 1000000,
            "trainable_parameters": 1000000,
            "model_size_mb": 10.0,
        }
        return mock_manager

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "memory_usage" in data
        assert "uptime_seconds" in data

        assert data["status"] == "healthy"
        assert isinstance(data["gpu_available"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))

    @patch("api.server.model_manager")
    def test_load_model_endpoint(
        self, mock_manager, client, mock_model_manager, sample_checkpoint
    ):
        """Test model loading endpoint."""
        mock_manager.load_model = AsyncMock()
        mock_manager.model_info = mock_model_manager.model_info
        mock_manager.load_time = 2.5

        response = client.post(
            "/load", params={"model_path": str(sample_checkpoint), "device": "cpu"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "model_info" in data
        assert "load_time" in data

    def test_predict_without_model(self, client):
        """Test prediction without loaded model."""
        # Create a simple test image
        image = Image.new("RGB", (224, 224), color="red")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        response = client.post(
            "/predict", files={"file": ("test.jpg", img_buffer, "image/jpeg")}
        )

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]

    @patch("api.server.model_manager")
    def test_predict_single_image(self, mock_manager, client, mock_model_manager):
        """Test single image prediction."""
        # Setup mock model manager
        mock_manager.model = mock_model_manager.model
        mock_manager.predict_single = Mock(
            return_value={
                "predictions": [
                    {"class_id": 0, "class_name": "class_0", "confidence": 0.85}
                ],
                "confidence_scores": [0.85],
                "processing_time": 0.1,
            }
        )

        # Create test image
        image = Image.new("RGB", (224, 224), color="red")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        response = client.post(
            "/predict", files={"file": ("test.jpg", img_buffer, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "predictions" in data
        assert "confidence_scores" in data
        assert "processing_time" in data

        assert len(data["predictions"]) == 1
        assert data["predictions"][0]["class_id"] == 0
        assert data["predictions"][0]["confidence"] == 0.85

    @patch("api.server.model_manager")
    def test_predict_batch_images(self, mock_manager, client, mock_model_manager):
        """Test batch image prediction."""
        # Setup mock model manager
        mock_manager.model = mock_model_manager.model
        mock_manager.predict_batch = Mock(
            return_value=[
                {
                    "predictions": [
                        {"class_id": 0, "class_name": "class_0", "confidence": 0.85}
                    ],
                    "confidence_scores": [0.85],
                    "processing_time": 0.05,
                },
                {
                    "predictions": [
                        {"class_id": 1, "class_name": "class_1", "confidence": 0.92}
                    ],
                    "confidence_scores": [0.92],
                    "processing_time": 0.05,
                },
            ]
        )

        # Create test images
        files = []
        for i in range(2):
            image = Image.new("RGB", (224, 224), color=["red", "blue"][i])
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            files.append(("files", (f"test_{i}.jpg", img_buffer, "image/jpeg")))

        response = client.post("/predict/batch", files=files)

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "results" in data
        assert "total_processing_time" in data
        assert "batch_size" in data

        assert len(data["results"]) == 2
        assert data["batch_size"] == 2

    @patch("api.server.model_manager")
    def test_model_info_endpoint(self, mock_manager, client, mock_model_manager):
        """Test model info endpoint."""
        mock_manager.model = mock_model_manager.model
        mock_manager.get_health_info = Mock(
            return_value={
                "model_loaded": True,
                "device": "cpu",
                "model_info": mock_model_manager.model_info,
                "config": mock_model_manager.config.dict(),
            }
        )

        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        assert data["model_loaded"] is True
        assert "model_info" in data
        assert "config" in data

    def test_predict_invalid_image(self, client):
        """Test prediction with invalid image file."""
        # Create invalid file
        invalid_data = b"this is not an image"

        response = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(invalid_data), "text/plain")},
        )

        # Should return 503 (model not loaded) or 400 (invalid image)
        assert response.status_code in [400, 503]

    def test_predict_large_image(self, client):
        """Test prediction with large image."""
        # Create large image
        large_image = Image.new("RGB", (2048, 2048), color="green")
        img_buffer = io.BytesIO()
        large_image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        # Should handle large images (or return appropriate error)
        response = client.post(
            "/predict", files={"file": ("large.jpg", img_buffer, "image/jpeg")}
        )

        # Should return 503 (model not loaded) - but shouldn't crash
        assert response.status_code in [400, 503]

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_api_documentation(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data

        # Check that our endpoints are documented
        assert "/health" in openapi_data["paths"]
        assert "/predict" in openapi_data["paths"]
        assert "/predict/batch" in openapi_data["paths"]

    def test_request_validation(self, client):
        """Test request validation."""
        # Test invalid request parameters
        response = client.post("/predict/batch")  # No files
        assert response.status_code == 422  # Validation error

        # Test with invalid confidence threshold
        image = Image.new("RGB", (224, 224), color="red")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        response = client.post(
            "/predict/batch",
            files=[("files", ("test.jpg", img_buffer, "image/jpeg"))],
            params={"confidence_threshold": 1.5},  # Invalid threshold > 1.0
        )

        assert response.status_code == 422


class TestModelManager:
    """Test ModelManager functionality."""

    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        from api.server import ModelManager

        manager = ModelManager()

        assert manager.model is None
        assert manager.config is None
        assert manager.device is None
        assert manager.class_names is None
        assert manager.model_info == {}

    @pytest.mark.asyncio
    async def test_load_model(self, sample_checkpoint, sample_config_file):
        """Test model loading functionality."""
        from api.server import ModelManager

        manager = ModelManager()

        with patch("transformers.AutoModel.from_pretrained") as mock_pretrained:
            mock_model = Mock()
            mock_model.config.hidden_size = 384
            mock_pretrained.return_value = mock_model

            await manager.load_model(
                str(sample_checkpoint), str(sample_config_file), "cpu"
            )

            assert manager.model is not None
            assert manager.config is not None
            assert manager.device == torch.device("cpu")
            assert manager.load_time is not None

    def test_preprocess_image(self, sample_image):
        """Test image preprocessing."""
        from api.server import ModelManager

        manager = ModelManager()
        manager.transform_manager = Mock()

        # Mock transform
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        manager.transform_manager.get_val_transform.return_value = mock_transform

        tensor = manager.preprocess_image(sample_image)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)  # Batch dimension added

    def test_preprocess_batch(self, sample_images):
        """Test batch image preprocessing."""
        from api.server import ModelManager

        manager = ModelManager()
        manager.transform_manager = Mock()

        # Mock transform
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        manager.transform_manager.get_val_transform.return_value = mock_transform

        tensor = manager.preprocess_batch(sample_images)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (len(sample_images), 3, 224, 224)

    def test_predict_single(self, sample_image):
        """Test single image prediction."""
        from api.server import ModelManager

        manager = ModelManager()
        manager.model = Mock()
        manager.device = torch.device("cpu")
        manager.class_names = ["class_0", "class_1", "class_2"]
        manager.transform_manager = Mock()

        # Mock preprocessing
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(3, 224, 224)
        manager.transform_manager.get_val_transform.return_value = mock_transform

        # Mock model output
        mock_logits = torch.tensor([[0.1, 0.8, 0.1]])  # class_1 has highest score
        manager.model.return_value = {"logits": mock_logits}

        result = manager.predict_single(sample_image)

        assert "predictions" in result
        assert "confidence_scores" in result
        assert "processing_time" in result

        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["class_id"] == 1  # Highest score class
        assert result["predictions"][0]["class_name"] == "class_1"


class TestAPIIntegration:
    """Integration tests for API functionality."""

    @pytest.mark.slow
    def test_api_startup_shutdown(self):
        """Test API startup and shutdown."""
        # This would typically test the lifespan events
        # For now, just verify the app can be created
        from api.server import app

        assert app is not None
        assert app.title == "DINOv3 Classification API"

    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404
        response = client.get("/nonexistent")
        assert response.status_code == 404

        # Test method not allowed
        response = client.delete("/health")
        assert response.status_code == 405

    def test_request_timeout_handling(self, client):
        """Test request timeout handling."""
        # This would test timeout behavior
        # For now, just ensure the endpoint exists
        response = client.get("/health")
        assert response.status_code == 200

    @patch("api.server.model_manager")
    def test_concurrent_requests(self, mock_manager, client, mock_model_manager):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading

        # Setup mock
        mock_manager.model = mock_model_manager.model
        mock_manager.predict_single = Mock(
            return_value={
                "predictions": [
                    {"class_id": 0, "class_name": "class_0", "confidence": 0.85}
                ],
                "confidence_scores": [0.85],
                "processing_time": 0.1,
            }
        )

        def make_request():
            image = Image.new("RGB", (224, 224), color="red")
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            response = client.post(
                "/predict", files={"file": ("test.jpg", img_buffer, "image/jpeg")}
            )
            return response.status_code

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        assert all(status_code == 200 for status_code in results)

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        # Should return metrics in Prometheus format
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

        # Should contain some basic metrics
        metrics_text = response.text
        assert "python_info" in metrics_text or "process_" in metrics_text


class TestAPIPerformance:
    """Performance tests for API."""

    @pytest.mark.slow
    @patch("api.server.model_manager")
    def test_prediction_latency(self, mock_manager, client, mock_model_manager):
        """Test prediction latency."""
        import time

        # Setup mock with realistic timing
        def mock_predict_single(image, **kwargs):
            time.sleep(0.01)  # Simulate 10ms processing time
            return {
                "predictions": [
                    {"class_id": 0, "class_name": "class_0", "confidence": 0.85}
                ],
                "confidence_scores": [0.85],
                "processing_time": 0.01,
            }

        mock_manager.model = mock_model_manager.model
        mock_manager.predict_single = mock_predict_single

        # Create test image
        image = Image.new("RGB", (224, 224), color="red")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)

        # Measure end-to-end latency
        start_time = time.time()

        response = client.post(
            "/predict", files={"file": ("test.jpg", img_buffer, "image/jpeg")}
        )

        end_time = time.time()
        latency = end_time - start_time

        assert response.status_code == 200
        assert latency < 1.0  # Should be less than 1 second

    @pytest.mark.slow
    @patch("api.server.model_manager")
    def test_throughput(self, mock_manager, client, mock_model_manager):
        """Test API throughput."""
        import time

        # Setup mock
        mock_manager.model = mock_model_manager.model
        mock_manager.predict_single = Mock(
            return_value={
                "predictions": [
                    {"class_id": 0, "class_name": "class_0", "confidence": 0.85}
                ],
                "confidence_scores": [0.85],
                "processing_time": 0.001,
            }
        )

        # Make multiple requests and measure throughput
        num_requests = 10
        start_time = time.time()

        for i in range(num_requests):
            image = Image.new("RGB", (224, 224), color="red")
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)

            response = client.post(
                "/predict", files={"file": (f"test_{i}.jpg", img_buffer, "image/jpeg")}
            )
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time

        assert throughput > 1.0  # Should handle at least 1 request per second

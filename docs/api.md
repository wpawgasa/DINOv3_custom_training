# API Documentation

Complete reference for the DINOv3 FastAPI server, including endpoints, request/response formats, and integration examples.

## Overview

The DINOv3 API server provides a production-ready REST API for image classification using fine-tuned DINOv3 models. It supports both single image and batch processing with comprehensive monitoring and error handling.

**Base URL**: `http://localhost:8000`

**Content-Type**: `application/json` for JSON payloads, `multipart/form-data` for file uploads

## Authentication

Currently, the API does not require authentication, but it can be easily extended with API key authentication:

```python
# Future authentication header
headers = {
    "Authorization": "Bearer your-api-key",
    "Content-Type": "application/json"
}
```

## Endpoints

### Health Check

Check the health status and availability of the API server.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true,
  "memory_usage": {
    "gpu_allocated_mb": 2048.5,
    "gpu_cached_mb": 2560.0,
    "cpu_percent": 45.2
  },
  "uptime_seconds": 3600.5
}
```

**Example**:
```bash
curl -X GET "http://localhost:8000/health"
```

---

### Load Model

Load a DINOv3 model from a checkpoint file.

**Endpoint**: `POST /load`

**Parameters**:
- `model_path` (string, required): Path to the model checkpoint file
- `config_path` (string, optional): Path to configuration file
- `device` (string, optional): Device to load model on (`auto`, `cpu`, `cuda`). Default: `auto`

**Response**:
```json
{
  "status": "success",
  "message": "Model loaded successfully",
  "model_info": {
    "variant": "dinov3_vitb16",
    "total_parameters": 86000000,
    "trainable_parameters": 86000000,
    "model_size_mb": 345.2
  },
  "load_time": 2.45
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/load" \
  -d "model_path=/path/to/model.pth" \
  -d "config_path=/path/to/config.yaml" \
  -d "device=cuda"
```

---

### Single Image Prediction

Classify a single image and return predictions with confidence scores.

**Endpoint**: `POST /predict`

**Request Methods**:

#### Method 1: File Upload
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "return_features=true" \
  -F "confidence_threshold=0.5"
```

#### Method 2: JSON with Base64 Image
```python
import requests
import base64

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_base64": image_b64,
        "return_features": False,
        "return_attention": False,
        "confidence_threshold": 0.7
    }
)
```

**Request Parameters**:
- `file` (file): Image file to classify (when using file upload)
- `image_base64` (string): Base64 encoded image (when using JSON)
- `image_url` (string): URL of image to classify (not yet implemented)
- `return_features` (boolean): Return feature embeddings. Default: `false`
- `return_attention` (boolean): Return attention maps. Default: `false`
- `confidence_threshold` (float): Minimum confidence threshold (0.0-1.0). Default: `0.0`

**Response**:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "predictions": [
    {
      "class_id": 2,
      "class_name": "dog",
      "confidence": 0.87
    }
  ],
  "confidence_scores": [0.87],
  "processing_time": 0.15,
  "features": [0.23, 0.45, -0.12, ...],  // Only if return_features=true
  "attention_maps": {  // Only if return_attention=true
    "shape": [1, 197, 197],
    "data": [[...]]
  }
}
```

**Status Codes**:
- `200`: Success
- `400`: Invalid request (bad image, invalid parameters)
- `503`: Model not loaded
- `500`: Internal server error

---

### Batch Image Prediction

Process multiple images in a single request for improved throughput.

**Endpoint**: `POST /predict/batch`

**Request**:
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "return_features=false" \
  -F "confidence_threshold=0.5" \
  -F "batch_size=32"
```

**Request Parameters**:
- `files` (array of files): Multiple image files to classify
- `return_features` (boolean): Return feature embeddings for each image. Default: `false`
- `return_attention` (boolean): Return attention maps for each image. Default: `false`
- `confidence_threshold` (float): Minimum confidence threshold (0.0-1.0). Default: `0.0`
- `batch_size` (integer): Processing batch size. Default: `32`

**Response**:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440001",
  "results": [
    {
      "request_id": "550e8400-e29b-41d4-a716-446655440001_0",
      "predictions": [{"class_id": 0, "class_name": "cat", "confidence": 0.92}],
      "confidence_scores": [0.92],
      "processing_time": 0.05,
      "features": null,
      "attention_maps": null
    },
    {
      "request_id": "550e8400-e29b-41d4-a716-446655440001_1",
      "predictions": [{"class_id": 1, "class_name": "dog", "confidence": 0.85}],
      "confidence_scores": [0.85],
      "processing_time": 0.05,
      "features": null,
      "attention_maps": null
    }
  ],
  "total_processing_time": 0.25,
  "batch_size": 2
}
```

---

### Model Information

Get detailed information about the currently loaded model.

**Endpoint**: `GET /model/info`

**Response**:
```json
{
  "model_loaded": true,
  "device": "cuda:0",
  "model_info": {
    "variant": "dinov3_vitb16",
    "total_parameters": 86000000,
    "trainable_parameters": 86000000,
    "model_size_mb": 345.2
  },
  "config": {
    "name": "experiment_name",
    "model": {
      "variant": "dinov3_vitb16",
      "task_type": "classification",
      "num_classes": 10
    },
    "training": {...},
    "data": {...}
  },
  "gpu_memory": {
    "allocated_mb": 2048.5,
    "cached_mb": 2560.0,
    "max_allocated_mb": 2800.0
  }
}
```

---

### Prometheus Metrics

Get Prometheus-formatted metrics for monitoring.

**Endpoint**: `GET /metrics`

**Response**: Plain text Prometheus metrics
```
# HELP dinov3_requests_total Total requests
# TYPE dinov3_requests_total counter
dinov3_requests_total{method="POST",endpoint="/predict",status="success"} 150.0

# HELP dinov3_request_duration_seconds Request duration
# TYPE dinov3_request_duration_seconds histogram
dinov3_request_duration_seconds_bucket{le="0.1"} 45.0
dinov3_request_duration_seconds_bucket{le="0.25"} 120.0
dinov3_request_duration_seconds_bucket{le="0.5"} 140.0
...
```

**Access Restriction**: This endpoint is typically restricted to monitoring systems in production.

---

## OpenAPI Documentation

The API provides interactive documentation via Swagger UI and ReDoc:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2023-12-01T12:00:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | `INVALID_IMAGE` | Invalid image format or corrupted file |
| 400 | `INVALID_PARAMETERS` | Invalid request parameters |
| 413 | `FILE_TOO_LARGE` | Uploaded file exceeds size limit |
| 422 | `VALIDATION_ERROR` | Request validation failed |
| 503 | `MODEL_NOT_LOADED` | No model loaded on the server |
| 503 | `SERVER_OVERLOADED` | Server is overloaded, try again later |
| 500 | `INTERNAL_ERROR` | Internal server error |

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Single Prediction**: 10 requests/second per IP
- **Batch Prediction**: 5 requests/second per IP
- **Model Loading**: 2 requests/minute per IP

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1638360000
```

## Client Libraries

### Python Client

```python
import requests
from pathlib import Path

class DINOv3Client:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self):
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def load_model(self, model_path, config_path=None, device="auto"):
        data = {"model_path": model_path, "device": device}
        if config_path:
            data["config_path"] = config_path

        response = self.session.post(f"{self.base_url}/load", data=data)
        response.raise_for_status()
        return response.json()

    def predict(self, image_path, confidence_threshold=0.0, return_features=False):
        files = {"file": open(image_path, "rb")}
        data = {
            "confidence_threshold": confidence_threshold,
            "return_features": return_features
        }

        response = self.session.post(
            f"{self.base_url}/predict",
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, image_paths, batch_size=32, confidence_threshold=0.0):
        files = [("files", open(path, "rb")) for path in image_paths]
        data = {
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold
        }

        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
        finally:
            # Close all file handles
            for _, file_handle in files:
                file_handle.close()

# Usage example
client = DINOv3Client("http://localhost:8000")

# Check health
health = client.health_check()
print(f"Server status: {health['status']}")

# Load model
client.load_model("/path/to/model.pth", "/path/to/config.yaml")

# Single prediction
result = client.predict("test_image.jpg", confidence_threshold=0.7)
print(f"Prediction: {result['predictions'][0]['class_name']}")

# Batch prediction
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = client.predict_batch(image_paths)
print(f"Processed {results['batch_size']} images")
```

### JavaScript Client

```javascript
class DINOv3Client {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) throw new Error('Health check failed');
        return response.json();
    }

    async loadModel(modelPath, configPath = null, device = 'auto') {
        const formData = new FormData();
        formData.append('model_path', modelPath);
        formData.append('device', device);
        if (configPath) formData.append('config_path', configPath);

        const response = await fetch(`${this.baseUrl}/load`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Model loading failed');
        return response.json();
    }

    async predict(imageFile, confidenceThreshold = 0.0) {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('confidence_threshold', confidenceThreshold);

        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Prediction failed');
        return response.json();
    }

    async predictBatch(imageFiles, batchSize = 32, confidenceThreshold = 0.0) {
        const formData = new FormData();
        imageFiles.forEach(file => formData.append('files', file));
        formData.append('batch_size', batchSize);
        formData.append('confidence_threshold', confidenceThreshold);

        const response = await fetch(`${this.baseUrl}/predict/batch`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Batch prediction failed');
        return response.json();
    }
}

// Usage example
const client = new DINOv3Client('http://localhost:8000');

// Load model
await client.loadModel('/path/to/model.pth');

// Predict single image
const fileInput = document.getElementById('imageInput');
const result = await client.predict(fileInput.files[0], 0.7);
console.log('Prediction:', result.predictions[0].class_name);
```

## Performance Optimization

### Batch Size Optimization

For batch predictions, optimal batch size depends on:
- Available GPU memory
- Model size
- Input image size

**Recommended batch sizes**:
- ViT-S/16: 32-64
- ViT-B/16: 16-32
- ViT-L/16: 8-16
- ViT-H+/16: 4-8

### Request Optimization

1. **Keep connections alive**: Use `requests.Session()` in Python
2. **Compress images**: Use JPEG with quality 85-95 for best size/quality tradeoff
3. **Batch requests**: Use batch endpoints for multiple images
4. **Async processing**: Use async/await for concurrent requests

### Monitoring Endpoints

Monitor API performance using these metrics:

```python
import requests

# Get metrics
response = requests.get("http://localhost:8000/metrics")
metrics = response.text

# Parse specific metrics
import re
request_count = re.search(r'dinov3_requests_total.*?(\d+\.?\d*)', metrics)
avg_response_time = re.search(r'dinov3_request_duration_seconds_sum.*?(\d+\.?\d*)', metrics)
```

## Security Considerations

### Input Validation

- Maximum file size: 10MB per image
- Supported formats: JPEG, PNG, BMP, TIFF, WebP
- Image dimensions: Up to 4096x4096 pixels

### Network Security

When deploying in production:

1. **Use HTTPS**: Enable TLS encryption
2. **API Gateway**: Use an API gateway for authentication
3. **Rate Limiting**: Configure appropriate rate limits
4. **CORS**: Configure CORS policy for web applications

### Example Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://dinov3-api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=api burst=10 nodelay;

        # File size limit
        client_max_body_size 100M;
    }
}
```

## Troubleshooting

### Common Issues

1. **"Model not loaded" error**:
   ```bash
   curl -X POST "http://localhost:8000/load" -d "model_path=/path/to/model.pth"
   ```

2. **Out of memory errors**: Reduce batch size or use smaller model variant

3. **Slow responses**: Check GPU utilization and consider model optimization

4. **Connection errors**: Verify server is running and accessible

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG python src/api/server.py
```

### Health Monitoring

```python
# Automated health check
import requests
import time

def monitor_api_health(url, interval=30):
    while True:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            health = response.json()
            print(f"Status: {health['status']}, Uptime: {health['uptime_seconds']:.0f}s")
        except Exception as e:
            print(f"Health check failed: {e}")
        time.sleep(interval)

monitor_api_health("http://localhost:8000")
```

---

For more information, see the [Deployment Guide](deployment.md) and [Troubleshooting](troubleshooting.md) documentation.

"""
DINOv3 FastAPI Production Server

High-performance API server for DINOv3 model inference with batch processing,
monitoring, and production-ready features.
"""

import os
import sys
import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import logging
from contextlib import asynccontextmanager

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from models.model_factory import create_model
from data.augmentations import create_transforms
from utils.config import load_hierarchical_config
from utils.schemas import ExperimentConfig
from utils.logging import setup_logging


# Prometheus metrics
REQUEST_COUNT = Counter('dinov3_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('dinov3_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('dinov3_active_requests', 'Active requests')
MODEL_INFERENCE_DURATION = Histogram('dinov3_inference_duration_seconds', 'Model inference duration')
BATCH_SIZE_HISTOGRAM = Histogram('dinov3_batch_size', 'Batch sizes processed')


class PredictionRequest(BaseModel):
    """Single prediction request model."""
    image_url: Optional[str] = Field(None, description="URL of image to classify")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    return_features: bool = Field(False, description="Return feature embeddings")
    return_attention: bool = Field(False, description="Return attention maps")
    confidence_threshold: float = Field(0.0, description="Minimum confidence threshold")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    images: List[PredictionRequest] = Field(..., description="List of images to process")
    batch_size: Optional[int] = Field(32, description="Processing batch size")


class PredictionResponse(BaseModel):
    """Single prediction response model."""
    request_id: str
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float
    features: Optional[List[float]] = None
    attention_maps: Optional[Dict[str, Any]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    request_id: str
    results: List[PredictionResponse]
    total_processing_time: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime_seconds: float


class ModelManager:
    """Manages model loading, inference, and optimization."""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.transform_manager = None
        self.device = None
        self.class_names = None
        self.model_info = {}
        self.load_time = None
        
    async def load_model(
        self, 
        model_path: str, 
        config_path: Optional[str] = None,
        device: str = "auto"
    ):
        """Load model and configuration."""
        start_time = time.time()
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get configuration
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
        elif config_path:
            config_dict = load_hierarchical_config(base_config=config_path)
        else:
            raise ValueError("No configuration found")
        
        self.config = ExperimentConfig(**config_dict)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create model
        self.model = create_model(self.config.model)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel/DistributedDataParallel
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create transforms
        self.transform_manager = create_transforms(
            domain=self.config.augmentation.get('domain', 'natural'),
            image_size=self.config.data.image_size
        )
        
        # Get model info
        self.model_info = self.model.get_model_info()
        self.class_names = getattr(checkpoint, 'class_names', None)
        
        self.load_time = time.time() - start_time
        
        logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
        logger.info(f"Model: {self.model_info['variant']}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Parameters: {self.model_info['total_parameters']:,}")
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess single image for inference."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        transform = self.transform_manager.get_val_transform()
        tensor = transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess batch of images for inference."""
        tensors = []
        transform = self.transform_manager.get_val_transform()
        
        for image in images:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            tensor = transform(image)
            tensors.append(tensor)
        
        return torch.stack(tensors)
    
    @torch.no_grad()
    def predict_single(
        self, 
        image: Image.Image,
        return_features: bool = False,
        return_attention: bool = False,
        confidence_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Perform inference on single image."""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        outputs = self.model(input_tensor)
        
        # Extract results
        if isinstance(outputs, dict):
            logits = outputs['logits']
            features = outputs.get('features', None)
            attention = outputs.get('attention', None)
        else:
            logits = outputs
            features = None
            attention = None
        
        # Get predictions
        probabilities = F.softmax(logits, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        # Filter by confidence threshold
        valid_predictions = confidences >= confidence_threshold
        
        # Prepare results
        predictions = []
        confidence_scores = []
        
        for i in range(len(predicted)):
            if valid_predictions[i]:
                pred_class = predicted[i].item()
                confidence = confidences[i].item()
                
                prediction = {
                    'class_id': pred_class,
                    'class_name': self.class_names[pred_class] if self.class_names else f'class_{pred_class}',
                    'confidence': confidence
                }
                predictions.append(prediction)
                confidence_scores.append(confidence)
        
        result = {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'processing_time': time.time() - start_time
        }
        
        # Add features if requested
        if return_features and features is not None:
            result['features'] = features.cpu().numpy().tolist()
        
        # Add attention maps if requested
        if return_attention and attention is not None:
            result['attention_maps'] = {
                'shape': list(attention.shape),
                'data': attention.cpu().numpy().tolist()
            }
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self, 
        images: List[Image.Image],
        return_features: bool = False,
        return_attention: bool = False,
        confidence_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform batch inference."""
        start_time = time.time()
        
        if not images:
            return []
        
        # Preprocess batch
        input_tensor = self.preprocess_batch(images).to(self.device)
        
        # Record batch size
        BATCH_SIZE_HISTOGRAM.observe(len(images))
        
        # Inference
        with MODEL_INFERENCE_DURATION.time():
            outputs = self.model(input_tensor)
        
        # Extract results
        if isinstance(outputs, dict):
            logits = outputs['logits']
            features = outputs.get('features', None)
            attention = outputs.get('attention', None)
        else:
            logits = outputs
            features = None
            attention = None
        
        # Get predictions
        probabilities = F.softmax(logits, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
        
        # Process each result
        results = []
        batch_processing_time = time.time() - start_time
        
        for i in range(len(images)):
            # Filter by confidence threshold
            confidence = confidences[i].item()
            
            if confidence >= confidence_threshold:
                pred_class = predicted[i].item()
                
                predictions = [{
                    'class_id': pred_class,
                    'class_name': self.class_names[pred_class] if self.class_names else f'class_{pred_class}',
                    'confidence': confidence
                }]
                confidence_scores = [confidence]
            else:
                predictions = []
                confidence_scores = []
            
            result = {
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'processing_time': batch_processing_time / len(images)
            }
            
            # Add features if requested
            if return_features and features is not None:
                result['features'] = features[i].cpu().numpy().tolist()
            
            # Add attention maps if requested
            if return_attention and attention is not None:
                result['attention_maps'] = {
                    'shape': list(attention[i].shape),
                    'data': attention[i].cpu().numpy().tolist()
                }
            
            results.append(result)
        
        return results
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get model health information."""
        health_info = {
            'model_loaded': self.model is not None,
            'device': str(self.device) if self.device else None,
            'model_info': self.model_info,
            'config': self.config.dict() if self.config else None
        }
        
        # Memory info
        if self.device and self.device.type == 'cuda':
            health_info['gpu_memory'] = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'cached_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024
            }
        
        return health_info


# Global model manager
model_manager = ModelManager()
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting DINOv3 API server...")
    
    # Load model if specified in environment
    model_path = os.getenv('MODEL_PATH')
    config_path = os.getenv('CONFIG_PATH')
    device = os.getenv('DEVICE', 'auto')
    
    if model_path:
        try:
            await model_manager.load_model(model_path, config_path, device)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Continue without model for health checks
    
    yield
    
    # Shutdown
    logger.info("Shutting down DINOv3 API server...")


# Create FastAPI app
app = FastAPI(
    title="DINOv3 Classification API",
    description="Production API for DINOv3 image classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Setup Prometheus monitoring
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Setup logging
logger = setup_logging(
    name="dinov3_api",
    level=os.getenv('LOG_LEVEL', 'INFO')
)


def require_model():
    """Dependency to ensure model is loaded."""
    if model_manager.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Use /load endpoint to load a model first."
        )
    return model_manager


def parse_image(image_data: Union[UploadFile, str]) -> Image.Image:
    """Parse image from various input formats."""
    if isinstance(image_data, UploadFile):
        # File upload
        try:
            image = Image.open(image_data.file)
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    elif isinstance(image_data, str):
        # Base64 encoded image
        try:
            import base64
            import io
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid image format")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - start_time
    
    health_info = {
        'status': 'healthy',
        'model_loaded': model_manager.model is not None,
        'gpu_available': torch.cuda.is_available(),
        'memory_usage': {},
        'uptime_seconds': uptime
    }
    
    # Add memory info
    if torch.cuda.is_available():
        health_info['memory_usage']['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        health_info['memory_usage']['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    
    import psutil
    health_info['memory_usage']['cpu_percent'] = psutil.virtual_memory().percent
    
    return health_info


@app.post("/load")
async def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    device: str = "auto"
):
    """Load model endpoint."""
    try:
        await model_manager.load_model(model_path, config_path, device)
        return {
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_info': model_manager.model_info,
            'load_time': model_manager.load_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest = None,
    file: UploadFile = File(None),
    manager: ModelManager = Depends(require_model)
):
    """Single image prediction endpoint."""
    ACTIVE_REQUESTS.inc()
    request_id = str(uuid.uuid4())
    
    try:
        with REQUEST_DURATION.time():
            # Parse image
            if file:
                image = parse_image(file)
            elif request and request.image_base64:
                image = parse_image(request.image_base64)
            elif request and request.image_url:
                # TODO: Implement URL image loading
                raise HTTPException(status_code=501, detail="URL image loading not implemented")
            else:
                raise HTTPException(status_code=400, detail="No image provided")
            
            # Get prediction parameters
            return_features = request.return_features if request else False
            return_attention = request.return_attention if request else False
            confidence_threshold = request.confidence_threshold if request else 0.0
            
            # Perform prediction
            result = manager.predict_single(
                image,
                return_features=return_features,
                return_attention=return_attention,
                confidence_threshold=confidence_threshold
            )
            
            response = PredictionResponse(
                request_id=request_id,
                **result
            )
            
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="success").inc()
        return response
        
    except HTTPException:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    return_features: bool = False,
    return_attention: bool = False,
    confidence_threshold: float = 0.0,
    batch_size: int = 32,
    manager: ModelManager = Depends(require_model)
):
    """Batch image prediction endpoint."""
    ACTIVE_REQUESTS.inc()
    request_id = str(uuid.uuid4())
    
    try:
        start_time = time.time()
        
        with REQUEST_DURATION.time():
            # Parse images
            images = []
            for file in files:
                image = parse_image(file)
                images.append(image)
            
            # Process in batches
            all_results = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_results = manager.predict_batch(
                    batch_images,
                    return_features=return_features,
                    return_attention=return_attention,
                    confidence_threshold=confidence_threshold
                )
                
                # Convert to PredictionResponse objects
                for result in batch_results:
                    response = PredictionResponse(
                        request_id=f"{request_id}_{len(all_results)}",
                        **result
                    )
                    all_results.append(response)
            
            total_time = time.time() - start_time
            
            batch_response = BatchPredictionResponse(
                request_id=request_id,
                results=all_results,
                total_processing_time=total_time,
                batch_size=len(images)
            )
        
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="success").inc()
        return batch_response
        
    except HTTPException:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="error").inc()
        raise
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/model/info")
async def get_model_info(manager: ModelManager = Depends(require_model)):
    """Get model information."""
    return manager.get_health_info()


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    workers = int(os.getenv('WORKERS', '1'))
    log_level = os.getenv('LOG_LEVEL', 'info')
    
    # Run server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True
    )
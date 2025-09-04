import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m'   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class StructuredLogger:
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            self._setup_console_handler()
    
    def _setup_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        formatter = ColoredFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def setup_file_handler(self, log_file: Path, level: str = "INFO", use_json: bool = False):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        extra = {'extra_fields': kwargs} if kwargs else {}
        self.logger.critical(message, extra=extra)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        metric_data = {'metrics': metrics}
        if step is not None:
            metric_data['step'] = step
        
        self.info("Metrics logged", **metric_data)
    
    def log_config(self, config: Dict[str, Any]):
        self.info("Configuration", config=config)
    
    def log_model_info(self, model_name: str, num_params: int, model_size_mb: float):
        self.info(
            "Model information",
            model_name=model_name,
            num_parameters=num_params,
            model_size_mb=model_size_mb
        )
    
    def log_training_progress(self, epoch: int, step: int, loss: float, **metrics):
        self.info(
            f"Training progress - Epoch {epoch}, Step {step}",
            epoch=epoch,
            step=step,
            loss=loss,
            **metrics
        )
    
    def log_validation_results(self, epoch: int, **metrics):
        self.info(
            f"Validation results - Epoch {epoch}",
            epoch=epoch,
            **metrics
        )


def setup_logging(
    name: str = "dinov3_training",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    use_json_logs: bool = False
) -> StructuredLogger:
    logger = StructuredLogger(name, level)
    
    if log_dir:
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.setup_file_handler(log_file, file_level, use_json_logs)
    
    return logger


def get_logger(name: str = "dinov3_training") -> StructuredLogger:
    return StructuredLogger(name)


class LoggerContext:
    def __init__(self, logger: StructuredLogger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
        self.original_format = {}
        
        for handler in self.logger.logger.handlers:
            if hasattr(handler, 'formatter'):
                self.original_format[handler] = handler.formatter
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handler, formatter in self.original_format.items():
            handler.setFormatter(formatter)
    
    def info(self, message: str, **kwargs):
        combined_kwargs = {**self.context, **kwargs}
        self.logger.info(message, **combined_kwargs)
    
    def debug(self, message: str, **kwargs):
        combined_kwargs = {**self.context, **kwargs}
        self.logger.debug(message, **combined_kwargs)
    
    def warning(self, message: str, **kwargs):
        combined_kwargs = {**self.context, **kwargs}
        self.logger.warning(message, **combined_kwargs)
    
    def error(self, message: str, **kwargs):
        combined_kwargs = {**self.context, **kwargs}
        self.logger.error(message, **combined_kwargs)


def with_context(logger: StructuredLogger, **context) -> LoggerContext:
    return LoggerContext(logger, context)


_default_logger = None

def get_default_logger() -> StructuredLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger
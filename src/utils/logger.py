"""
Logging utilities for the project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger."""
    return logging.getLogger(name)


class TrainingLogger:
    """Logger specifically for training."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=experiment_name,
            log_file=log_file,
            level=logging.INFO
        )
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics."""
        msg = f"Epoch {epoch}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(msg)
    
    def log_params(self, params: dict):
        """Log hyperparameters."""
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
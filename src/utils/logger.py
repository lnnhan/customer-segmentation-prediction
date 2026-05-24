"""
Simple logging utility for the coffee project.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name="coffee_project", log_dir="logs", level=logging.INFO):
    """Setup logger with file and console output."""
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        Path(log_dir) / f"{name}_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name="coffee_project"):
    """Get existing logger or create new one."""
    logger = logging.getLogger(name)
    return logger if logger.handlers else setup_logger(name)

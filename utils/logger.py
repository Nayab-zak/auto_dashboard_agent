#!/usr/bin/env python3
# logger.py - Logging configuration for the application

import logging
import os
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'logs'
log_dir.mkdir(exist_ok=True)

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create the logger that will be imported by other modules
logger = logging.getLogger('dashboard_agent')
logger.setLevel(logging.INFO)
logger.info("Logger initialized successfully")

# Make sure the logger is exported
__all__ = ['logger']

# Print confirmation to ensure the module is loaded correctly
print("Logger module loaded successfully")

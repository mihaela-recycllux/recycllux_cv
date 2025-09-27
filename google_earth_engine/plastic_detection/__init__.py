"""
Plastic Detection System - Google Earth Engine

A comprehensive, object-oriented system for detecting floating plastic debris
using multi-sensor satellite data from Google Earth Engine.
"""

__version__ = "1.0.0"
__author__ = "Varun Burde"
__email__ = "varun@recycllux.com"

from .config.config import config
from .downloader.downloader import GEEDownloader
from .filters.filter_manager import FilterManager
from .utils.visualization import VisualizationUtils

__all__ = [
    'config',
    'GEEDownloader',
    'FilterManager',
    'VisualizationUtils'
]
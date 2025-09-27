#!/usr/bin/env python3
"""
Configuration settings for the plastic detection system.

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025
"""

import ee
from datetime import datetime, timedelta
import os

class Settings:
    """Configuration class for plastic detection project"""
    
    # Google Earth Engine Configuration
    EE_PROJECT_ID = 'recycllux-satellite-data'
    
    # Default Region of Interest coordinates (Mediterranean area known for plastic pollution)
    DEFAULT_ROI_COORDS = [14.0, 46.0, 14.2, 46.15]
    
    @classmethod
    def get_default_roi(cls):
        """Get default ROI as Earth Engine Geometry (requires EE to be initialized)"""
        return ee.Geometry.Rectangle(cls.DEFAULT_ROI_COORDS)
    
    # Time Configuration
    DEFAULT_START_DATE = '2023-01-01'
    DEFAULT_END_DATE = '2023-12-31'
    
    # Data Collection Parameters
    SENTINEL2_CLOUD_THRESHOLD = 20  # Maximum cloud coverage percentage
    SENTINEL1_INSTRUMENT_MODE = 'IW'  # Interferometric Wide Swath
    
    # Processing Parameters
    SENTINEL2_SCALE = 10  # meters per pixel
    SENTINEL1_SCALE = 20  # meters per pixel
    MODIS_SCALE = 500    # meters per pixel
    LANDSAT_SCALE = 30   # meters per pixel
    
    # Output Configuration
    OUTPUT_DIR = '/Users/varunburde/projects/Recyllux/google_earth_engine/plastic_detection/outputs'
    OUTPUT_FORMAT = 'GeoTIFF'
    OUTPUT_CRS = 'EPSG:4326'
    
    # Visualization Parameters
    RGB_VIS_PARAMS = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2']
    }
    
    NDVI_VIS_PARAMS = {
        'min': -1,
        'max': 1,
        'palette': ['red', 'yellow', 'green']
    }
    
    SAR_VIS_PARAMS = {
        'min': -25,
        'max': 0,
        'palette': ['000000', 'FFFFFF']
    }
    
    FALSE_COLOR_VIS_PARAMS = {
        'min': 0,
        'max': 3000,
        'bands': ['B8', 'B4', 'B3']
    }
    
    # Water Detection Parameters
    NDWI_VIS_PARAMS = {
        'min': -1,
        'max': 1,
        'palette': ['red', 'yellow', 'cyan', 'blue']
    }
    
    # Timeout Settings
    DOWNLOAD_TIMEOUT = 60
    PROCESSING_TIMEOUT = 120
    
    # Thumbnail Settings
    THUMBNAIL_DIMENSIONS = 512
    THUMBNAIL_FORMAT = 'png'
    
    # Plastic Detection Specific Parameters
    PLASTIC_INDICES = {
        'FDI': {  # Floating Debris Index
            'description': 'Floating Debris Index for plastic detection',
            'formula': '(NIR - Red) / (SWIR1 + Red)'
        },
        'NDWI': {  # Normalized Difference Water Index
            'description': 'Water detection index',
            'formula': '(Green - NIR) / (Green + NIR)'
        },
        'FAI': {  # Floating Algae Index
            'description': 'Floating Algae Index (can help distinguish from plastic)',
            'formula': 'NIR - (Red + (SWIR1 - Red) * (842 - 665) / (1610 - 665))'
        }
    }
    
    # Spectral Bands Configuration
    SENTINEL2_BANDS = {
        'B1': {'name': 'Coastal aerosol', 'wavelength': '443nm', 'resolution': 60},
        'B2': {'name': 'Blue', 'wavelength': '490nm', 'resolution': 10},
        'B3': {'name': 'Green', 'wavelength': '560nm', 'resolution': 10},
        'B4': {'name': 'Red', 'wavelength': '665nm', 'resolution': 10},
        'B5': {'name': 'Red Edge 1', 'wavelength': '705nm', 'resolution': 20},
        'B6': {'name': 'Red Edge 2', 'wavelength': '740nm', 'resolution': 20},
        'B7': {'name': 'Red Edge 3', 'wavelength': '783nm', 'resolution': 20},
        'B8': {'name': 'NIR', 'wavelength': '842nm', 'resolution': 10},
        'B8A': {'name': 'Red Edge 4', 'wavelength': '865nm', 'resolution': 20},
        'B9': {'name': 'Water vapor', 'wavelength': '945nm', 'resolution': 60},
        'B11': {'name': 'SWIR 1', 'wavelength': '1610nm', 'resolution': 20},
        'B12': {'name': 'SWIR 2', 'wavelength': '2190nm', 'resolution': 20}
    }
    
    SENTINEL1_BANDS = {
        'VV': {'name': 'Vertical transmit, Vertical receive', 'polarization': 'VV'},
        'VH': {'name': 'Vertical transmit, Horizontal receive', 'polarization': 'VH'}
    }
    
    @classmethod
    def get_output_path(cls, filename):
        """Get full output path for a given filename"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def get_timestamp_filename(cls, prefix, extension='tif'):
        """Generate a timestamped filename"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}.{extension}"
    
    @classmethod
    def get_date_range_last_n_days(cls, n_days=30):
        """Get date range for the last n days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

class PlasticDetectionConfig:
    """Specific configuration for plastic detection algorithms"""
    
    # Known plastic hotspots coordinates for validation
    PLASTIC_HOTSPOT_COORDS = {
        'Mediterranean': [14.0, 46.0, 14.2, 46.15],
        'Great_Pacific_Patch': [-155.0, 35.0, -135.0, 42.0],
        'Caribbean': [-85.0, 15.0, -75.0, 25.0],
        'Bay_of_Bengal': [85.0, 15.0, 95.0, 25.0]
    }
    
    @classmethod
    def get_plastic_hotspots(cls):
        """Get plastic hotspots as Earth Engine Geometries (requires EE to be initialized)"""
        return {
            name: ee.Geometry.Rectangle(coords) 
            for name, coords in cls.PLASTIC_HOTSPOT_COORDS.items()
        }
    
    # Spectral signatures for different materials (approximate values)
    SPECTRAL_SIGNATURES = {
        'plastic_bottles': {
            'blue': 0.15,
            'green': 0.25,
            'red': 0.35,
            'nir': 0.40,
            'swir1': 0.30,
            'swir2': 0.20
        },
        'plastic_bags': {
            'blue': 0.12,
            'green': 0.22,
            'red': 0.32,
            'nir': 0.45,
            'swir1': 0.35,
            'swir2': 0.25
        },
        'seawater': {
            'blue': 0.08,
            'green': 0.04,
            'red': 0.02,
            'nir': 0.01,
            'swir1': 0.005,
            'swir2': 0.002
        },
        'sea_foam': {
            'blue': 0.30,
            'green': 0.35,
            'red': 0.40,
            'nir': 0.35,
            'swir1': 0.25,
            'swir2': 0.15
        }
    }
    
    # Thresholds for plastic detection
    DETECTION_THRESHOLDS = {
        'FDI_MIN': 0.1,      # Minimum FDI for potential plastic
        'FDI_MAX': 0.8,      # Maximum FDI to exclude land
        'NDWI_MAX': 0.3,     # Maximum NDWI to ensure water areas
        'NIR_MIN': 0.05,     # Minimum NIR reflectance
        'SIZE_MIN': 10,      # Minimum size in pixels
        'SIZE_MAX': 1000     # Maximum size in pixels (to exclude large objects)
    }
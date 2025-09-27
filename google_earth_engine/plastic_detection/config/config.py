"""
Configuration settings for Plastic Detection System using Google Earth Engine

This module contains all configuration parameters for the plastic detection pipeline,
including area of interest, time periods, satellite parameters, and detection thresholds.
"""

import os
from typing import Tuple, Dict, Any
import datetime

class Config:
    """Configuration class for plastic detection parameters"""

    def __init__(self):
        # Google Earth Engine authentication
        self.service_account = os.environ.get('GEE_SERVICE_ACCOUNT')
        self.private_key_path = os.environ.get('GEE_PRIVATE_KEY_PATH')

        # Area of Interest - Romanian Black Sea Coast
        # Near Constanța port and Danube Delta
        self.aoi_bounds = {
            'min_lon': 28.5,
            'max_lon': 29.2,
            'min_lat': 44.0,
            'max_lat': 44.5
        }

        # Time period for analysis
        self.start_date = '2024-07-01'
        self.end_date = '2024-07-31'

        # Satellite data parameters
        self.satellite_params = {
            'sentinel2': {
                'collection': 'COPERNICUS/S2_SR_HARMONIZED',
                'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                'scale': 10,  # meters
                'cloud_filter': 20  # maximum cloud cover percentage
            },
            'sentinel1': {
                'collection': 'COPERNICUS/S1_GRD',
                'bands': ['VV', 'VH'],
                'scale': 10,  # meters
                'orbit': 'ASCENDING'  # or 'DESCENDING'
            }
        }

        # Image processing parameters
        self.processing_params = {
            'target_resolution': 10,  # meters
            'image_size': (768, 512),  # width, height in pixels
            'crs': 'EPSG:4326'  # WGS84
        }

        # Detection thresholds
        self.detection_thresholds = {
            'fdi': {
                'threshold': 0.01,
                'adaptive': True,
                'percentile': 95
            },
            'ndvi': {
                'water_threshold': 0.0,
                'vegetation_threshold': 0.3
            },
            'ndwi': {
                'water_threshold': 0.0,
                'adaptive': True
            },
            'plastic_index': {
                'threshold': 0.01,
                'adaptive': True
            },
            'anomaly_detection': {
                'contamination': 0.01,  # expected proportion of outliers
                'n_estimators': 100
            }
        }

        # Base output directory - CHANGE THIS TO YOUR DESIRED OUTPUT LOCATION
        self.base_output_dir = '/Users/varunburde/projects/Recyllux/google_earth_engine/plastic_detection/data'

        # Output directories (relative to base_output_dir)
        self.output_dirs = {
            'data': self.base_output_dir,
            'images': os.path.join(self.base_output_dir, 'images'),
            'vectors': os.path.join(self.base_output_dir, 'vectors'),
            'statistics': os.path.join(self.base_output_dir, 'statistics'),
            'logs': os.path.join(self.base_output_dir, 'logs')
        }

        # Visualization parameters
        self.viz_params = {
            'rgb_stretch': 3.5,
            'dpi': 300,
            'figure_size': (16, 12),
            'colormaps': {
                'fdi': 'RdBu_r',
                'ndvi': 'RdYlGn',
                'plastic': 'Reds',
                'water': 'Blues'
            }
        }

    def get_aoi_geometry(self) -> Dict[str, Any]:
        """Get AOI as Earth Engine geometry"""
        return {
            'type': 'Polygon',
            'coordinates': [[
                [self.aoi_bounds['min_lon'], self.aoi_bounds['min_lat']],
                [self.aoi_bounds['max_lon'], self.aoi_bounds['min_lat']],
                [self.aoi_bounds['max_lon'], self.aoi_bounds['max_lat']],
                [self.aoi_bounds['min_lon'], self.aoi_bounds['max_lat']],
                [self.aoi_bounds['min_lon'], self.aoi_bounds['min_lat']]
            ]]
        }

    def get_time_filter(self) -> Tuple[str, str]:
        """Get time period for filtering"""
        return (self.start_date, self.end_date)

    def get_output_path(self, filename: str, subdir: str = 'data') -> str:
        """Get full path for output file"""
        base_dir = self.output_dirs.get(subdir, self.output_dirs['data'])
        return os.path.join(base_dir, filename)

    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check AOI bounds
            assert self.aoi_bounds['min_lon'] < self.aoi_bounds['max_lon']
            assert self.aoi_bounds['min_lat'] < self.aoi_bounds['max_lat']

            # Check dates
            start = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
            assert start < end

            # Check thresholds
            assert 0 < self.detection_thresholds['fdi']['threshold'] < 1
            assert 0 < self.detection_thresholds['anomaly_detection']['contamination'] < 1

            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Global configuration instance
config = Config()
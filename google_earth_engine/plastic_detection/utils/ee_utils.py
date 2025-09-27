#!/usr/bin/env python3
"""
Earth Engine utilities for plastic detection project.

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025
"""

import ee
import signal
import time
from functools import wraps
from config.settings import Settings

def with_timeout(seconds=60):
    """Decorator to add timeout to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except Exception as e:
                signal.alarm(0)
                raise e
        return wrapper
    return decorator

def initialize_earth_engine():
    """Initialize Google Earth Engine with proper authentication"""
    try:
        ee.Initialize(project=Settings.EE_PROJECT_ID)
        print('✓ Earth Engine initialized successfully')
        return True
    except ee.EEException as e:
        print(f"Authentication required: {e}")
        try:
            ee.Authenticate()
            ee.Initialize(project=Settings.EE_PROJECT_ID)
            print('✓ Authenticated and initialized Earth Engine')
            return True
        except Exception as auth_error:
            print(f"✗ Failed to authenticate Earth Engine: {auth_error}")
            return False

def validate_date_format(date_string):
    """Validate date format (YYYY-MM-DD)"""
    try:
        time.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def create_roi_from_coordinates(min_lon, min_lat, max_lon, max_lat):
    """Create a region of interest from coordinates"""
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

def get_image_collection_info(collection):
    """Get information about an image collection"""
    try:
        size = collection.size().getInfo()
        if size > 0:
            first_image = ee.Image(collection.first())
            date_range = collection.aggregate_array('system:time_start')
            dates = date_range.getInfo()
            
            if dates:
                start_date = ee.Date(min(dates)).format('YYYY-MM-dd').getInfo()
                end_date = ee.Date(max(dates)).format('YYYY-MM-dd').getInfo()
            else:
                start_date = end_date = "Unknown"
                
            return {
                'size': size,
                'date_range': f"{start_date} to {end_date}",
                'bands': first_image.bandNames().getInfo()
            }
        else:
            return {'size': 0, 'date_range': 'No data', 'bands': []}
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return {'size': 0, 'date_range': 'Error', 'bands': []}

def mask_clouds_sentinel2(image):
    """Mask clouds in Sentinel-2 image using QA60 band"""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)

def scale_sentinel2(image):
    """Apply scaling factor to Sentinel-2 data"""
    return image.multiply(0.0001).copyProperties(image, ['system:time_start'])

def calculate_spectral_indices(image):
    """Calculate various spectral indices for an image"""
    # Normalize band names for Sentinel-2
    blue = image.select('B2')
    green = image.select('B3')  
    red = image.select('B4')
    nir = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # NDWI - Normalized Difference Water Index
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    
    # FDI - Floating Debris Index (for plastic detection)
    fdi = nir.subtract(red).divide(swir1.add(red)).rename('FDI')
    
    # FAI - Floating Algae Index
    # FAI = NIR - (Red + (SWIR1 - Red) * (842 - 665) / (1610 - 665))
    fai = nir.subtract(red.add(swir1.subtract(red).multiply(0.108))).rename('FAI')
    
    # MNDWI - Modified Normalized Difference Water Index
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI')
    
    # Add all indices to the image
    return image.addBands([ndvi, ndwi, fdi, fai, mndwi])

def apply_water_mask(image, threshold=0.0):
    """Apply water mask using NDWI"""
    ndwi = image.select('NDWI')
    water_mask = ndwi.gt(threshold)
    return image.updateMask(water_mask)

def filter_by_size(image, min_pixels=10, max_pixels=1000):
    """Filter objects by size (number of connected pixels)"""
    # This is a simplified size filter - more sophisticated methods would use connected components
    kernel = ee.Kernel.circle(radius=2)
    opened = image.morphologyOpen(kernel)
    return opened

def get_safe_download_url(image, roi, scale=10, max_pixels=1e8):
    """Safely generate download URL with error handling"""
    try:
        download_url = image.getDownloadURL({
            'scale': scale,
            'crs': Settings.OUTPUT_CRS,
            'region': roi,
            'fileFormat': Settings.OUTPUT_FORMAT,
            'maxPixels': max_pixels
        })
        return download_url
    except Exception as e:
        print(f"Error generating download URL at scale {scale}: {e}")
        # Try with higher scale (lower resolution)
        try:
            higher_scale = scale * 2
            download_url = image.getDownloadURL({
                'scale': higher_scale,
                'crs': Settings.OUTPUT_CRS,
                'region': roi,
                'fileFormat': Settings.OUTPUT_FORMAT,
                'maxPixels': max_pixels
            })
            print(f"Generated URL with reduced resolution (scale: {higher_scale})")
            return download_url
        except Exception as e2:
            print(f"Failed to generate download URL even with reduced resolution: {e2}")
            return None

def get_safe_thumbnail_url(image, roi, vis_params=None):
    """Safely generate thumbnail URL with error handling"""
    try:
        if vis_params:
            visualization = image.visualize(**vis_params)
        else:
            visualization = image
            
        thumbnail_url = visualization.getThumbURL({
            'dimensions': Settings.THUMBNAIL_DIMENSIONS,
            'region': roi,
            'format': Settings.THUMBNAIL_FORMAT
        })
        return thumbnail_url
    except Exception as e:
        print(f"Error generating thumbnail URL: {e}")
        return None

def retry_ee_operation(operation, max_retries=3, delay=1):
    """Retry Earth Engine operations with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2
    
def print_collection_stats(collection, name="Collection"):
    """Print statistics about an image collection"""
    info = get_image_collection_info(collection)
    print(f"\n{name} Statistics:")
    print(f"  Images found: {info['size']}")
    print(f"  Date range: {info['date_range']}")
    print(f"  Available bands: {', '.join(info['bands'][:10])}{'...' if len(info['bands']) > 10 else ''}")

def create_timestamp():
    """Create a timestamp string for filenames"""
    return time.strftime('%Y%m%d_%H%M%S')

def safe_getinfo(ee_object, default=None):
    """Safely get info from Earth Engine object"""
    try:
        return ee_object.getInfo()
    except Exception as e:
        print(f"Error getting info: {e}")
        return default
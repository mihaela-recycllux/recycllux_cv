#!/usr/bin/env python3
"""
Multi-Satellite Data Comparison with Google Earth Engine

This script demonstrates how to download and compare data from different satellite missions:
- Sentinel-1 (SAR)
- Sentinel-2 (Optical)
- Landsat 8/9 (Optical)
- MODIS (Moderate Resolution)

It shows how different satellites provide complementary information for various applications.

Author: Varun Burde 
email: varun@recycllux.com
Date: 2025

"""

import os
import ee
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import signal

def with_timeout(seconds=60):
    """Decorator to add timeout to functions"""
    def decorator(func):
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
                return None
        return wrapper
    return decorator

def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project='recycllux-satellite-data')
        print('Earth Engine initialized')
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project='recycllux-satellite-data')
        print('Authenticated and initialized')

def plot_comparison_grid(images, titles, suptitle="Multi-Satellite Comparison"):
    """Plot multiple images in a grid for comparison"""
    n_images = len(images)
    if n_images <= 2:
        rows, cols = 1, n_images
        figsize = (15, 7)
    elif n_images <= 4:
        rows, cols = 2, 2
        figsize = (15, 15)
    else:
        rows, cols = 2, 3
        figsize = (18, 12)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, (image, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(image)
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()

@with_timeout(60)  # 60 second timeout
def download_sentinel2_rgb(roi, start_date, end_date):
    """Download Sentinel-2 RGB image"""
    print("Downloading Sentinel-2 RGB...")
    
    # Use smaller ROI for direct downloads
    roi_bounds = roi.bounds()
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} Sentinel-2 images")
    
    if collection_size.getInfo() == 0:
        print("No Sentinel-2 data available for the specified criteria")
        return None, None, None
    
    # Get median composite
    image = collection.median()
    
    # Create RGB visualization
    rgb_vis = image.visualize(
        bands=['B4', 'B3', 'B2'],
        min=0,
        max=3000
    )
    
    # Get thumbnail
    thumbnail_url = rgb_vis.getThumbURL({
        'dimensions': 512,
        'region': roi,
        'format': 'png'
    })
    
    # Use smaller scale for download to avoid size limits
    download_url = image.select(['B4', 'B3', 'B2']).getDownloadURL({
        'scale': 20,  # Increased scale to reduce file size
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"Sentinel-2 RGB download URL: {download_url}")
    
    return image, thumbnail_url, download_url

@with_timeout(60)  # 60 second timeout
def download_landsat_rgb(roi, start_date, end_date):
    """Download Landsat 8/9 RGB image"""
    print("Downloading Landsat 8/9 RGB...")
    
    # Build collection (combine Landsat 8 and 9)
    l8_collection = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .map(lambda image: image.multiply(0.0000275).add(-0.2))  # Scale factors for L8
    )
    
    l9_collection = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUD_COVER', 20))
        .map(lambda image: image.multiply(0.0000275).add(-0.2))  # Scale factors for L9
    )
    
    # Merge collections
    landsat_collection = l8_collection.merge(l9_collection)
    
    # Get median composite
    image = landsat_collection.median()
    
    # Create RGB visualization
    rgb_vis = image.visualize(
        bands=['SR_B4', 'SR_B3', 'SR_B2'],
        min=0,
        max=0.3
    )
    
    # Get thumbnail
    thumbnail_url = rgb_vis.getThumbURL({
        'dimensions': 512,
        'region': roi,
        'format': 'png'
    })
    
    # Download URL
    download_url = image.select(['SR_B4', 'SR_B3', 'SR_B2']).getDownloadURL({
        'scale': 30,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"Landsat RGB download URL: {download_url}")
    
    return image, thumbnail_url, download_url

@with_timeout(60)  # 60 second timeout
def download_sentinel1_vv(roi, start_date, end_date):
    """Download Sentinel-1 VV polarization"""
    print("Downloading Sentinel-1 SAR...")
    
    # Build collection
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
    )
    
    # Get median composite
    image = collection.median()
    
    # Create visualization in dB scale
    sar_vis = image.visualize(
        min=-25,
        max=0,
        palette=['000000', 'FFFFFF']
    )
    
    # Get thumbnail
    thumbnail_url = sar_vis.getThumbURL({
        'dimensions': 512,
        'region': roi,
        'format': 'png'
    })
    
    # Download URL
    download_url = image.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"Sentinel-1 SAR download URL: {download_url}")
    
    return image, thumbnail_url, download_url

@with_timeout(60)  # 60 second timeout
def download_modis_data(roi, start_date, end_date):
    """Download MODIS data for comparison"""
    print("Downloading MODIS data...")
    
    # Build MODIS Terra collection with updated dataset
    collection = (
        ee.ImageCollection('MODIS/061/MOD09A1')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} MODIS images")
    
    if collection_size.getInfo() == 0:
        print("No MODIS data available for the specified criteria")
        return None, None, None
    
    # Get median composite
    image = collection.median()
    
    # Create RGB visualization (bands 1, 4, 3 for RGB)
    rgb_vis = image.visualize(
        bands=['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'],
        min=0,
        max=3000
    )
    
    # Get thumbnail
    thumbnail_url = rgb_vis.getThumbURL({
        'dimensions': 512,
        'region': roi,
        'format': 'png'
    })
    
    # Download URL
    download_url = image.select(['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03']).getDownloadURL({
        'scale': 500,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"MODIS download URL: {download_url}")
    
    return image, thumbnail_url, download_url

def compare_optical_resolutions(roi, start_date, end_date):
    """Compare different spatial resolutions of Sentinel-2 bands"""
    print("Comparing Sentinel-2 band resolutions...")
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    if collection.size().getInfo() == 0:
        print("No data available for resolution comparison")
        return None, None
    
    # Get median composite
    image = collection.median()
    
    # 10m bands RGB
    rgb_10m = image.visualize(
        bands=['B4', 'B3', 'B2'],
        min=0,
        max=3000
    )
    
    # 20m bands false color (B11, B8A, B5)
    false_color_20m = image.visualize(
        bands=['B11', 'B8A', 'B5'],
        min=0,
        max=3000
    )
    
    # Get thumbnails
    try:
        rgb_thumb_url = rgb_10m.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        false_thumb_url = false_color_20m.getThumbURL({
            'dimensions': 256,  # Smaller to represent 20m resolution
            'region': roi,
            'format': 'png'
        })
        
        # Display comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # RGB 10m
        response_rgb = requests.get(rgb_thumb_url)
        if response_rgb.status_code == 200:
            img_rgb = Image.open(io.BytesIO(response_rgb.content))
            axes[0].imshow(np.array(img_rgb))
        axes[0].set_title('Sentinel-2 RGB (10m resolution)\nB04-B03-B02')
        axes[0].axis('off')
        
        # False color 20m
        response_false = requests.get(false_thumb_url)
        if response_false.status_code == 200:
            img_false = Image.open(io.BytesIO(response_false.content))
            axes[1].imshow(np.array(img_false))
        axes[1].set_title('Sentinel-2 False Color (20m resolution)\nB11-B8A-B05')
        axes[1].axis('off')
        
        plt.suptitle('Resolution Comparison: 10m vs 20m bands')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display resolution comparison: {e}")
    
    return rgb_10m, false_color_20m

def temporal_comparison(roi):
    """Compare the same area across different seasons"""
    print("Performing temporal comparison...")
    
    time_intervals = [
        ('2023-03-01', '2023-03-31'),  # Spring
        ('2023-07-01', '2023-07-31'),  # Summer
        ('2023-10-01', '2023-10-31'),  # Autumn
    ]
    
    season_names = ['Spring', 'Summer', 'Autumn']
    season_images = []
    
    for i, (start, end) in enumerate(time_intervals):
        # Build collection with updated dataset
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )
        
        # Get median composite
        image = collection.median()
        
        # Create RGB visualization
        rgb_vis = image.visualize(
            bands=['B4', 'B3', 'B2'],
            min=0,
            max=3000
        )
        
        try:
            thumbnail_url = rgb_vis.getThumbURL({
                'dimensions': 512,
                'region': roi,
                'format': 'png'
            })
            
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                season_images.append(np.array(img))
            else:
                season_images.append(np.zeros((512, 512, 3)))
        except Exception as e:
            print(f"Could not download {season_names[i]} image: {e}")
            season_images.append(np.zeros((512, 512, 3)))
    
    # Plot temporal comparison
    if season_images:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (image, season) in enumerate(zip(season_images, season_names)):
            axes[i].imshow(image)
            axes[i].set_title(f'{season} 2023')
            axes[i].axis('off')
        
        plt.suptitle('Temporal Comparison - Seasonal Changes')
        plt.tight_layout()
        plt.show()
    
    return season_images

def main():
    """Main function to run multi-satellite comparison examples"""
    print("=== Multi-Satellite Data Comparison with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest - smaller area to avoid size limits
    roi = ee.Geometry.Rectangle([14.0, 46.0, 14.1, 46.1])  # Reduced from 14.2, 46.15
    start_date = '2023-07-01'
    end_date = '2023-07-31'
    
    print(f"Area of Interest: {roi.getInfo()}")
    print(f"Time Interval: {start_date} to {end_date}\n")
    
    print("SATELLITE CHARACTERISTICS:")
    print("- Sentinel-2: 10-60m resolution, 13 spectral bands, 5-day revisit")
    print("- Sentinel-1: 5-20m resolution, SAR (all-weather), 6-day revisit")
    print("- Landsat 8/9: 15-100m resolution, 11 bands, 16-day revisit")
    print("- MODIS: 250m-1km resolution, daily global coverage\n")
    
    try:
        # Example 1: Multi-satellite RGB comparison
        print("1. Multi-Satellite RGB Comparison")
        
        images = []
        titles = []
        
        # Sentinel-2
        try:
            s2_image, s2_thumb_url, s2_download_url = download_sentinel2_rgb(roi, start_date, end_date)
            
            response = requests.get(s2_thumb_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                images.append(np.array(img))
                titles.append("Sentinel-2 RGB (10m)")
        except Exception as e:
            print(f"Could not download Sentinel-2: {e}")
        
        # Landsat
        try:
            l_image, l_thumb_url, l_download_url = download_landsat_rgb(roi, start_date, end_date)
            
            response = requests.get(l_thumb_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                images.append(np.array(img))
                titles.append("Landsat 8/9 RGB (30m)")
        except Exception as e:
            print(f"Could not download Landsat: {e}")
        
        # Sentinel-1
        try:
            s1_image, s1_thumb_url, s1_download_url = download_sentinel1_vv(roi, start_date, end_date)
            
            response = requests.get(s1_thumb_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                images.append(np.array(img))
                titles.append("Sentinel-1 SAR VV (20m)")
        except Exception as e:
            print(f"Could not download Sentinel-1: {e}")
        
        # MODIS
        try:
            m_image, m_thumb_url, m_download_url = download_modis_data(roi, start_date, end_date)
            
            response = requests.get(m_thumb_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                images.append(np.array(img))
                titles.append("MODIS Terra (500m)")
        except Exception as e:
            print(f"Could not download MODIS: {e}")
        
        if images:
            plot_comparison_grid(images, titles, "Multi-Satellite Comparison")
            print("✓ Multi-satellite comparison completed\n")
        
        # Example 2: Resolution comparison
        print("2. Spatial Resolution Comparison")
        try:
            rgb_10m, false_color_20m = compare_optical_resolutions(roi, start_date, end_date)
            print("✓ Resolution comparison completed\n")
        except Exception as e:
            print(f"Resolution comparison failed: {e}\n")
        
        # Example 3: Temporal comparison
        print("3. Temporal Analysis")
        try:
            seasonal_images = temporal_comparison(roi)
            print("✓ Temporal comparison completed\n")
        except Exception as e:
            print(f"Temporal comparison failed: {e}\n")
        
        print("=== Multi-Satellite Analysis completed! ===")
        print("\nSATELLITE SELECTION GUIDE:")
        print("OPTICAL DATA:")
        print("- Sentinel-2: Best for detailed land monitoring (agriculture, forestry)")
        print("- Landsat: Long-term studies, historical analysis")
        print("- MODIS: Large-scale monitoring, daily global coverage")
        print("\nSAR DATA:")
        print("- Sentinel-1: All-weather monitoring, change detection")
        print("- Best for: flood mapping, ship detection, deformation")
        print("\nRESOLUTION TRADE-OFFS:")
        print("- Higher resolution: More detail, smaller coverage")
        print("- Lower resolution: Less detail, larger coverage, more frequent")
        print("\nTips for multi-satellite analysis:")
        print("- Combine optical and SAR for comprehensive monitoring")
        print("- Use temporal stacks for change detection")
        print("- Consider cloud cover patterns in area selection")
        print("- Match acquisition dates when comparing satellites")
        print("- Use Google Earth Engine for large-scale analysis")
        print("- Export tasks to Drive for large datasets")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")

if __name__ == "__main__":
    main()

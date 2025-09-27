#!/usr/bin/env python3
"""
Sentinel-2 Basic Data Download with Google Earth Engine

This script demonstrates how to download basic Sentinel-2 data using Google Earth Engine.
It covers:
- Setting up Earth Engine authentication
- Basic true color images (RGB)
- False color composites
- Individual band downloads
- Different processing levels (L1C vs L2A)

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
import time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds):
    """Decorator to add timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Set the signal handler and a timeout alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError:
                print(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                return None
            finally:
                # Disable the alarm
                signal.alarm(0)
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

def plot_image(img_array, title="Image"):
    """Simple image plotting function"""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_array)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

@with_timeout(60)  # 60 second timeout
def download_sentinel2_true_color(roi, start_date, end_date):
    """
    Download Sentinel-2 true color (RGB) image
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Downloading Sentinel-2 True Color Image...")
    
    # Build Sentinel-2 collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("No data available for the specified criteria")
        return None, None
    
    # Get median composite
    image = collection.median()
    
    # Select RGB bands (B4=Red, B3=Green, B2=Blue)
    rgb_image = image.select(['B4', 'B3', 'B2'])
    
    # Create visualization
    rgb_vis = rgb_image.visualize(
        min=0,
        max=3000
    )
    
    # Get thumbnail for display
    try:
        thumbnail_url = rgb_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            plot_image(np.array(img), "Sentinel-2 True Color RGB")
    except Exception as e:
        print(f"Could not display image: {e}")
    
    # Generate download URL with very high scale to ensure small file size
    try:
        download_url = rgb_image.getDownloadURL({
            'scale': 50,  # Much higher scale to reduce file size
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        print(f"RGB download URL: {download_url}")
        print("✓ True color image processed successfully")
        return rgb_image, download_url
        
    except Exception as e:
        print(f"Error generating download URL: {e}")
        # Try with even higher scale
        try:
            download_url = rgb_image.getDownloadURL({
                'scale': 100,
                'crs': 'EPSG:4326', 
                'region': roi,
                'fileFormat': 'GeoTIFF'
            })
            print(f"RGB download URL (reduced resolution): {download_url}")
            print("✓ True color image processed successfully (reduced resolution)")
            return rgb_image, download_url
        except Exception as e2:
            print(f"Error with reduced resolution: {e2}")
            return None, None

@with_timeout(60)  # 60 second timeout
def download_sentinel2_false_color(roi, start_date, end_date):
    """
    Download Sentinel-2 false color (NIR-Red-Green) image
    Better for vegetation analysis
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Downloading Sentinel-2 False Color Image...")
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("No data available for the specified criteria")
        return None, None
    
    # Get median composite
    image = collection.median()
    
    # Select false color bands (B8=NIR, B4=Red, B3=Green)
    false_color_image = image.select(['B8', 'B4', 'B3'])
    
    # Create visualization
    false_color_vis = false_color_image.visualize(
        min=0,
        max=3000
    )
    
    # Get thumbnail for display
    try:
        thumbnail_url = false_color_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            plot_image(np.array(img), "Sentinel-2 False Color (NIR-Red-Green)")
    except Exception as e:
        print(f"Could not display image: {e}")
    
    # Generate download URL with very high scale to ensure small file size
    try:
        download_url = false_color_image.getDownloadURL({
            'scale': 50,  # Much higher scale to reduce file size
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        print(f"False color download URL: {download_url}")
        print("✓ False color image processed successfully")
        return false_color_image, download_url
        
    except Exception as e:
        print(f"Error generating download URL: {e}")
        # Try with even higher scale
        try:
            download_url = false_color_image.getDownloadURL({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': roi,
                'fileFormat': 'GeoTIFF'
            })
            print(f"False color download URL (reduced resolution): {download_url}")
            print("✓ False color image processed successfully (reduced resolution)")
            return false_color_image, download_url
        except Exception as e2:
            print(f"Error with reduced resolution: {e2}")
            return None, None

@with_timeout(60)  # 60 second timeout
def download_sentinel2_ndvi(roi, start_date, end_date):
    """
    Download Sentinel-2 NDVI (Normalized Difference Vegetation Index)
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Downloading Sentinel-2 NDVI...")
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("No data available for the specified criteria")
        return None, None
    
    # Get median composite
    image = collection.median()
    
    # Calculate NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Create visualization
    ndvi_vis = ndvi.visualize(
        min=-1,
        max=1,
        palette=['red', 'yellow', 'green']
    )
    
    # Get thumbnail for display
    try:
        thumbnail_url = ndvi_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            
            # Plot NDVI with appropriate styling
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(np.array(img))
            ax.set_title('NDVI (Normalized Difference Vegetation Index)')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()
    except Exception as e:
        print(f"Could not display NDVI: {e}")
    
    # Generate download URL with very high scale to ensure small file size
    try:
        download_url = ndvi.getDownloadURL({
            'scale': 50,  # Much higher scale to reduce file size
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        print(f"NDVI download URL: {download_url}")
        print("✓ NDVI calculated and processed successfully")
        return ndvi, download_url
        
    except Exception as e:
        print(f"Error generating download URL: {e}")
        # Try with even higher scale
        try:
            download_url = ndvi.getDownloadURL({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': roi,
                'fileFormat': 'GeoTIFF'
            })
            print(f"NDVI download URL (reduced resolution): {download_url}")
            print("✓ NDVI calculated and processed successfully (reduced resolution)")
            return ndvi, download_url
        except Exception as e2:
            print(f"Error with reduced resolution: {e2}")
            return None, None

@with_timeout(120)  # 120 second timeout for multiple bands
def download_all_bands(roi, start_date, end_date):
    """
    Download all available Sentinel-2 bands separately
    
    Sentinel-2 L2A bands:
    - B01: Coastal aerosol (443nm)
    - B02: Blue (490nm)
    - B03: Green (560nm)
    - B04: Red (665nm)
    - B05: Red Edge 1 (705nm)
    - B06: Red Edge 2 (740nm)
    - B07: Red Edge 3 (783nm)
    - B08: NIR (842nm)
    - B8A: Red Edge 4 (865nm)
    - B09: Water vapor (945nm)
    - B11: SWIR 1 (1610nm)
    - B12: SWIR 2 (2190nm)
    """
    print("Downloading all Sentinel-2 bands...")
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("No data available for the specified criteria")
        return None, None
    
    # Get median composite
    image = collection.median()
    
    # Available bands in L2A
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    band_descriptions = [
        "Coastal aerosol", "Blue", "Green", "Red", "Red Edge 1", "Red Edge 2",
        "Red Edge 3", "NIR", "Red Edge 4", "Water vapor", "SWIR 1", "SWIR 2"
    ]
    
    # Select all bands
    multiband_image = image.select(bands)
    
    # Create visualization for first 6 bands
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (band, desc) in enumerate(zip(bands[:6], band_descriptions[:6])):
        try:
            band_image = image.select(band)
            band_vis = band_image.visualize(min=0, max=3000, palette=['black', 'white'])
            
            thumbnail_url = band_vis.getThumbURL({
                'dimensions': 256,
                'region': roi,
                'format': 'png'
            })
            
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                axes[i].imshow(np.array(img), cmap='gray')
            
            axes[i].set_title(f'{band}: {desc}')
            axes[i].axis('off')
            
        except Exception as e:
            print(f"Could not display band {band}: {e}")
            axes[i].text(0.5, 0.5, f'{band}\nError', ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{band}: {desc}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Generate download URLs for individual bands and multiband
    download_urls = {}
    
    # Individual bands with error handling
    for band, desc in zip(bands, band_descriptions):
        band_image = image.select(band)
        try:
            url = band_image.getDownloadURL({
                'scale': 50,  # Higher scale to reduce file size
                'crs': 'EPSG:4326',
                'region': roi,
                'fileFormat': 'GeoTIFF'
            })
            download_urls[f"{band}_{desc.replace(' ', '_')}"] = url
            print(f"✓ {band} ({desc}) download URL: {url}")
        except Exception as e:
            print(f"✗ Error with {band} at scale 50: {e}")
            # Try with higher scale
            try:
                url = band_image.getDownloadURL({
                    'scale': 100,
                    'crs': 'EPSG:4326',
                    'region': roi,
                    'fileFormat': 'GeoTIFF'
                })
                download_urls[f"{band}_{desc.replace(' ', '_')}"] = url
                print(f"✓ {band} ({desc}) download URL (reduced resolution): {url}")
            except Exception as e2:
                print(f"✗ Failed to generate URL for {band}: {e2}")
                continue
    
    # Multiband image with higher scale to reduce file size
    try:
        multiband_url = multiband_image.getDownloadURL({
            'scale': 50,
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        download_urls['All_Bands'] = multiband_url
        print(f"✓ All bands download URL: {multiband_url}")
    except Exception as e:
        print(f"✗ Error with multiband at scale 50: {e}")
        try:
            multiband_url = multiband_image.getDownloadURL({
                'scale': 100,
                'crs': 'EPSG:4326',
                'region': roi,
                'fileFormat': 'GeoTIFF'
            })
            download_urls['All_Bands'] = multiband_url
            print(f"✓ All bands download URL (reduced resolution): {multiband_url}")
        except Exception as e2:
            print(f"✗ Failed to generate multiband URL: {e2}")
    
    return multiband_image, download_urls

def main():
    """Main function to run the examples"""
    print("=== Sentinel-2 Basic Download Examples with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest (very small ROI for testing)
    roi = ee.Geometry.Rectangle([14.0, 46.0, 14.05, 46.05])
    start_date = '2023-07-01'
    end_date = '2023-07-31'
    
    print(f"Area of Interest: {roi.getInfo()}")
    print(f"Time Interval: {start_date} to {end_date}")
    print("Data Collection: Sentinel-2 L2A\n")
    
    try:
        # Example 1: True color image
        print("1. True Color RGB Image")
        try:
            true_color, rgb_url = download_sentinel2_true_color(roi, start_date, end_date)
            if true_color is not None:
                print("✓ True color image processed successfully\n")
            else:
                print("✗ True color image processing failed\n")
        except Exception as e:
            print(f"✗ Error in true color processing: {e}\n")
        
        # Example 2: False color image
        print("2. False Color (NIR-Red-Green) Image")
        try:
            false_color, false_url = download_sentinel2_false_color(roi, start_date, end_date)
            if false_color is not None:
                print("✓ False color image processed successfully\n")
            else:
                print("✗ False color image processing failed\n")
        except Exception as e:
            print(f"✗ Error in false color processing: {e}\n")
        
        # Example 3: NDVI
        print("3. NDVI (Vegetation Index)")
        try:
            ndvi, ndvi_url = download_sentinel2_ndvi(roi, start_date, end_date)
            if ndvi is not None:
                print("✓ NDVI calculated and processed successfully\n")
            else:
                print("✗ NDVI processing failed\n")
        except Exception as e:
            print(f"✗ Error in NDVI processing: {e}\n")
        
        # Example 4: All bands
        print("4. Individual Band Downloads")
        try:
            all_bands, band_urls = download_all_bands(roi, start_date, end_date)
            if all_bands is not None:
                print("✓ All bands processed successfully\n")
            else:
                print("✗ Band downloads processing failed\n")
        except Exception as e:
            print(f"✗ Error in band downloads processing: {e}\n")
        
        print("=== Examples completed! ===")
        print("\nTips for further exploration:")
        print("- Try different time intervals")
        print("- Experiment with different areas (change roi)")
        print("- Compare L1C vs L2A processing levels")
        print("- Calculate other vegetation indices (EVI, SAVI, etc.)")
        print("- Use batch exports for larger areas")
        print("- Export to Google Drive for large datasets")
        print("- Combine with other Earth Engine datasets")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sentinel-1 SAR Data Download with Google Earth Engine

This script demonstrates how to download Sentinel-1 SAR (Synthetic Aperture Radar) data using Google Earth Engine.
Sentinel-1 provides all-weather, day-and-night imaging capabilities.

SAR data is fundamentally different from optical data:
- Works in all weather conditions
- Not affected by cloud cover
- Measures surface roughness and moisture
- Different polarizations provide different information

This script covers:
- VV and VH polarizations
- Different acquisition modes (IW, EW)
- Ascending vs Descending orbits
- Basic SAR processing techniques

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
        ee.Initialize(project='gedospatial-data')
        print('Earth Engine initialized')
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project='gedospatial-data')
        print('Authenticated and initialized')

def plot_sar_image(image_array, title="SAR Image", db_scale=True):
    """
    Plot SAR image with appropriate scaling
    
    Args:
        image_array: SAR image array
        title: Plot title
        db_scale: Whether to convert to decibel scale
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if db_scale and image_array.max() > 0:
        # Convert to decibel scale for better visualization
        image_db = 10 * np.log10(np.maximum(image_array, 1e-10))
        im = ax.imshow(image_db, cmap='gray', vmin=-25, vmax=0)
        plt.colorbar(im, ax=ax, label='Backscatter (dB)')
    else:
        im = ax.imshow(image_array, cmap='gray')
        plt.colorbar(im, ax=ax, label='Linear backscatter')
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

@with_timeout(60)  # 60 second timeout
def download_sentinel1_vv_vh(roi, start_date, end_date):
    """
    Download Sentinel-1 VV and VH polarization data
    
    VV: Vertical transmit, Vertical receive
    VH: Vertical transmit, Horizontal receive
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Downloading Sentinel-1 VV and VH polarizations...")
    
    # Build Sentinel-1 collection
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )
    
    # Get median composite
    image = collection.median()
    
    # Select VV and VH bands
    vv_vh = image.select(['VV', 'VH'])
    
    # Generate download URLs
    vv_url = vv_vh.select('VV').getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    vh_url = vv_vh.select('VH').getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"VV polarization download URL: {vv_url}")
    print(f"VH polarization download URL: {vh_url}")
    
    # Create RGB visualization for web display
    rgb_vis = vv_vh.visualize(
        bands=['VV', 'VH'],
        min=[-25, -30],
        max=[0, -5],
        palette=['000000', 'FFFFFF']
    )
    
    # Get thumbnail for display
    try:
        thumbnail_url = rgb_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        # Download and display thumbnail
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img_array)
            plt.title('Sentinel-1 VV and VH Polarizations (Composite)')
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Could not display thumbnail: {e}")
    
    return vv_vh, vv_url, vh_url

@with_timeout(60)  # 60 second timeout
def create_sar_rgb_composite(roi, start_date, end_date):
    """
    Create RGB composite from SAR polarizations
    Common composition: VV/VH ratio, VH, VV
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Creating Sentinel-1 RGB composite...")
    
    # Build collection
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )
    
    # Get median composite
    image = collection.median()
    
    # Calculate VV/VH ratio
    vv = image.select('VV')
    vh = image.select('VH')
    ratio = vv.divide(vh.add(0.001))  # Avoid division by zero
    
    # Create RGB composite
    rgb = ee.Image.cat([
        ratio.multiply(0.1),  # Red: VV/VH ratio
        vh.multiply(8),       # Green: VH
        vv.multiply(3)        # Blue: VV
    ]).rename(['R', 'G', 'B'])
    
    # Visualize
    rgb_vis = rgb.visualize(
        bands=['R', 'G', 'B'],
        min=[0, 0, 0],
        max=[1, 1, 1]
    )
    
    # Get thumbnail
    try:
        thumbnail_url = rgb_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img_array)
            plt.title('Sentinel-1 SAR RGB Composite\n(Red: VV/VH ratio, Green: VH, Blue: VV)')
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Could not display thumbnail: {e}")
    
    # Generate download URL
    download_url = rgb.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"RGB composite download URL: {download_url}")
    
    return rgb, download_url

@with_timeout(60)  # 60 second timeout
def compare_ascending_descending(roi, start_date, end_date):
    """
    Compare Sentinel-1 data from ascending and descending orbits
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Comparing ascending vs descending orbit data...")
    
    # Ascending orbit collection
    asc_collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
    )
    
    # Descending orbit collection
    des_collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
    )
    
    # Get median composites
    asc_image = asc_collection.median()
    des_image = des_collection.median()
    
    # Check if both collections have data
    asc_count = asc_collection.size()
    des_count = des_collection.size()
    
    print(f"Ascending orbit images available: {asc_count.getInfo()}")
    print(f"Descending orbit images available: {des_count.getInfo()}")
    
    if asc_count.getInfo() > 0 and des_count.getInfo() > 0:
        # Generate download URLs
        asc_url = asc_image.getDownloadURL({
            'scale': 20,
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        
        des_url = des_image.getDownloadURL({
            'scale': 20,
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        
        print(f"Ascending orbit download URL: {asc_url}")
        print(f"Descending orbit download URL: {des_url}")
        
        # Create side-by-side visualization
        combined = ee.Image.cat([asc_image, des_image]).rename(['Ascending', 'Descending'])
        
        # Get thumbnail for comparison
        try:
            asc_thumb_url = asc_image.visualize(min=-25, max=0, palette=['000000', 'FFFFFF']).getThumbURL({
                'dimensions': 256,
                'region': roi,
                'format': 'png'
            })
            
            des_thumb_url = des_image.visualize(min=-25, max=0, palette=['000000', 'FFFFFF']).getThumbURL({
                'dimensions': 256,
                'region': roi,
                'format': 'png'
            })
            
            # Display both
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ascending
            response_asc = requests.get(asc_thumb_url)
            if response_asc.status_code == 200:
                img_asc = Image.open(io.BytesIO(response_asc.content))
                axes[0].imshow(np.array(img_asc), cmap='gray')
            axes[0].set_title('Ascending Orbit (VV)')
            axes[0].axis('off')
            
            # Descending
            response_des = requests.get(des_thumb_url)
            if response_des.status_code == 200:
                img_des = Image.open(io.BytesIO(response_des.content))
                axes[1].imshow(np.array(img_des), cmap='gray')
            axes[1].set_title('Descending Orbit (VV)')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not display comparison: {e}")
        
        return asc_image, des_image, asc_url, des_url
    else:
        print("Note: Could not find both ascending and descending data for the specified time period.")
        return None, None, None, None

@with_timeout(60)  # 60 second timeout
def calculate_sar_coherence(roi, start_date, end_date):
    """
    Calculate coherence from two SAR acquisitions (interferometry)
    Note: This is a simplified example - real coherence calculation requires
    proper temporal baseline consideration
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Calculating SAR coherence (simplified example)...")
    
    # Build collection
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select(['VV', 'VH'])
    )
    
    # Get median composite
    image = collection.median()
    
    # Simplified coherence-like measure
    vv = image.select('VV')
    vh = image.select('VH')
    coherence = vv.multiply(vh).sqrt().divide(vv.add(vh).add(0.001))
    
    # Visualize
    coherence_vis = coherence.visualize(
        min=0,
        max=1,
        palette=['000000', '0000FF', '00FFFF', '00FF00', 'FFFF00', 'FF0000']
    )
    
    # Get thumbnail
    try:
        thumbnail_url = coherence_vis.getThumbURL({
            'dimensions': 512,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(thumbnail_url)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(img_array)
            plt.title('Simplified Coherence-like Measure')
            plt.axis('off')
            plt.show()
    except Exception as e:
        print(f"Could not display coherence: {e}")
    
    # Generate download URL
    download_url = coherence.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"Coherence download URL: {download_url}")
    
    return coherence, download_url

def main():
    """Main function to run the SAR examples"""
    print("=== Sentinel-1 SAR Download Examples with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest (smaller area for faster processing)
    roi = ee.Geometry.Rectangle([11.9, 46.4, 12.0, 46.5])
    start_date = '2023-07-01'
    end_date = '2023-07-31'
    
    print(f"Area of Interest: {roi.getInfo()}")
    print(f"Time Interval: {start_date} to {end_date}")
    print("Data Collection: Sentinel-1 IW (Interferometric Wide Swath)\n")
    
    print("SAR DATA CHARACTERISTICS:")
    print("- VV: Better for detecting volume scattering (vegetation, urban areas)")
    print("- VH: Better for detecting surface scattering (water, smooth surfaces)")
    print("- SAR works day/night and in all weather conditions")
    print("- Backscatter intensity depends on surface roughness and moisture\n")
    
    try:
        # Example 1: VV and VH polarizations
        print("1. VV and VH Polarization Data")
        vv_vh_image, vv_url, vh_url = download_sentinel1_vv_vh(roi, start_date, end_date)
        print("✓ VV and VH data processed successfully\n")
        
        # Example 2: RGB composite
        print("2. SAR RGB Composite")
        rgb_composite, rgb_url = create_sar_rgb_composite(roi, start_date, end_date)
        print("✓ RGB composite created successfully\n")
        
        # Example 3: Ascending vs Descending
        print("3. Ascending vs Descending Orbits")
        asc_data, des_data, asc_url, des_url = compare_ascending_descending(roi, start_date, end_date)
        if asc_data is not None:
            print("✓ Orbit comparison completed successfully\n")
        else:
            print("→ Orbit comparison skipped (insufficient data)\n")
        
        # Example 4: Coherence-like measure
        print("4. Coherence-like Measure")
        coherence, coherence_url = calculate_sar_coherence(roi, start_date, end_date)
        print("✓ Coherence measure calculated successfully\n")
        
        print("=== SAR Examples completed successfully! ===")
        print("\nKey SAR Applications:")
        print("- Ship detection (VV polarization over water)")
        print("- Flood mapping (temporal change detection)")
        print("- Crop monitoring (VH/VV ratio)")
        print("- Deformation monitoring (interferometry)")
        print("- Ice monitoring (temporal analysis)")
        print("\nTips for further exploration:")
        print("- Try different time periods for temporal analysis")
        print("- Experiment with different areas (urban vs rural)")
        print("- Calculate temporal differences for change detection")
        print("- Explore different SAR indices (RVI, RFDI, etc.)")
        print("- Use Google Earth Engine Apps for interactive visualization")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")
        print("Note: SAR data availability might be limited for some areas/times.")

if __name__ == "__main__":
    main()

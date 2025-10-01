#!/usr/bin/env python3
"""
Advanced Sentinel-2 Band Analysis with Google Earth Engine

This script demonstrates advanced band combinations and spectral indices
for various applications like agriculture, water detection, urban analysis, etc.

Spectral bands and their applications:
- B01    # Generate download URL
    ndwi_url = ndwi.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })): Coastal aerosol, atmospheric correction
- B02 (490nm): Blue, water detection
- B03 (560nm): Green, vegetation health
- B04 (665nm): Red, chlorophyll absorption
- B05 (705nm): Red Edge 1, vegetation stress
- B06 (740nm): Red Edge 2, vegetation health
- B07 (783nm): Red Edge 3, vegetation structure
- B08 (842nm): NIR, biomass, vegetation vigor
- B8A (865nm): Red Edge 4, vegetation analysis
- B09 (945nm): Water vapor
- B11 (1610nm): SWIR 1, moisture content
- B12 (2190nm): SWIR 2, geology, soil

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

def plot_index_with_colorbar(img_array, title, cmap='RdYlGn'):
    """Plot an index with appropriate colorbar"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(img_array, cmap=cmap)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Index Value')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

@with_timeout(60)  # 60 second timeout
def calculate_vegetation_indices(roi, start_date, end_date):
    """
    Calculate various vegetation indices from Sentinel-2 data
    
    Vegetation Indices:
    - NDVI: (NIR - Red) / (NIR + Red)
    - EVI: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    - SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    - NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    """
    print("Calculating vegetation indices...")
    
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
        return {}
    
    # Get median composite
    image = collection.median()
    
    # Scale factor for Sentinel-2 SR data
    image = image.multiply(0.0001)
    
    # Select bands
    blue = image.select('B2')
    red = image.select('B4')
    redEdge = image.select('B5')
    nir = image.select('B8')
    
    # Calculate vegetation indices
    # NDVI
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # EVI
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': nir,
            'RED': red,
            'BLUE': blue
        }).rename('EVI')
    
    # SAVI (Soil Adjusted Vegetation Index)
    savi = nir.subtract(red).divide(nir.add(red).add(0.5)).multiply(1.5).rename('SAVI')
    
    # NDRE (Normalized Difference Red Edge)
    ndre = nir.subtract(redEdge).divide(nir.add(redEdge)).rename('NDRE')
    
    # Combine all indices
    indices = ee.Image.cat([ndvi, evi, savi, ndre])
    
    # Create visualizations
    ndvi_vis = ndvi.visualize(min=-1, max=1, palette=['red', 'yellow', 'green'])
    evi_vis = evi.visualize(min=-1, max=1, palette=['red', 'yellow', 'green'])
    savi_vis = savi.visualize(min=-1, max=1, palette=['red', 'yellow', 'green'])
    ndre_vis = ndre.visualize(min=-1, max=1, palette=['red', 'yellow', 'green'])
    
    # Get thumbnails for display
    try:
        # Create subplot for all indices
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        indices_vis = [
            (ndvi_vis, "NDVI (Normalized Difference Vegetation Index)"),
            (evi_vis, "EVI (Enhanced Vegetation Index)"),
            (savi_vis, "SAVI (Soil Adjusted Vegetation Index)"),
            (ndre_vis, "NDRE (Normalized Difference Red Edge)")
        ]
        
        for i, (vis, title) in enumerate(indices_vis):
            row, col = i // 2, i % 2
            
            thumbnail_url = vis.getThumbURL({
                'dimensions': 512,
                'region': roi,
                'format': 'png'
            })
            
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                axes[row, col].imshow(np.array(img))
            
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display indices: {e}")
    
    # Generate download URLs
    ndvi_url = ndvi.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    evi_url = evi.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"NDVI download URL: {ndvi_url}")
    print(f"EVI download URL: {evi_url}")
    
    return {"NDVI": ndvi, "EVI": evi, "SAVI": savi, "NDRE": ndre}

@with_timeout(60)  # 60 second timeout
def calculate_water_indices(roi, start_date, end_date):
    """
    Calculate water-related indices from Sentinel-2 data
    
    Water Indices:
    - NDWI: (Green - NIR) / (Green + NIR)
    - MNDWI: (Green - SWIR1) / (Green + SWIR1)
    - AWEI: Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2
    """
    print("Calculating water indices...")
    
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
        return {}
    
    # Get median composite
    image = collection.median()
    
    # Scale factor for Sentinel-2 SR data
    image = image.multiply(0.0001)

@with_timeout(60)  # 60 second timeout
def calculate_built_up_indices(roi, start_date, end_date):
    """
    Calculate built-up/urban area indices
    
    Built-up Indices:
    - NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)
    - UI: (SWIR2 - NIR) / (SWIR2 + NIR)
    - BU: (Red + SWIR1) - (NIR + Blue)
    """
    print("Calculating built-up area indices...")
    
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
        return {}
    
    # Get median composite and scale
    image = collection.median().multiply(0.0001)
    
    # Select bands
    blue = image.select('B2')
    red = image.select('B4')
    nir = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    
    # Calculate built-up indices
    # NDBI (Normalized Difference Built-up Index)
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')
    
    # UI (Urban Index)
    ui = swir2.subtract(nir).divide(swir2.add(nir)).rename('UI')
    
    # BU (Built-up Index)
    bu = red.add(swir1).subtract(nir.add(blue)).rename('BU')
    
    # Create visualizations
    ndbi_vis = ndbi.visualize(min=-1, max=1, palette=['blue', 'white', 'red'])
    ui_vis = ui.visualize(min=-1, max=1, palette=['blue', 'white', 'red'])
    bu_vis = bu.visualize(min=-0.5, max=0.5, palette=['blue', 'white', 'red'])
    
    # Display built-up indices
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        builtup_indices = [
            (ndbi_vis, "NDBI"),
            (ui_vis, "UI"),
            (bu_vis, "BU")
        ]
        
        for i, (vis, title) in enumerate(builtup_indices):
            thumbnail_url = vis.getThumbURL({
                'dimensions': 512,
                'region': roi,
                'format': 'png'
            })
            
            response = requests.get(thumbnail_url)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                axes[i].imshow(np.array(img))
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display built-up indices: {e}")
    
    # Generate download URL
    ndbi_url = ndbi.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"NDBI download URL: {ndbi_url}")
    
    return {"NDBI": ndbi, "UI": ui, "BU": bu}

@with_timeout(60)  # 60 second timeout
def create_advanced_composites(roi, start_date, end_date):
    """
    Create advanced false-color composites using different band combinations
    """
    print("Creating advanced composites...")
    
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
        return
    
    # Get median composite
    image = collection.median()
    
    # Scale the image
    image = image.multiply(0.0001).multiply(255).uint8()

def main():
    """Main function to run advanced band analysis examples"""
    print("=== Advanced Sentinel-2 Band Analysis with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest (smaller ROI for faster processing)
    roi = ee.Geometry.Rectangle([14.0, 46.0, 14.1, 46.1])
    start_date = '2023-07-01'
    end_date = '2023-07-31'
    
    print(f"Area of Interest: {roi.getInfo()}")
    print(f"Time Interval: {start_date} to {end_date}")
    print("Data Collection: Sentinel-2 L2A\n")
    
    print("SPECTRAL BAND APPLICATIONS:")
    print("- Blue (490nm): Water body detection, atmospheric correction")
    print("- Green (560nm): Vegetation vigor, water quality")
    print("- Red (665nm): Chlorophyll absorption, vegetation health")
    print("- Red Edge (705-865nm): Vegetation stress, leaf area index")
    print("- NIR (842nm): Biomass, vegetation structure")
    print("- SWIR (1610-2190nm): Moisture content, geology, burned areas\n")
    
    try:
        # Example 1: Vegetation indices
        print("1. Vegetation Indices Analysis")
        vegetation_indices = calculate_vegetation_indices(roi, start_date, end_date)
        print("✓ Vegetation indices calculated successfully\n")
        
        # Example 2: Water indices
        print("2. Water Detection Indices")
        water_indices = calculate_water_indices(roi, start_date, end_date)
        print("✓ Water indices calculated successfully\n")
        
        # Example 3: Built-up area indices
        print("3. Built-up Area Indices")
        builtup_indices = calculate_built_up_indices(roi, start_date, end_date)
        print("✓ Built-up indices calculated successfully\n")
        
        # Example 4: Advanced composites
        print("4. Advanced Band Composites")
        composites = create_advanced_composites(roi, start_date, end_date)
        print("✓ Advanced composites created successfully\n")
        
        print("=== Advanced Analysis completed successfully! ===")
        print("\nINDEX INTERPRETATION GUIDE:")
        print("VEGETATION INDICES:")
        print("- NDVI > 0.3: Dense vegetation")
        print("- NDVI 0.1-0.3: Sparse vegetation")
        print("- NDVI < 0.1: Non-vegetated areas")
        print("\nWATER INDICES:")
        print("- NDWI > 0: Water bodies")
        print("- MNDWI > 0: Open water")
        print("- AWEI > 0: Water presence")
        print("\nBUILT-UP INDICES:")
        print("- NDBI > 0: Built-up areas")
        print("- Higher values = more urbanized")
        print("\nTips for further exploration:")
        print("- Combine indices for land cover classification")
        print("- Use temporal analysis for change detection")
        print("- Apply thresholds for binary masks")
        print("- Calculate index differences for monitoring")
        print("- Use Google Earth Engine Apps for interactive analysis")
        print("- Export large areas using Earth Engine batch processing")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")

if __name__ == "__main__":
    main()

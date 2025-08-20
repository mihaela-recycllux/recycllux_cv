#!/usr/bin/env python3
"""
Time Series Analysis and Change Detection with Google Earth Engine

This script demonstrates how to perform temporal analysis using satellite data:
- Time series creation and analysis
- Change detection techniques
- Vegetation phenology monitoring
- Urban expansion tracking
- Disaster monitoring (floods, fires, etc.)

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
import pandas as pd
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

@with_timeout(120)  # 120 second timeout for time series
def create_monthly_time_series(roi, year=2023):
    """
    Create a monthly time series of NDVI values
    
    Args:
        roi: Region of interest (ee.Geometry)
        year: Year for the time series
    """
    print(f"Creating monthly NDVI time series for {year}...")
    
    # Function to add NDVI band with harmonized collection
    def add_ndvi(image):
        # Ensure we have the required bands
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.select(['B2', 'B3', 'B4', 'B8']).addBands(ndvi)
    
    # Create monthly time series
    months = []
    ndvi_values = []
    ndvi_images = []
    
    for month in range(1, 13):
        # Calculate date range for month
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        month_str = start_date.format('YYYY-MM').getInfo()
        months.append(month_str)
        
        print(f"Processing {month_str}...")
        
        # Build collection with updated dataset
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .select(['B2', 'B3', 'B4', 'B8'])  # Select only required bands
            .map(add_ndvi)
        )
        
        # Get median composite
        if collection.size().getInfo() > 0:
            median_image = collection.median()
            ndvi_image = median_image.select('NDVI')
            
            # Calculate mean NDVI over the region
            mean_ndvi = ndvi_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=10,
                maxPixels=1e9
            ).getInfo()
            
            ndvi_value = mean_ndvi.get('NDVI', np.nan)
            ndvi_values.append(ndvi_value)
            ndvi_images.append(ndvi_image)
        else:
            print(f"No data available for {month_str}")
            ndvi_values.append(np.nan)
            ndvi_images.append(None)
    
    # Plot time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot mean NDVI over time
    valid_months = [m for m, v in zip(months, ndvi_values) if not np.isnan(v)]
    valid_values = [v for v in ndvi_values if not np.isnan(v)]
    
    ax1.plot(valid_months, valid_values, 'o-', linewidth=2, markersize=8)
    ax1.set_title('NDVI Time Series - Monthly Averages')
    ax1.set_ylabel('Mean NDVI')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.0)
    
    # Rotate x-axis labels
    for label in ax1.get_xticklabels():
        label.set_rotation(45)
    
    # Plot some example NDVI maps
    months_to_show = [2, 5, 8, 11]  # March, June, September, December
    
    for i, month_idx in enumerate(months_to_show):
        if month_idx < len(ndvi_images) and ndvi_images[month_idx] is not None:
            ax = plt.subplot(2, 4, 5 + i)
            
            try:
                # Create visualization
                ndvi_vis = ndvi_images[month_idx].visualize(
                    min=-0.2,
                    max=0.8,
                    palette=['red', 'yellow', 'green']
                )
                
                # Get thumbnail
                thumbnail_url = ndvi_vis.getThumbURL({
                    'dimensions': 256,
                    'region': roi,
                    'format': 'png'
                })
                
                response = requests.get(thumbnail_url)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    ax.imshow(np.array(img))
                
                ax.set_title(months[month_idx])
            except Exception as e:
                print(f"Could not display {months[month_idx]}: {e}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(months[month_idx])
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return months, ndvi_images, ndvi_values

@with_timeout(60)  # 60 second timeout
def detect_changes_rgb(roi, date1, date2):
    """
    Perform change detection using RGB imagery
    
    Args:
        roi: Region of interest (ee.Geometry)
        date1: First date (before)
        date2: Second date (after)
    """
    print(f"Detecting changes between {date1} and {date2}...")
    
    # Function to get image for date
    def get_image_for_date(date):
        start_date = ee.Date(date)
        end_date = start_date.advance(10, 'day')  # 10-day window
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )
        
        return collection.median().select(['B4', 'B3', 'B2'])
    
    # Get images
    image1 = get_image_for_date(date1)
    image2 = get_image_for_date(date2)
    
    # Calculate change magnitude
    diff = image2.subtract(image1)
    change_magnitude = diff.select(['B4', 'B3', 'B2']).reduce(ee.Reducer.sum()).abs()
    
    # Create visualizations
    image1_vis = image1.visualize(min=0, max=3000)
    image2_vis = image2.visualize(min=0, max=3000)
    change_vis = change_magnitude.visualize(min=0, max=2000, palette=['black', 'red', 'yellow', 'white'])
    
    # Get thumbnails and display
    try:
        # Get thumbnail URLs
        thumb1_url = image1_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        thumb2_url = image2_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        change_thumb_url = change_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        
        # Download and display
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Before image
        response1 = requests.get(thumb1_url)
        if response1.status_code == 200:
            img1 = Image.open(io.BytesIO(response1.content))
            axes[0, 0].imshow(np.array(img1))
        axes[0, 0].set_title(f'Before: {date1}')
        axes[0, 0].axis('off')
        
        # After image
        response2 = requests.get(thumb2_url)
        if response2.status_code == 200:
            img2 = Image.open(io.BytesIO(response2.content))
            axes[0, 1].imshow(np.array(img2))
        axes[0, 1].set_title(f'After: {date2}')
        axes[0, 1].axis('off')
        
        # Change magnitude
        response_change = requests.get(change_thumb_url)
        if response_change.status_code == 200:
            img_change = Image.open(io.BytesIO(response_change.content))
            axes[1, 0].imshow(np.array(img_change))
        axes[1, 0].set_title('Change Magnitude')
        axes[1, 0].axis('off')
        
        # Change detection binary mask
        change_threshold = change_magnitude.gt(1000)  # Threshold for significant change
        change_mask_vis = change_threshold.visualize(palette=['black', 'white'])
        
        change_mask_url = change_mask_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response_mask = requests.get(change_mask_url)
        if response_mask.status_code == 200:
            img_mask = Image.open(io.BytesIO(response_mask.content))
            axes[1, 1].imshow(np.array(img_mask))
        axes[1, 1].set_title('Significant Changes')
        axes[1, 1].axis('off')
        
        plt.suptitle('RGB Change Detection Analysis')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display change detection results: {e}")
    
    # Generate download URLs
    image1_url = image1.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    image2_url = image2.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"Before image download URL: {image1_url}")
    print(f"After image download URL: {image2_url}")
    
    return image1, image2, change_magnitude

@with_timeout(120)  # Increase timeout to 120 seconds for change detection
def ndvi_change_detection(roi, date1, date2):
    """
    Perform change detection using NDVI
    
    Args:
        roi: Region of interest (ee.Geometry)
        date1: First date (before)
        date2: Second date (after)
    """
    print(f"Performing NDVI change detection between {date1} and {date2}...")
    
    # Function to calculate NDVI for a date
    def get_ndvi_for_date(date):
        start_date = ee.Date(date)
        end_date = start_date.advance(10, 'day')
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        )
        
        median_image = collection.median()
        ndvi = median_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi
    
    # Get NDVI images
    ndvi1 = get_ndvi_for_date(date1)
    ndvi2 = get_ndvi_for_date(date2)
    
    # Calculate NDVI difference
    ndvi_diff = ndvi2.subtract(ndvi1)
    
    # Classification of changes
    vegetation_loss = ndvi_diff.lt(-0.1)  # Significant decrease
    vegetation_gain = ndvi_diff.gt(0.1)   # Significant increase
    no_change = ndvi_diff.abs().lte(0.1)
    
    # Create change classification map
    change_map = vegetation_loss.multiply(-1).add(vegetation_gain.multiply(1))
    
    # Create visualizations
    ndvi1_vis = ndvi1.visualize(min=-0.2, max=0.8, palette=['red', 'yellow', 'green'])
    ndvi2_vis = ndvi2.visualize(min=-0.2, max=0.8, palette=['red', 'yellow', 'green'])
    ndvi_diff_vis = ndvi_diff.visualize(min=-0.5, max=0.5, palette=['red', 'white', 'blue'])
    change_map_vis = change_map.visualize(min=-1, max=1, palette=['red', 'gray', 'green'])
    
    # Display results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # NDVI before
        thumb1_url = ndvi1_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response1 = requests.get(thumb1_url)
        if response1.status_code == 200:
            img1 = Image.open(io.BytesIO(response1.content))
            axes[0, 0].imshow(np.array(img1))
        axes[0, 0].set_title(f'NDVI Before: {date1}')
        axes[0, 0].axis('off')
        
        # NDVI after
        thumb2_url = ndvi2_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response2 = requests.get(thumb2_url)
        if response2.status_code == 200:
            img2 = Image.open(io.BytesIO(response2.content))
            axes[0, 1].imshow(np.array(img2))
        axes[0, 1].set_title(f'NDVI After: {date2}')
        axes[0, 1].axis('off')
        
        # NDVI difference
        diff_thumb_url = ndvi_diff_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response_diff = requests.get(diff_thumb_url)
        if response_diff.status_code == 200:
            img_diff = Image.open(io.BytesIO(response_diff.content))
            axes[1, 0].imshow(np.array(img_diff))
        axes[1, 0].set_title('NDVI Difference (After - Before)')
        axes[1, 0].axis('off')
        
        # Change classification
        change_thumb_url = change_map_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response_change = requests.get(change_thumb_url)
        if response_change.status_code == 200:
            img_change = Image.open(io.BytesIO(response_change.content))
            axes[1, 1].imshow(np.array(img_change))
        axes[1, 1].set_title('Change Classification\n(Red: Loss, Gray: No Change, Green: Gain)')
        axes[1, 1].axis('off')
        
        plt.suptitle('NDVI Change Detection Analysis')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display NDVI change detection: {e}")
    
    # Calculate statistics
    loss_area = vegetation_loss.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    gain_area = vegetation_gain.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    total_area = ee.Image.pixelArea().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e9
    ).getInfo()
    
    loss_area_km2 = loss_area.get('nd', 0) / 1e6
    gain_area_km2 = gain_area.get('nd', 0) / 1e6
    total_area_km2 = total_area.get('area', 1) / 1e6
    
    print(f"\nChange Detection Statistics:")
    print(f"Vegetation Loss: {loss_area_km2:.2f} km² ({loss_area_km2/total_area_km2*100:.1f}%)")
    print(f"Vegetation Gain: {gain_area_km2:.2f} km² ({gain_area_km2/total_area_km2*100:.1f}%)")
    
    # Generate download URLs
    ndvi1_url = ndvi1.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    ndvi_diff_url = ndvi_diff.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"NDVI before download URL: {ndvi1_url}")
    print(f"NDVI difference download URL: {ndvi_diff_url}")
    
    return ndvi1, ndvi2, ndvi_diff, change_map

@with_timeout(60)  # 60 second timeout
def sar_flood_detection(roi, date_before, date_flood):
    """
    Detect floods using SAR data (water appears dark in SAR)
    
    Args:
        roi: Region of interest (ee.Geometry)
        date_before: Date before flood
        date_flood: Date during/after flood
    """
    print(f"Detecting flood using SAR data...")
    
    # Function to get SAR image for date
    def get_sar_for_date(date):
        start_date = ee.Date(date)
        end_date = start_date.advance(5, 'day')
        
        collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .select('VV')
        )
        
        return collection.median()
    
    # Get SAR images
    sar_before = get_sar_for_date(date_before)
    sar_flood = get_sar_for_date(date_flood)
    
    # Calculate difference (flooded areas will show decrease in backscatter)
    sar_diff = sar_flood.subtract(sar_before)
    
    # Detect potential flood areas (significant decrease in backscatter)
    flood_threshold = -3  # dB decrease
    potential_flood = sar_diff.lt(flood_threshold)
    
    # Create visualizations
    sar_before_vis = sar_before.visualize(min=-25, max=0, palette=['000000', 'FFFFFF'])
    sar_flood_vis = sar_flood.visualize(min=-25, max=0, palette=['000000', 'FFFFFF'])
    sar_diff_vis = sar_diff.visualize(min=-10, max=10, palette=['red', 'white', 'blue'])
    flood_vis = potential_flood.visualize(palette=['white', 'blue'])
    
    # Display results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # SAR before
        before_url = sar_before_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response1 = requests.get(before_url)
        if response1.status_code == 200:
            img1 = Image.open(io.BytesIO(response1.content))
            axes[0, 0].imshow(np.array(img1))
        axes[0, 0].set_title(f'SAR Before: {date_before}')
        axes[0, 0].axis('off')
        
        # SAR during flood
        flood_url = sar_flood_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response2 = requests.get(flood_url)
        if response2.status_code == 200:
            img2 = Image.open(io.BytesIO(response2.content))
            axes[0, 1].imshow(np.array(img2))
        axes[0, 1].set_title(f'SAR During Flood: {date_flood}')
        axes[0, 1].axis('off')
        
        # SAR difference
        diff_url = sar_diff_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response_diff = requests.get(diff_url)
        if response_diff.status_code == 200:
            img_diff = Image.open(io.BytesIO(response_diff.content))
            axes[1, 0].imshow(np.array(img_diff))
        axes[1, 0].set_title('SAR Difference (Flood - Before)')
        axes[1, 0].axis('off')
        
        # Flood detection
        flood_mask_url = flood_vis.getThumbURL({'dimensions': 512, 'region': roi, 'format': 'png'})
        response_flood = requests.get(flood_mask_url)
        if response_flood.status_code == 200:
            img_flood = Image.open(io.BytesIO(response_flood.content))
            axes[1, 1].imshow(np.array(img_flood))
        axes[1, 1].set_title(f'Potential Flood Areas\n(Decrease > {abs(flood_threshold)} dB)')
        axes[1, 1].axis('off')
        
        plt.suptitle('SAR Flood Detection Analysis')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not display flood detection: {e}")
    
    # Calculate flood area
    flood_area = potential_flood.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=20,
        maxPixels=1e9
    ).getInfo()
    
    flood_area_km2 = flood_area.get('VV', 0) / 1e6
    print(f"\nFlood Detection Results:")
    print(f"Potential flood area: {flood_area_km2:.2f} km²")
    
    # Generate download URLs
    sar_before_url = sar_before.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    potential_flood_url = potential_flood.getDownloadURL({
        'scale': 20,
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    
    print(f"SAR before download URL: {sar_before_url}")
    print(f"Flood mask download URL: {potential_flood_url}")
    
    return sar_before, sar_flood, sar_diff, potential_flood

def main():
    """Main function to run time series and change detection examples"""
    print("=== Time Series Analysis and Change Detection with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest - smaller area to avoid processing issues
    roi = ee.Geometry.Rectangle([14.0, 46.0, 14.1, 46.1])  # Reduced area size
    
    print(f"Area of Interest: {roi.getInfo()}\n")
    
    print("TIME SERIES APPLICATIONS:")
    print("- Vegetation phenology monitoring")
    print("- Agricultural crop monitoring") 
    print("- Urban expansion tracking")
    print("- Disaster impact assessment")
    print("- Climate change indicators\n")
    
    try:
        # Example 1: Monthly NDVI time series
        print("1. Monthly NDVI Time Series")
        months, ndvi_images, mean_ndvi = create_monthly_time_series(roi, 2023)
        print("✓ Time series analysis completed\n")
        
        # Example 2: RGB change detection
        print("2. RGB Change Detection")
        try:
            img1, img2, change_mag = detect_changes_rgb(roi, '2023-03-15', '2023-09-15')
            print("✓ RGB change detection completed\n")
        except Exception as e:
            print(f"RGB change detection failed: {e}\n")
        
        # Example 3: NDVI change detection
        print("3. NDVI Change Detection")
        try:
            ndvi1, ndvi2, ndvi_diff, change_map = ndvi_change_detection(roi, '2023-03-15', '2023-09-15')
            print("✓ NDVI change detection completed\n")
        except Exception as e:
            print(f"NDVI change detection failed: {e}\n")
        
        # Example 4: SAR flood detection
        print("4. SAR-based Flood Detection")
        try:
            sar_before, sar_flood, sar_diff, flood_mask = sar_flood_detection(roi, '2023-06-01', '2023-06-15')
            print("✓ SAR flood detection completed\n")
        except Exception as e:
            print(f"SAR flood detection failed: {e}\n")
        
        print("=== Time Series Analysis completed! ===")
        print("\nCHANGE DETECTION METHODS:")
        print("IMAGE DIFFERENCING:")
        print("- Simple subtraction of pixel values")
        print("- Good for detecting magnitude of change")
        print("- Sensitive to atmospheric conditions")
        print("\nVEGETATION INDICES:")
        print("- NDVI differences for vegetation monitoring")
        print("- Less sensitive to atmospheric effects")
        print("- Good for agricultural and forest monitoring")
        print("\nSAR ANALYSIS:")
        print("- All-weather monitoring capability")
        print("- Excellent for flood and disaster detection")
        print("- Surface roughness change detection")
        print("\nTips for effective change detection:")
        print("- Use cloud-free images when possible")
        print("- Consider seasonal variations")
        print("- Apply appropriate thresholds")
        print("- Validate results with ground truth data")
        print("- Use multiple dates for trend analysis")
        print("- Leverage Google Earth Engine's temporal filtering")
        print("- Export large analyses to Google Drive")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")

if __name__ == "__main__":
    main()

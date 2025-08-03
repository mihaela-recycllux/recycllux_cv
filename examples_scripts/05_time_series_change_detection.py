#!/usr/bin/env python3
"""
Time Series Analysis and Change Detection Script

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
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

def setup_credentials():
    """Set up Sentinel Hub credentials"""
    load_dotenv()
    config = SHConfig()
    
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    
    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! Please provide SH_CLIENT_ID and SH_CLIENT_SECRET in your .env file")
        return None
    
    return config

def create_monthly_time_series(config, bbox, year=2023, size=(256, 256)):
    """
    Create a monthly time series of NDVI values
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        year: Year for the time series
        size: Output image size in pixels
    """
    print(f"Creating monthly NDVI time series for {year}...")
    
    evalscript_ndvi = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B04", "B08"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """
    
    # Create monthly time intervals
    months = []
    ndvi_images = []
    mean_ndvi_values = []
    
    for month in range(1, 13):
        # Calculate start and end dates for each month
        start_date = dt.date(year, month, 1)
        if month == 12:
            end_date = dt.date(year + 1, 1, 1) - dt.timedelta(days=1)
        else:
            end_date = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
        
        time_interval = (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        months.append(start_date.strftime('%Y-%m'))
        
        print(f"Processing {start_date.strftime('%B %Y')}...")
        
        request = SentinelHubRequest(
            evalscript=evalscript_ndvi,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config,
        )
        
        try:
            ndvi_data = request.get_data()[0].squeeze()
            ndvi_images.append(ndvi_data)
            
            # Calculate mean NDVI (excluding invalid values)
            valid_ndvi = ndvi_data[(ndvi_data >= -1) & (ndvi_data <= 1)]
            mean_ndvi = np.mean(valid_ndvi) if len(valid_ndvi) > 0 else np.nan
            mean_ndvi_values.append(mean_ndvi)
            
        except Exception as e:
            print(f"Could not get data for {start_date.strftime('%B')}: {e}")
            ndvi_images.append(np.full(size[::-1], np.nan))
            mean_ndvi_values.append(np.nan)
    
    # Plot time series
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot mean NDVI over time
    ax1.plot(months, mean_ndvi_values, 'o-', linewidth=2, markersize=8)
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
        ax = plt.subplot(2, 4, 5 + i)
        if not np.all(np.isnan(ndvi_images[month_idx])):
            im = ax.imshow(ndvi_images[month_idx], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
            ax.set_title(months[month_idx])
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(months[month_idx])
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return months, ndvi_images, mean_ndvi_values

def detect_changes_rgb(config, bbox, date1, date2, size=(512, 512)):
    """
    Perform change detection using RGB imagery
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        date1: First date (before)
        date2: Second date (after)
        size: Output image size in pixels
    """
    print(f"Detecting changes between {date1} and {date2}...")
    
    evalscript_rgb = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    
    # Download image 1 (before)
    request1 = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date1, date1),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Download image 2 (after)
    request2 = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date2, date2),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    image1 = request1.get_data()[0]
    image2 = request2.get_data()[0]
    
    # Calculate change magnitude
    change_magnitude = np.sqrt(np.sum((image2.astype(float) - image1.astype(float))**2, axis=2))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title(f'Before: {date1}')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    axes[0, 1].imshow(image2)
    axes[0, 1].set_title(f'After: {date2}')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    # Change magnitude
    im = axes[1, 0].imshow(change_magnitude, cmap='hot')
    axes[1, 0].set_title('Change Magnitude')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(im, ax=axes[1, 0])
    
    # Change detection binary mask
    threshold = np.percentile(change_magnitude, 90)  # Top 10% of changes
    change_mask = change_magnitude > threshold
    axes[1, 1].imshow(change_mask, cmap='gray')
    axes[1, 1].set_title(f'Significant Changes (>{threshold:.1f})')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.suptitle('RGB Change Detection Analysis')
    plt.tight_layout()
    plt.show()
    
    return image1, image2, change_magnitude, change_mask

def ndvi_change_detection(config, bbox, date1, date2, size=(512, 512)):
    """
    Perform change detection using NDVI
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        date1: First date (before)
        date2: Second date (after)
        size: Output image size in pixels
    """
    print(f"Performing NDVI change detection between {date1} and {date2}...")
    
    evalscript_ndvi = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B04", "B08"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
        return [ndvi];
    }
    """
    
    # Download NDVI for date 1
    request1 = SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date1, date1),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Download NDVI for date 2
    request2 = SentinelHubRequest(
        evalscript=evalscript_ndvi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date2, date2),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    ndvi1 = request1.get_data()[0].squeeze()
    ndvi2 = request2.get_data()[0].squeeze()
    
    # Calculate NDVI difference
    ndvi_diff = ndvi2 - ndvi1
    
    # Classification of changes
    vegetation_loss = ndvi_diff < -0.1  # Significant decrease
    vegetation_gain = ndvi_diff > 0.1   # Significant increase
    no_change = np.abs(ndvi_diff) <= 0.1
    
    # Create change classification map
    change_map = np.zeros_like(ndvi_diff)
    change_map[vegetation_loss] = -1  # Red for loss
    change_map[no_change] = 0         # Gray for no change
    change_map[vegetation_gain] = 1   # Green for gain
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # NDVI before
    im1 = axes[0, 0].imshow(ndvi1, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axes[0, 0].set_title(f'NDVI Before: {date1}')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # NDVI after
    im2 = axes[0, 1].imshow(ndvi2, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axes[0, 1].set_title(f'NDVI After: {date2}')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # NDVI difference
    im3 = axes[1, 0].imshow(ndvi_diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title('NDVI Difference (After - Before)')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Change classification
    colors = ['red', 'gray', 'green']
    im4 = axes[1, 1].imshow(change_map, cmap='RdGy', vmin=-1, vmax=1)
    axes[1, 1].set_title('Change Classification\n(Red: Loss, Gray: No Change, Green: Gain)')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.suptitle('NDVI Change Detection Analysis')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_pixels = ndvi_diff.size
    loss_pixels = np.sum(vegetation_loss)
    gain_pixels = np.sum(vegetation_gain)
    no_change_pixels = np.sum(no_change)
    
    print(f"\nChange Detection Statistics:")
    print(f"Vegetation Loss: {loss_pixels} pixels ({loss_pixels/total_pixels*100:.1f}%)")
    print(f"Vegetation Gain: {gain_pixels} pixels ({gain_pixels/total_pixels*100:.1f}%)")
    print(f"No Significant Change: {no_change_pixels} pixels ({no_change_pixels/total_pixels*100:.1f}%)")
    
    return ndvi1, ndvi2, ndvi_diff, change_map

def sar_flood_detection(config, bbox, date_before, date_flood, size=(512, 512)):
    """
    Detect floods using SAR data (water appears dark in SAR)
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        date_before: Date before flood
        date_flood: Date during/after flood
        size: Output image size in pixels
    """
    print(f"Detecting flood using SAR data...")
    
    evalscript_sar = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["VV"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.VV];
    }
    """
    
    # Download SAR before flood
    request_before = SentinelHubRequest(
        evalscript=evalscript_sar,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=(date_before, date_before),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Download SAR during flood
    request_flood = SentinelHubRequest(
        evalscript=evalscript_sar,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=(date_flood, date_flood),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    sar_before = request_before.get_data()[0].squeeze()
    sar_flood = request_flood.get_data()[0].squeeze()
    
    # Convert to dB
    sar_before_db = 10 * np.log10(np.maximum(sar_before, 1e-10))
    sar_flood_db = 10 * np.log10(np.maximum(sar_flood, 1e-10))
    
    # Calculate difference (flooded areas will show decrease in backscatter)
    sar_diff = sar_flood_db - sar_before_db
    
    # Detect potential flood areas (significant decrease in backscatter)
    flood_threshold = -3  # dB decrease
    potential_flood = sar_diff < flood_threshold
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # SAR before
    im1 = axes[0, 0].imshow(sar_before_db, cmap='gray', vmin=-25, vmax=0)
    axes[0, 0].set_title(f'SAR Before: {date_before}')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0], label='Backscatter (dB)')
    
    # SAR during flood
    im2 = axes[0, 1].imshow(sar_flood_db, cmap='gray', vmin=-25, vmax=0)
    axes[0, 1].set_title(f'SAR During Flood: {date_flood}')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1], label='Backscatter (dB)')
    
    # SAR difference
    im3 = axes[1, 0].imshow(sar_diff, cmap='RdBu', vmin=-10, vmax=10)
    axes[1, 0].set_title('SAR Difference (Flood - Before)')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(im3, ax=axes[1, 0], label='Difference (dB)')
    
    # Flood detection
    axes[1, 1].imshow(potential_flood, cmap='Blues')
    axes[1, 1].set_title(f'Potential Flood Areas\n(Decrease > {abs(flood_threshold)} dB)')
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    plt.suptitle('SAR Flood Detection Analysis')
    plt.tight_layout()
    plt.show()
    
    flood_area_pixels = np.sum(potential_flood)
    total_pixels = potential_flood.size
    print(f"\nFlood Detection Results:")
    print(f"Potential flood area: {flood_area_pixels} pixels ({flood_area_pixels/total_pixels*100:.1f}%)")
    
    return sar_before_db, sar_flood_db, sar_diff, potential_flood

def main():
    """Main function to run time series and change detection examples"""
    print("=== Time Series Analysis and Change Detection ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest
    bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)
    
    print(f"Area of Interest: {bbox}\n")
    
    print("TIME SERIES APPLICATIONS:")
    print("- Vegetation phenology monitoring")
    print("- Agricultural crop monitoring") 
    print("- Urban expansion tracking")
    print("- Disaster impact assessment")
    print("- Climate change indicators\n")
    
    try:
        # Example 1: Monthly NDVI time series
        print("1. Monthly NDVI Time Series")
        months, ndvi_images, mean_ndvi = create_monthly_time_series(config, bbox, 2023)
        print("✓ Time series analysis completed\n")
        
        # Example 2: RGB change detection
        print("2. RGB Change Detection")
        try:
            img1, img2, change_mag, change_mask = detect_changes_rgb(
                config, bbox, '2023-03-15', '2023-09-15'
            )
            print("✓ RGB change detection completed\n")
        except Exception as e:
            print(f"RGB change detection failed: {e}\n")
        
        # Example 3: NDVI change detection
        print("3. NDVI Change Detection")
        try:
            ndvi1, ndvi2, ndvi_diff, change_map = ndvi_change_detection(
                config, bbox, '2023-03-15', '2023-09-15'
            )
            print("✓ NDVI change detection completed\n")
        except Exception as e:
            print(f"NDVI change detection failed: {e}\n")
        
        # Example 4: SAR flood detection
        print("4. SAR-based Flood Detection")
        try:
            sar_before, sar_flood, sar_diff, flood_mask = sar_flood_detection(
                config, bbox, '2023-06-01', '2023-06-15'
            )
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()

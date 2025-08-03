#!/usr/bin/env python3
"""
Multi-Satellite Data Comparison Script

This script demonstrates how to download and compare data from different satellite missions:
- Sentinel-1 (SAR)
- Sentinel-2 (Optical)
- Landsat 8/9 (Optical)
- Sentinel-3 (Ocean and land monitoring)

It shows how different satellites provide complementary information for various applications.

Author: Learning Script
Date: 2025
"""

import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
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
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                axes[i].imshow(image)
            else:
                # Single band or grayscale
                axes[i].imshow(image.squeeze(), cmap='gray')
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()

def download_sentinel2_rgb(config, bbox, time_interval, size=(512, 512)):
    """Download Sentinel-2 RGB image"""
    print("Downloading Sentinel-2 RGB...")
    
    evalscript = """
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
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    return request.get_data()[0]

def download_landsat_rgb(config, bbox, time_interval, size=(512, 512)):
    """Download Landsat 8/9 RGB image"""
    print("Downloading Landsat 8/9 RGB...")
    
    evalscript = """
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
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.LANDSAT_OT_L2,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    return request.get_data()[0]

def download_sentinel1_vv(config, bbox, time_interval, size=(512, 512)):
    """Download Sentinel-1 VV polarization"""
    print("Downloading Sentinel-1 SAR...")
    
    evalscript = """
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
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    sar_data = request.get_data()[0].squeeze()
    # Convert to dB scale for visualization
    sar_db = 10 * np.log10(np.maximum(sar_data, 1e-10))
    return sar_db

def compare_optical_resolutions(config, bbox, time_interval):
    """Compare different spatial resolutions of Sentinel-2 bands"""
    print("Comparing Sentinel-2 band resolutions...")
    
    # 10m bands (B02, B03, B04, B08)
    evalscript_10m = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08"]
            }],
            output: {
                bands: 4,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08];
    }
    """
    
    # 20m bands (B05, B06, B07, B8A, B11, B12)
    evalscript_20m = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B05", "B06", "B07", "B8A", "B11", "B12"]
            }],
            output: {
                bands: 6,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B05, sample.B06, sample.B07, sample.B8A, sample.B11, sample.B12];
    }
    """
    
    # Download 10m bands
    request_10m = SentinelHubRequest(
        evalscript=evalscript_10m,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(512, 512),
        config=config,
    )
    
    # Download 20m bands
    request_20m = SentinelHubRequest(
        evalscript=evalscript_20m,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=(256, 256),  # Smaller size to represent 20m resolution
        config=config,
    )
    
    data_10m = request_10m.get_data()[0]
    data_20m = request_20m.get_data()[0]
    
    # Create RGB from 10m bands
    rgb_10m = data_10m[:, :, [2, 1, 0]]  # B04, B03, B02
    rgb_10m = np.clip(rgb_10m * 3.5, 0, 1)
    
    # Create false color from 20m bands
    false_color_20m = data_20m[:, :, [4, 3, 0]]  # B11, B8A, B05
    false_color_20m = np.clip(false_color_20m * 3.5, 0, 1)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(rgb_10m)
    axes[0].set_title('Sentinel-2 RGB (10m resolution)\nB04-B03-B02')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].imshow(false_color_20m)
    axes[1].set_title('Sentinel-2 False Color (20m resolution)\nB11-B8A-B05')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.suptitle('Resolution Comparison: 10m vs 20m bands')
    plt.tight_layout()
    plt.show()
    
    return rgb_10m, false_color_20m

def temporal_comparison(config, bbox, size=(512, 512)):
    """Compare the same area across different seasons"""
    print("Performing temporal comparison...")
    
    time_intervals = [
        ('2023-03-01', '2023-03-31'),  # Spring
        ('2023-07-01', '2023-07-31'),  # Summer
        ('2023-10-01', '2023-10-31'),  # Autumn
    ]
    
    season_names = ['Spring', 'Summer', 'Autumn']
    season_images = []
    
    evalscript = """
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
    
    for time_interval in time_intervals:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config,
        )
        
        try:
            image = request.get_data()[0]
            season_images.append(image)
        except Exception as e:
            print(f"Could not download data for {time_interval}: {e}")
            season_images.append(np.zeros((size[1], size[0], 3)))
    
    # Plot temporal comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (image, season) in enumerate(zip(season_images, season_names)):
        axes[i].imshow(image)
        axes[i].set_title(f'{season} 2023')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.suptitle('Temporal Comparison - Seasonal Changes')
    plt.tight_layout()
    plt.show()
    
    return season_images

def main():
    """Main function to run multi-satellite comparison examples"""
    print("=== Multi-Satellite Data Comparison ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest
    bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)
    time_interval = ('2023-07-01', '2023-07-31')
    
    print(f"Area of Interest: {bbox}")
    print(f"Time Interval: {time_interval}\n")
    
    print("SATELLITE CHARACTERISTICS:")
    print("- Sentinel-2: 10-60m resolution, 13 spectral bands, 5-day revisit")
    print("- Sentinel-1: 5-20m resolution, SAR (all-weather), 6-day revisit")
    print("- Landsat 8/9: 15-100m resolution, 11 bands, 16-day revisit")
    print("- Sentinel-3: 300m-1.2km resolution, ocean/land monitoring\n")
    
    try:
        # Example 1: Multi-satellite RGB comparison
        print("1. Multi-Satellite RGB Comparison")
        
        images = []
        titles = []
        
        # Sentinel-2
        try:
            s2_rgb = download_sentinel2_rgb(config, bbox, time_interval)
            images.append(s2_rgb)
            titles.append("Sentinel-2 RGB (10m)")
        except Exception as e:
            print(f"Could not download Sentinel-2: {e}")
        
        # Landsat
        try:
            landsat_rgb = download_landsat_rgb(config, bbox, time_interval)
            images.append(landsat_rgb)
            titles.append("Landsat 8/9 RGB (30m)")
        except Exception as e:
            print(f"Could not download Landsat: {e}")
        
        # Sentinel-1
        try:
            s1_vv = download_sentinel1_vv(config, bbox, time_interval)
            images.append(s1_vv)
            titles.append("Sentinel-1 SAR VV (20m)")
        except Exception as e:
            print(f"Could not download Sentinel-1: {e}")
        
        if images:
            plot_comparison_grid(images, titles, "Multi-Satellite Comparison")
            print("✓ Multi-satellite comparison completed\n")
        
        # Example 2: Resolution comparison
        print("2. Spatial Resolution Comparison")
        try:
            rgb_10m, false_color_20m = compare_optical_resolutions(config, bbox, time_interval)
            print("✓ Resolution comparison completed\n")
        except Exception as e:
            print(f"Resolution comparison failed: {e}\n")
        
        # Example 3: Temporal comparison
        print("3. Temporal Analysis")
        try:
            seasonal_images = temporal_comparison(config, bbox)
            print("✓ Temporal comparison completed\n")
        except Exception as e:
            print(f"Temporal comparison failed: {e}\n")
        
        print("=== Multi-Satellite Analysis completed! ===")
        print("\nSATELLITE SELECTION GUIDE:")
        print("OPTICAL DATA:")
        print("- Sentinel-2: Best for detailed land monitoring (agriculture, forestry)")
        print("- Landsat: Long-term studies, historical analysis")
        print("- Sentinel-3: Large-scale monitoring, ocean color")
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()

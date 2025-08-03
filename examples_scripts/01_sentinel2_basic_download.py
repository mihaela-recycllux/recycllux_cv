#!/usr/bin/env python3
"""
Sentinel-2 Basic Data Download Script

This script demonstrates how to download basic Sentinel-2 data using different bands and configurations.
It covers:
- Setting up authentication
- Basic true color images (RGB)
- False color composites
- Individual band downloads±
- Different processing levels (L1C vs L2A)

Author: Varun Burde 
email: varun@recycllux.com
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
    
    # Load credentials from environment variables
    config.sh_client_id = os.getenv("SH_CLIENT_ID")
    config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    
    if not config.sh_client_id or not config.sh_client_secret:
        print("Warning! Please provide SH_CLIENT_ID and SH_CLIENT_SECRET in your .env file")
        print("You can get these credentials from: https://apps.sentinel-hub.com/")
        return None
    
    return config

def plot_image(image, factor=1, clip_range=None, **kwargs):
    """Simple image plotting function"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def download_sentinel2_true_color(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-2 true color (RGB) image
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Downloading Sentinel-2 True Color Image...")
    
    # Evalscript for true color RGB composite
    # Uses bands B04 (Red), B03 (Green), B02 (Blue)
    evalscript_true_color = """
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
        evalscript=evalscript_true_color,
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
    
    image = request.get_data()[0]
    plot_image(image, factor=3.5/255, clip_range=(0, 1))
    return image

def download_sentinel2_false_color(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-2 false color (NIR-Red-Green) image
    Better for vegetation analysis
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Downloading Sentinel-2 False Color Image...")
    
    # Evalscript for false color composite
    # Uses bands B08 (NIR), B04 (Red), B03 (Green)
    evalscript_false_color = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B03", "B04", "B08"]
            }],
            output: {
                bands: 3
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B08, sample.B04, sample.B03];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_false_color,
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
    
    image = request.get_data()[0]
    plot_image(image, factor=3.5/255, clip_range=(0, 1))
    return image

def download_sentinel2_ndvi(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-2 NDVI (Normalized Difference Vegetation Index)
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Downloading Sentinel-2 NDVI...")
    
    # Evalscript for NDVI calculation
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
    
    ndvi_data = request.get_data()[0]
    
    # Plot NDVI with appropriate colormap
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(ndvi_data.squeeze(), cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('NDVI (Normalized Difference Vegetation Index)')
    plt.colorbar(im, ax=ax, label='NDVI Value')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    return ndvi_data

def download_all_sentinel2_bands(config, bbox, time_interval, size=(512, 512)):
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
    
    # Available bands in L2A
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    band_data = {}
    
    for band in bands:
        print(f"Downloading band {band}...")
        
        evalscript = f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{
                    bands: ["{band}"]
                }}],
                output: {{
                    bands: 1,
                    sampleType: "FLOAT32"
                }}
            }};
        }}
        
        function evaluatePixel(sample) {{
            return [sample.{band}];
        }}
        """
        
        request = SentinelHubRequest(
            evalscript=evalscript,
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
        
        data = request.get_data()[0]
        band_data[band] = data.squeeze()
    
    # Plot first 6 bands as example
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, band in enumerate(bands[:6]):
        axes[i].imshow(band_data[band], cmap='gray')
        axes[i].set_title(f'Band {band}')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return band_data

def main():
    """Main function to run the examples"""
    print("=== Sentinel-2 Basic Download Examples ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest (example: Lake Bled, Slovenia)
    bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)
    time_interval = ('2023-07-01', '2023-07-31')
    
    print(f"Area of Interest: {bbox}")
    print(f"Time Interval: {time_interval}")
    print(f"Data Collection: Sentinel-2 L2A\n")
    
    try:
        # Example 1: True color image
        print("1. True Color RGB Image")
        true_color = download_sentinel2_true_color(config, bbox, time_interval)
        print("✓ True color image downloaded successfully\n")
        
        # Example 2: False color image
        print("2. False Color (NIR-Red-Green) Image")
        false_color = download_sentinel2_false_color(config, bbox, time_interval)
        print("✓ False color image downloaded successfully\n")
        
        # Example 3: NDVI
        print("3. NDVI (Vegetation Index)")
        ndvi = download_sentinel2_ndvi(config, bbox, time_interval)
        print("✓ NDVI calculated and downloaded successfully\n")
        
        # Example 4: All bands
        print("4. Individual Band Downloads")
        all_bands = download_all_sentinel2_bands(config, bbox, time_interval)
        print("✓ All bands downloaded successfully\n")
        
        print("=== Examples completed successfully! ===")
        print("\nTips for further exploration:")
        print("- Try different time intervals")
        print("- Experiment with different areas (change bbox)")
        print("- Compare L1C vs L2A processing levels")
        print("- Calculate other vegetation indices (EVI, SAVI, etc.)")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sentinel-1 SAR Data Download Script

This script demonstrates how to download Sentinel-1 SAR (Synthetic Aperture Radar) data.
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

def plot_sar_image(image, title="SAR Image", db_scale=True):
    """
    Plot SAR image with appropriate scaling
    
    Args:
        image: SAR image array
        title: Plot title
        db_scale: Whether to convert to decibel scale
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if db_scale and image.max() > 0:
        # Convert to decibel scale for better visualization
        image_db = 10 * np.log10(np.maximum(image, 1e-10))
        im = ax.imshow(image_db, cmap='gray', vmin=-25, vmax=0)
        plt.colorbar(im, ax=ax, label='Backscatter (dB)')
    else:
        im = ax.imshow(image, cmap='gray')
        plt.colorbar(im, ax=ax, label='Linear backscatter')
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def download_sentinel1_vv_vh(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-1 VV and VH polarization data
    
    VV: Vertical transmit, Vertical receive
    VH: Vertical transmit, Horizontal receive
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Downloading Sentinel-1 VV and VH polarizations...")
    
    # Evalscript for VV and VH polarizations
    evalscript_sar = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.VV, sample.VH];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_sar,
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
    
    sar_data = request.get_data()[0]
    
    # Extract VV and VH channels
    vv_data = sar_data[:, :, 0]
    vh_data = sar_data[:, :, 1]
    
    # Plot both polarizations
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # VV polarization
    vv_db = 10 * np.log10(np.maximum(vv_data, 1e-10))
    im1 = axes[0].imshow(vv_db, cmap='gray', vmin=-25, vmax=0)
    axes[0].set_title('VV Polarization (dB)')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], label='Backscatter (dB)')
    
    # VH polarization
    vh_db = 10 * np.log10(np.maximum(vh_data, 1e-10))
    im2 = axes[1].imshow(vh_db, cmap='gray', vmin=-30, vmax=-5)
    axes[1].set_title('VH Polarization (dB)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], label='Backscatter (dB)')
    
    plt.tight_layout()
    plt.show()
    
    return vv_data, vh_data

def download_sentinel1_rgb_composite(config, bbox, time_interval, size=(512, 512)):
    """
    Create RGB composite from SAR polarizations
    Common composition: VV/VH ratio, VH, VV
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Creating Sentinel-1 RGB composite...")
    
    # Evalscript for SAR RGB composite
    evalscript_rgb = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
            }],
            output: {
                bands: 3
            }
        };
    }
    
    function evaluatePixel(sample) {
        var vv = sample.VV;
        var vh = sample.VH;
        var ratio = vv / (vh + 0.001); // Avoid division by zero
        
        // Enhance and normalize for RGB display
        var r = Math.min(1, Math.max(0, (ratio - 1) / 10));
        var g = Math.min(1, Math.max(0, vh * 8));
        var b = Math.min(1, Math.max(0, vv * 3));
        
        return [r, g, b];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    rgb_image = request.get_data()[0]
    
    # Plot RGB composite
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb_image)
    ax.set_title('Sentinel-1 SAR RGB Composite\n(Red: VV/VH ratio, Green: VH, Blue: VV)')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    return rgb_image

def compare_ascending_descending(config, bbox, time_interval, size=(512, 512)):
    """
    Compare Sentinel-1 data from ascending and descending orbits
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Comparing ascending vs descending orbit data...")
    
    evalscript_vv = """
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
    
    # Download ascending orbit data
    request_asc = SentinelHubRequest(
        evalscript=evalscript_vv,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW_ASC,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Download descending orbit data
    request_des = SentinelHubRequest(
        evalscript=evalscript_vv,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW_DES,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    try:
        asc_data = request_asc.get_data()[0].squeeze()
        des_data = request_des.get_data()[0].squeeze()
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Ascending orbit
        asc_db = 10 * np.log10(np.maximum(asc_data, 1e-10))
        im1 = axes[0].imshow(asc_db, cmap='gray', vmin=-25, vmax=0)
        axes[0].set_title('Ascending Orbit (VV)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(im1, ax=axes[0], label='Backscatter (dB)')
        
        # Descending orbit
        des_db = 10 * np.log10(np.maximum(des_data, 1e-10))
        im2 = axes[1].imshow(des_db, cmap='gray', vmin=-25, vmax=0)
        axes[1].set_title('Descending Orbit (VV)')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im2, ax=axes[1], label='Backscatter (dB)')
        
        plt.tight_layout()
        plt.show()
        
        return asc_data, des_data
        
    except Exception as e:
        print(f"Note: Could not download both ascending and descending data: {e}")
        print("This might be because data is not available for both orbits in the specified time period.")
        return None, None

def download_sentinel1_coherence(config, bbox, time_interval, size=(512, 512)):
    """
    Calculate coherence from two SAR acquisitions (interferometry)
    Note: This is a simplified example - real coherence calculation requires
    proper temporal baseline consideration
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
    """
    print("Calculating SAR coherence (simplified example)...")
    
    # This is a basic example - real coherence needs proper processing
    evalscript_coherence = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        // Simplified coherence-like measure
        var vv = sample.VV;
        var vh = sample.VH;
        var coherence = Math.sqrt(vv * vh) / (vv + vh + 0.001);
        return [coherence];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_coherence,
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
    
    coherence_data = request.get_data()[0].squeeze()
    
    # Plot coherence
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(coherence_data, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Simplified Coherence-like Measure')
    plt.colorbar(im, ax=ax, label='Coherence')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    
    return coherence_data

def main():
    """Main function to run the SAR examples"""
    print("=== Sentinel-1 SAR Download Examples ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest (example: Agricultural area)
    bbox = BBox(bbox=[11.9, 46.4, 12.2, 46.6], crs=CRS.WGS84)
    time_interval = ('2023-07-01', '2023-07-31')
    
    print(f"Area of Interest: {bbox}")
    print(f"Time Interval: {time_interval}")
    print("Data Collection: Sentinel-1 IW (Interferometric Wide Swath)\n")
    
    print("SAR DATA CHARACTERISTICS:")
    print("- VV: Better for detecting volume scattering (vegetation, urban areas)")
    print("- VH: Better for detecting surface scattering (water, smooth surfaces)")
    print("- SAR works day/night and in all weather conditions")
    print("- Backscatter intensity depends on surface roughness and moisture\n")
    
    try:
        # Example 1: VV and VH polarizations
        print("1. VV and VH Polarization Data")
        vv_data, vh_data = download_sentinel1_vv_vh(config, bbox, time_interval)
        print("✓ VV and VH data downloaded successfully\n")
        
        # Example 2: RGB composite
        print("2. SAR RGB Composite")
        rgb_composite = download_sentinel1_rgb_composite(config, bbox, time_interval)
        print("✓ RGB composite created successfully\n")
        
        # Example 3: Ascending vs Descending
        print("3. Ascending vs Descending Orbits")
        asc_data, des_data = compare_ascending_descending(config, bbox, time_interval)
        if asc_data is not None:
            print("✓ Orbit comparison completed successfully\n")
        else:
            print("→ Orbit comparison skipped (insufficient data)\n")
        
        # Example 4: Coherence-like measure
        print("4. Coherence-like Measure")
        coherence = download_sentinel1_coherence(config, bbox, time_interval)
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")
        print("Note: SAR data availability might be limited for some areas/times.")

if __name__ == "__main__":
    main()

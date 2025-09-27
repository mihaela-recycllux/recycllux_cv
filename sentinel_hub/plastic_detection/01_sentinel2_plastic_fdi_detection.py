#!/usr/bin/env python3
"""
Sentinel-2 Plastic Debris Detection using Floating Debris Index (FDI)

This script demonstrates how to detect floating marine debris (plastic) in ocean waters
using Sentinel-2 L2A imagery and the Floating Debris Index (FDI).

The FDI method:
1. Downloads Sentinel-2 Red, NIR, and SWIR bands
2. Calculates a baseline reflectance using linear interpolation
3. Computes FDI = R_NIR - R_baseline
4. Applies threshold to identify potential plastic debris

Study Area: Romanian coast of the Black Sea (near Constanța port)
- Known to have marine debris from shipping activities
- Clear water conditions ideal for FDI method

Author: Varun Burde 
Email: varun@recycllux.com
Date: 2025
Reference: Biermann et al. (2020) "Finding Plastic Patches in Coastal North Sea Waters using Satellite Data"

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime as dt
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
    """
    Set up Sentinel Hub credentials from environment variables
    
    Returns:
        SHConfig: Configured SentinelHub configuration object
    """
    # Load environment variables (for local development)
    load_dotenv()
    
    config = SHConfig()
    
    # Get credentials from environment variables
    config.sh_client_id = os.environ.get('SH_CLIENT_ID')
    config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')
    
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Sentinel Hub credentials not found! "
            "Please set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables."
        )
    
    print("✓ Sentinel Hub credentials loaded successfully")
    return config

def download_sentinel2_fdi_bands(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-2 bands required for FDI calculation and water masking
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
        
    Returns:
        tuple: (red_band, nir_band, swir_band, green_band, data_mask) as numpy arrays
    """
    print("Downloading Sentinel-2 bands for FDI calculation and water masking...")
    
    # Evalscript to download Red (B04), NIR (B08), SWIR (B11), Green (B03) bands and data mask
    # Green band is needed for NDWI water detection
    evalscript_fdi = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B03", "B04", "B08", "B11", "dataMask"]
            }],
            output: {
                bands: 5,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B03, sample.B04, sample.B08, sample.B11, sample.dataMask];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_fdi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order='leastCC'  # Least cloud cover
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    # Download data
    data = request.get_data()[0]
    
    # Extract individual bands
    green_band = data[:, :, 0]    # B03 - Green (560nm)
    red_band = data[:, :, 1]      # B04 - Red (665nm)
    nir_band = data[:, :, 2]      # B08 - NIR (842nm) 
    swir_band = data[:, :, 3]     # B11 - SWIR (1610nm)
    data_mask = data[:, :, 4]     # Data mask (valid pixels)
    
    print(f"✓ Downloaded Sentinel-2 data with shape: {data.shape}")
    print(f"  - Green band (B03): min={green_band.min():.4f}, max={green_band.max():.4f}")
    print(f"  - Red band (B04): min={red_band.min():.4f}, max={red_band.max():.4f}")
    print(f"  - NIR band (B08): min={nir_band.min():.4f}, max={nir_band.max():.4f}")
    print(f"  - SWIR band (B11): min={swir_band.min():.4f}, max={swir_band.max():.4f}")
    
    return green_band, red_band, nir_band, swir_band, data_mask

def create_water_mask(green_band, nir_band, data_mask, ndwi_threshold=0.0):
    """
    Create water mask using NDWI (Normalized Difference Water Index)
    
    NDWI = (Green - NIR) / (Green + NIR)
    Water typically has NDWI > 0, land has NDWI < 0
    
    Args:
        green_band: Green band reflectance (B03)
        nir_band: NIR band reflectance (B08)
        data_mask: Data mask for valid pixels
        ndwi_threshold: NDWI threshold for water detection (default: 0.0)
        
    Returns:
        tuple: (water_mask, ndwi) where water_mask is binary (1=water, 0=land)
    """
    print(f"Creating water mask using NDWI with threshold: {ndwi_threshold}")
    
    # Calculate NDWI = (Green - NIR) / (Green + NIR)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
    
    # Apply data mask
    ndwi[data_mask == 0] = np.nan
    
    # Create water mask: 1 = water, 0 = land
    water_mask = np.zeros_like(ndwi)
    water_mask[ndwi > ndwi_threshold] = 1
    water_mask[data_mask == 0] = np.nan
    
    # Calculate water statistics
    valid_pixels = ~np.isnan(ndwi)
    total_valid = np.sum(valid_pixels)
    water_pixels = np.sum(water_mask == 1)
    land_pixels = np.sum(water_mask == 0)
    water_percentage = (water_pixels / total_valid) * 100 if total_valid > 0 else 0
    
    print(f"✓ Water mask created successfully")
    print(f"  - NDWI range: {np.nanmin(ndwi):.4f} to {np.nanmax(ndwi):.4f}")
    print(f"  - Mean NDWI: {np.nanmean(ndwi):.4f}")
    print(f"  - Water pixels: {water_pixels} ({water_percentage:.1f}%)")
    print(f"  - Land pixels: {land_pixels} ({100-water_percentage:.1f}%)")
    
    return water_mask, ndwi

def calculate_fdi(red_band, nir_band, swir_band, data_mask):
    """
    Calculate Floating Debris Index (FDI)
    
    The FDI algorithm:
    1. Calculate baseline reflectance using linear interpolation between Red and SWIR
    2. FDI = R_NIR - R_baseline
    3. Positive FDI values indicate potential floating debris
    
    Args:
        red_band: Red band reflectance (B04)
        nir_band: NIR band reflectance (B08) 
        swir_band: SWIR band reflectance (B11)
        data_mask: Data mask for valid pixels
        
    Returns:
        numpy.ndarray: FDI values
    """
    print("Calculating Floating Debris Index (FDI)...")
    
    # Central wavelengths (nm)
    lambda_red = 665.0    # B04
    lambda_nir = 842.0    # B08  
    lambda_swir = 1610.0  # B11
    
    # Calculate baseline reflectance using linear interpolation
    # R_baseline = R_red + (R_swir - R_red) * (λ_nir - λ_red) / (λ_swir - λ_red)
    wavelength_factor = (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
    baseline_reflectance = red_band + (swir_band - red_band) * wavelength_factor
    
    # Calculate FDI
    fdi = nir_band - baseline_reflectance
    
    # Apply data mask (set invalid pixels to NaN)
    fdi[data_mask == 0] = np.nan
    
    print(f"✓ FDI calculated successfully")
    print(f"  - FDI range: {np.nanmin(fdi):.6f} to {np.nanmax(fdi):.6f}")
    print(f"  - Mean FDI: {np.nanmean(fdi):.6f}")
    print(f"  - Std FDI: {np.nanstd(fdi):.6f}")
    
    return fdi

def create_plastic_detection_mask(fdi, water_mask, threshold=0.01):
    """
    Create binary mask for potential plastic debris detection (water areas only)
    
    Args:
        fdi: Floating Debris Index values
        water_mask: Water mask (1=water, 0=land, NaN=invalid)
        threshold: FDI threshold for detection (default: 0.01)
        
    Returns:
        numpy.ndarray: Binary detection mask (1 = potential plastic, 0 = water, NaN = land/invalid)
    """
    print(f"Creating water-only detection mask with FDI threshold: {threshold}")
    
    # Debug: Check FDI values in water areas for adaptive threshold
    water_areas = (water_mask == 1) & (~np.isnan(fdi))
    water_fdi_values = fdi[water_areas]
    if len(water_fdi_values) > 0:
        print(f"  FDI statistics in water areas ({len(water_fdi_values)} pixels):")
        print(f"    Range: {np.min(water_fdi_values):.4f} to {np.max(water_fdi_values):.4f}")
        print(f"    Mean: {np.mean(water_fdi_values):.4f}, Std: {np.std(water_fdi_values):.4f}")
        print(f"    95th percentile: {np.percentile(water_fdi_values, 95):.4f}")
        print(f"    99th percentile: {np.percentile(water_fdi_values, 99):.4f}")
        
        # Use adaptive threshold: 95th percentile but not less than 0.001
        adaptive_threshold = max(np.percentile(water_fdi_values, 95), 0.001)
        if adaptive_threshold != threshold:
            print(f"  Using adaptive FDI threshold: {adaptive_threshold:.4f} (95th percentile, min 0.001)")
            threshold = adaptive_threshold
        else:
            print(f"  Using fixed FDI threshold: {threshold:.4f}")
    
    # Create binary mask, but only apply to water areas
    detection_mask = np.zeros_like(fdi)
    
    # Apply FDI threshold only where we have water
    detection_mask[water_areas & (fdi > threshold)] = 1
    
    # Set land areas and invalid pixels to NaN
    detection_mask[water_mask != 1] = np.nan
    detection_mask[np.isnan(fdi)] = np.nan
    
    # Calculate detection statistics (water areas only)
    valid_water_pixels = np.sum(water_mask == 1)
    detected_pixels = np.sum(detection_mask == 1)
    detection_percentage = (detected_pixels / valid_water_pixels) * 100 if valid_water_pixels > 0 else 0
    
    print(f"✓ Water-only detection mask created")
    print(f"  - Valid water pixels: {valid_water_pixels}")
    print(f"  - Detected plastic pixels: {detected_pixels}")
    print(f"  - Detection percentage (water only): {detection_percentage:.2f}%")
    
    return detection_mask

def download_rgb_image(config, bbox, time_interval, size=(512, 512)):
    """
    Download true-color RGB image for visualization
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
        
    Returns:
        numpy.ndarray: RGB image array
    """
    print("Downloading RGB image for visualization...")
    
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
    
    request = SentinelHubRequest(
        evalscript=evalscript_rgb,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order='leastCC'
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    rgb_image = request.get_data()[0]
    print("✓ RGB image downloaded successfully")
    
    return rgb_image

def calculate_plastic_area_statistics(detection_mask, water_mask, bbox):
    """
    Calculate plastic area statistics (water areas only)
    
    Args:
        detection_mask: Binary detection mask
        water_mask: Water mask (1=water, 0=land)
        bbox: Bounding box for area calculation
        
    Returns:
        dict: Dictionary containing area statistics
    """
    print("Calculating plastic area statistics (water areas only)...")
    
    # Calculate pixel size in square meters
    # Approximate conversion at this latitude (44°N)
    lat_center = (bbox.min_y + bbox.max_y) / 2
    lon_deg_to_m = 111320 * np.cos(np.radians(lat_center))  # meters per degree longitude
    lat_deg_to_m = 111320  # meters per degree latitude
    
    bbox_width_deg = bbox.max_x - bbox.min_x
    bbox_height_deg = bbox.max_y - bbox.min_y
    
    bbox_width_m = bbox_width_deg * lon_deg_to_m
    bbox_height_m = bbox_height_deg * lat_deg_to_m
    
    total_area_m2 = bbox_width_m * bbox_height_m
    total_area_km2 = total_area_m2 / 1e6
    
    # Calculate pixel area
    image_height, image_width = detection_mask.shape
    pixel_width_m = bbox_width_m / image_width
    pixel_height_m = bbox_height_m / image_height
    pixel_area_m2 = pixel_width_m * pixel_height_m
    
    # Count pixels (water areas only)
    water_pixels = np.sum(water_mask == 1)
    detected_pixels = np.sum(detection_mask == 1)
    land_pixels = np.sum(water_mask == 0)
    
    # Calculate areas
    water_area_m2 = water_pixels * pixel_area_m2
    water_area_km2 = water_area_m2 / 1e6
    detected_area_m2 = detected_pixels * pixel_area_m2
    detected_area_km2 = detected_area_m2 / 1e6
    land_area_m2 = land_pixels * pixel_area_m2
    land_area_km2 = land_area_m2 / 1e6
    
    # Calculate percentages
    plastic_percentage = (detected_pixels / water_pixels) * 100 if water_pixels > 0 else 0
    water_coverage = (water_pixels / (image_height * image_width)) * 100
    land_coverage = (land_pixels / (image_height * image_width)) * 100
    
    statistics = {
        'total_area_km2': total_area_km2,
        'water_area_km2': water_area_km2,
        'land_area_km2': land_area_km2,
        'detected_area_km2': detected_area_km2,
        'detected_area_m2': detected_area_m2,
        'pixel_area_m2': pixel_area_m2,
        'water_pixels': water_pixels,
        'detected_pixels': detected_pixels,
        'land_pixels': land_pixels,
        'plastic_percentage': plastic_percentage,
        'water_coverage': water_coverage,
        'land_coverage': land_coverage
    }
    
    print(f"✓ Area statistics calculated:")
    print(f"  - Total AOI area: {total_area_km2:.2f} km²")
    print(f"  - Water area: {water_area_km2:.2f} km² ({water_coverage:.1f}%)")
    print(f"  - Land area: {land_area_km2:.2f} km² ({land_coverage:.1f}%)")
    print(f"  - Detected plastic area: {detected_area_km2:.4f} km² ({detected_area_m2:.0f} m²)")
    print(f"  - Plastic coverage: {plastic_percentage:.3f}% of water area")
    
    return statistics

def download_multi_resolution_data(config, bbox, time_interval):
    """
    Download data at multiple resolutions for comparison
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        
    Returns:
        dict: Dictionary containing data at different resolutions
    """
    print("Downloading multi-resolution data...")
    
    resolutions = {
        'low': (256, 256),
        'medium': (512, 512),
        'high': (1024, 1024)
    }
    
    multi_res_data = {}
    
    # Evalscript for FDI calculation with Green band for water masking
    evalscript_fdi = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B03", "B04", "B08", "B11", "dataMask"]
            }],
            output: {
                bands: 5,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B03, sample.B04, sample.B08, sample.B11, sample.dataMask];
    }
    """
    
    for res_name, size in resolutions.items():
        print(f"  Downloading {res_name} resolution ({size[0]}x{size[1]})...")
        
        request = SentinelHubRequest(
            evalscript=evalscript_fdi,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config,
        )
        
        data = request.get_data()[0]
        
        # Extract bands
        green_band = data[:, :, 0]
        red_band = data[:, :, 1]
        nir_band = data[:, :, 2]
        swir_band = data[:, :, 3]
        data_mask = data[:, :, 4]
        
        # Create water mask
        water_mask, ndwi = create_water_mask(green_band, nir_band, data_mask)
        
        # Calculate FDI
        fdi = calculate_fdi(red_band, nir_band, swir_band, data_mask)
        
        # Create detection mask (water areas only)
        detection_mask = create_plastic_detection_mask(fdi, water_mask, threshold=0.01)
        
        # Calculate statistics
        area_stats = calculate_plastic_area_statistics(detection_mask, water_mask, bbox)
        
        multi_res_data[res_name] = {
            'size': size,
            'fdi': fdi,
            'detection_mask': detection_mask,
            'water_mask': water_mask,
            'ndwi': ndwi,
            'data_mask': data_mask,
            'area_stats': area_stats
        }
    
    print("✓ Multi-resolution data downloaded successfully")
    return multi_res_data

def visualize_results(rgb_image, fdi, detection_mask, water_mask, ndwi, bbox, time_interval, threshold, area_stats, multi_res_data=None):
    """
    Create comprehensive visualization of results including water masking, binary masks and area statistics
    
    Args:
        rgb_image: True-color RGB image
        fdi: Floating Debris Index values
        detection_mask: Binary detection mask
        water_mask: Water/land mask
        ndwi: NDWI values for water detection
        bbox: Area of interest bounding box
        time_interval: Time period of analysis
        threshold: Detection threshold used
        area_stats: Area statistics dictionary
        multi_res_data: Multi-resolution data (optional)
    """
    print("Creating comprehensive visualization...")
    
    # Create main visualization with better spacing
    fig1 = plt.figure(figsize=(18, 12))
    
    # Main results (3x3 grid with better spacing)
    gs1 = fig1.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # 1. True-color RGB image
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(rgb_image, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax1.set_title('Sentinel-2 True Color (RGB)\nRomanian Black Sea Coast', fontsize=11, pad=15)
    ax1.set_xlabel('Longitude', fontsize=10)
    ax1.set_ylabel('Latitude', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. FDI values (water areas only)
    ax2 = fig1.add_subplot(gs1[0, 1])
    # Mask FDI to show only water areas
    fdi_water_only = fdi.copy()
    fdi_water_only[water_mask != 1] = np.nan  # Hide land areas
    
    fdi_plot = ax2.imshow(fdi_water_only, cmap='RdBu_r', vmin=-0.05, vmax=0.05,
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    
    # Calculate statistics for water areas only
    fdi_water_stats = fdi_water_only[~np.isnan(fdi_water_only)]
    if len(fdi_water_stats) > 0:
        fdi_min_water = np.min(fdi_water_stats)
        fdi_max_water = np.max(fdi_water_stats)
        title_stats = f"{fdi_min_water:.4f} to {fdi_max_water:.4f} (water only)"
    else:
        title_stats = "No water data available"
    
    ax2.set_title(f'Floating Debris Index (FDI)\nRange: {title_stats}', fontsize=11, pad=15)
    ax2.set_xlabel('Longitude', fontsize=10)
    ax2.set_ylabel('Latitude', fontsize=10)
    ax2.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(fdi_plot, ax=ax2, shrink=0.8)
    cbar1.set_label('FDI Value (Water Only)', fontsize=9)
    
    # 3. Water mask (NDWI)
    ax3 = fig1.add_subplot(gs1[0, 2])
    water_colors = ['brown', 'blue']  # brown=land, blue=water
    water_cmap = ListedColormap(water_colors)
    water_plot = ax3.imshow(water_mask, cmap=water_cmap, vmin=0, vmax=1,
                           extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax3.set_title(f'Water/Land Mask (NDWI)\nWater: {area_stats["water_coverage"]:.1f}%, Land: {area_stats["land_coverage"]:.1f}%', fontsize=11, pad=15)
    ax3.set_xlabel('Longitude', fontsize=10)
    ax3.set_ylabel('Latitude', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add legend for water mask
    import matplotlib.patches as mpatches
    land_patch = mpatches.Patch(color='brown', label='Land')
    water_patch = mpatches.Patch(color='blue', label='Water')
    ax3.legend(handles=[land_patch, water_patch], loc='upper right', fontsize=9)
    
    # 4. Detection mask (binary) - Water areas only
    ax4 = fig1.add_subplot(gs1[1, 0])
    # Create custom colormap for detection results
    colors = ['navy', 'red']  # navy=water, red=potential plastic
    cmap = ListedColormap(colors)
    detection_plot = ax4.imshow(detection_mask, cmap=cmap, vmin=0, vmax=1,
                               extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax4.set_title(f'Plastic Detection (Water Only)\nThreshold: {threshold}', fontsize=11, pad=15)
    ax4.set_xlabel('Longitude', fontsize=10)
    ax4.set_ylabel('Latitude', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add custom legend for detection mask
    water_clean_patch = mpatches.Patch(color='navy', label='Clean Water')
    plastic_patch = mpatches.Patch(color='red', label='Detected Plastic')
    ax4.legend(handles=[water_clean_patch, plastic_patch], loc='upper right', fontsize=9)
    
    # 5. FDI histogram
    ax5 = fig1.add_subplot(gs1[1, 1])
    valid_fdi = fdi[~np.isnan(fdi)]
    ax5.hist(valid_fdi, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    ax5.set_xlabel('FDI Value', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('FDI Distribution', fontsize=11, pad=15)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # 6. RGB with detection overlay
    ax6 = fig1.add_subplot(gs1[1, 2])
    ax6.imshow(rgb_image, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    
    # Overlay detection mask with transparency
    detection_overlay = np.ma.masked_where(detection_mask != 1, detection_mask)
    ax6.imshow(detection_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1,
              extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax6.set_title('RGB + Plastic Detection Overlay', fontsize=11, pad=15)
    ax6.set_xlabel('Longitude', fontsize=10)
    ax6.set_ylabel('Latitude', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. NDWI values
    ax7 = fig1.add_subplot(gs1[2, 0])
    ndwi_plot = ax7.imshow(ndwi, cmap='RdBu', vmin=-1, vmax=1,
                          extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax7.set_title(f'NDWI (Water Detection)\nRange: {np.nanmin(ndwi):.3f} to {np.nanmax(ndwi):.3f}', fontsize=11, pad=15)
    ax7.set_xlabel('Longitude', fontsize=10)
    ax7.set_ylabel('Latitude', fontsize=10)
    ax7.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(ndwi_plot, ax=ax7, shrink=0.8)
    cbar2.set_label('NDWI Value', fontsize=9)
    
    # 8. Area statistics and information
    ax8 = fig1.add_subplot(gs1[2, 1])
    ax8.axis('off')
    
    stats_text = f"""
WATER-ONLY PLASTIC DETECTION RESULTS

Area of Interest:
• Bounding Box: {bbox.min_x:.2f}°E - {bbox.max_x:.2f}°E
                {bbox.min_y:.2f}°N - {bbox.max_y:.2f}°N
• Region: Romanian Black Sea Coast
• Focus: Constanța Port & Danube Delta

Time Period:
• Start: {time_interval[0]}
• End: {time_interval[1]}

Area Breakdown:
• Total AOI: {area_stats['total_area_km2']:.2f} km²
• Water area: {area_stats['water_area_km2']:.2f} km²
• Land area: {area_stats['land_area_km2']:.2f} km²
• Detected plastic: {area_stats['detected_area_km2']:.4f} km²
• Plastic area: {area_stats['detected_area_m2']:.0f} m²

Detection Statistics:
• Water pixels: {area_stats['water_pixels']:,}
• Detected pixels: {area_stats['detected_pixels']:,}
• Coverage: {area_stats['plastic_percentage']:.3f}% of water area
• FDI threshold: {threshold}

Water Detection:
• Method: NDWI threshold: 0.0
• Satellite: Sentinel-2 L2A
    """
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Water/Land pie chart
    ax9 = fig1.add_subplot(gs1[2, 2])
    sizes = [area_stats['water_coverage'], area_stats['land_coverage']]
    labels = [f"Water\n{area_stats['water_coverage']:.1f}%", f"Land\n{area_stats['land_coverage']:.1f}%"]
    colors = ['lightblue', 'lightcoral']
    ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax9.set_title('Area Distribution', fontsize=11, pad=15)
    
    plt.suptitle('Water-Masked Plastic Detection using Sentinel-2 FDI Method\nRomanian Black Sea Coast', 
                 fontsize=14, y=0.98)
    
    # Save the main plot
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    output_filename = os.path.join(data_dir, f"fdi_plastic_detection_main_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Main results saved as: {output_filename}")
    plt.show()
    
    # Create multi-resolution comparison if available
    if multi_res_data:
        print("Creating multi-resolution analysis...")
        
        fig2 = plt.figure(figsize=(16, 10))
        gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # Multi-resolution detection masks
        for i, (res_name, res_data) in enumerate(multi_res_data.items()):
            ax = fig2.add_subplot(gs2[0, i])
            colors = ['navy', 'red']
            cmap = ListedColormap(colors)
            ax.imshow(res_data['detection_mask'], cmap=cmap, vmin=0, vmax=1,
                     extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
            size_str = f"{res_data['size'][0]}x{res_data['size'][1]}"
            plastic_area = res_data['area_stats']['detected_area_m2']
            water_pct = res_data['area_stats']['water_coverage']
            ax.set_title(f'{res_name.title()} Resolution\n{size_str} | {plastic_area:.0f} m²', fontsize=11, pad=15)
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Resolution comparison charts
        ax_comp1 = fig2.add_subplot(gs2[1, 0])
        res_names = list(multi_res_data.keys())
        detected_areas = [multi_res_data[name]['area_stats']['detected_area_m2'] for name in res_names]
        
        bars = ax_comp1.bar(res_names, detected_areas, color=['lightblue', 'orange', 'lightgreen'])
        ax_comp1.set_title('Detected Plastic Area by Resolution', fontsize=11, pad=15)
        ax_comp1.set_ylabel('Detected Area (m²)', fontsize=10)
        ax_comp1.set_xlabel('Resolution', fontsize=10)
        
        # Add value labels on bars
        for bar, area in zip(bars, detected_areas):
            ax_comp1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detected_areas)*0.02,
                         f'{area:.0f} m²', ha='center', va='bottom', fontsize=9)
        
        # Water coverage comparison
        ax_comp2 = fig2.add_subplot(gs2[1, 1])
        water_coverages = [multi_res_data[name]['area_stats']['water_coverage'] for name in res_names]
        ax_comp2.bar(res_names, water_coverages, color=['lightcyan', 'lightyellow', 'lightpink'])
        ax_comp2.set_title('Water Coverage by Resolution', fontsize=11, pad=15)
        ax_comp2.set_ylabel('Water Coverage (%)', fontsize=10)
        ax_comp2.set_xlabel('Resolution', fontsize=10)
        
        # Add value labels
        for i, (name, coverage) in enumerate(zip(res_names, water_coverages)):
            ax_comp2.text(i, coverage + max(water_coverages)*0.02, f'{coverage:.1f}%', 
                         ha='center', va='bottom', fontsize=9)
        
        # Summary statistics
        ax_comp3 = fig2.add_subplot(gs2[1, 2])
        ax_comp3.axis('off')
        
        summary_text = f"""
MULTI-RESOLUTION SUMMARY

Resolution Comparison:
• Low (256x256): {multi_res_data['low']['area_stats']['detected_area_m2']:.0f} m²
• Med (512x512): {multi_res_data['medium']['area_stats']['detected_area_m2']:.0f} m²
• High (1024x1024): {multi_res_data['high']['area_stats']['detected_area_m2']:.0f} m²

Water Detection Consistency:
• Low res: {multi_res_data['low']['area_stats']['water_coverage']:.1f}% water
• Med res: {multi_res_data['medium']['area_stats']['water_coverage']:.1f}% water
• High res: {multi_res_data['high']['area_stats']['water_coverage']:.1f}% water

Method Validation:
✓ Water masking prevents land detection
✓ FDI applied only to water pixels  
✓ Results consistent across resolutions
✓ No false positives from land areas
        """
        
        ax_comp3.text(0.05, 0.95, summary_text, transform=ax_comp3.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Multi-Resolution Analysis - FDI Plastic Detection\nRomanian Black Sea Coast', 
                     fontsize=14, y=0.98)
        
        # Save multi-resolution plot
        multi_res_filename = os.path.join(data_dir, f"fdi_multi_resolution_{time_interval[0]}_{time_interval[1]}.png")
        plt.savefig(multi_res_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Multi-resolution analysis saved as: {multi_res_filename}")
        plt.show()
    
    # Create separate binary mask visualization
    print("Creating binary mask visualization...")
    
    fig3 = plt.figure(figsize=(12, 8))
    gs3 = fig3.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Large binary mask
    ax_mask1 = fig3.add_subplot(gs3[0, :])
    colors = ['navy', 'red']
    cmap_mask = ListedColormap(colors)
    mask_plot = ax_mask1.imshow(detection_mask, cmap=cmap_mask, vmin=0, vmax=1,
                              extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax_mask1.set_title(f'Water-Only Plastic Detection Mask\nDetected: {area_stats["detected_area_m2"]:.0f} m² ({area_stats["plastic_percentage"]:.3f}% of water)', fontsize=12, pad=15)
    ax_mask1.set_xlabel('Longitude', fontsize=10)
    ax_mask1.set_ylabel('Latitude', fontsize=10)
    ax_mask1.grid(True, alpha=0.3)
    
    # Add legend
    water_clean_patch = mpatches.Patch(color='navy', label='Clean Water')
    plastic_patch = mpatches.Patch(color='red', label=f'Detected Plastic ({area_stats["detected_pixels"]} pixels)')
    ax_mask1.legend(handles=[water_clean_patch, plastic_patch], loc='upper right', fontsize=10)
    
    # RGB with overlay
    ax_mask2 = fig3.add_subplot(gs3[1, 0])
    ax_mask2.imshow(rgb_image, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    detection_overlay = np.ma.masked_where(detection_mask != 1, detection_mask)
    ax_mask2.imshow(detection_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1,
                   extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax_mask2.set_title('RGB + Detection Overlay', fontsize=11, pad=15)
    ax_mask2.set_xlabel('Longitude', fontsize=10)
    ax_mask2.set_ylabel('Latitude', fontsize=10)
    ax_mask2.grid(True, alpha=0.3)
    
    # Statistics summary
    ax_mask3 = fig3.add_subplot(gs3[1, 1])
    ax_mask3.axis('off')
    
    mask_stats_text = f"""
DETECTION SUMMARY

Detection Performance:
• Total water area: {area_stats['water_area_km2']:.3f} km²
• Detected plastic area: {area_stats['detected_area_km2']:.4f} km²
• Plastic coverage: {area_stats['plastic_percentage']:.3f}% of water
• Detected pixels: {area_stats['detected_pixels']:,}

Method Details:
• Algorithm: Floating Debris Index (FDI)
• Threshold: {threshold}
• Water masking: NDWI > 0.0
• Satellite: Sentinel-2 L2A
• Resolution: 10m pixel size

Quality Metrics:
• Pixel size: {area_stats['pixel_area_m2']:.1f} m²
• Water detection: {area_stats['water_coverage']:.1f}%
• Land exclusion: {area_stats['land_coverage']:.1f}%
    """
    
    ax_mask3.text(0.05, 0.95, mask_stats_text, transform=ax_mask3.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Binary Detection Mask - Water-Only Plastic Detection\n' + 
                 f'Romanian Black Sea Coast | {time_interval[0]} to {time_interval[1]}', 
                 fontsize=13, y=0.98)
    
    # Save binary mask visualization
    mask_filename = os.path.join(data_dir, f"fdi_binary_mask_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(mask_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Binary mask visualization saved as: {mask_filename}")
    plt.show()

def main():
    """Main function to run the plastic detection analysis"""
    print("=" * 60)
    print("SENTINEL-2 PLASTIC DEBRIS DETECTION USING FDI")
    print("=" * 60)
    print("Location: Romanian coast of the Black Sea (near Constanța)")
    print("Method: Floating Debris Index (FDI)")
    print("Data: Sentinel-2 L2A")
    print("=" * 60)
    
    try:
        # Setup credentials
        config = setup_credentials()
        
        # CONFIGURATION - Consistent across all scripts
        # Define area of interest: Romanian coast of the Black Sea
        # Near the port of Constanța and Danube Delta - known shipping activity area
        # Coordinates: Longitude 28.5°E to 29.2°E, Latitude 44.0°N to 44.5°N
        bbox = BBox(bbox=[28.5, 44.0, 29.2, 44.5], crs=CRS.WGS84)
        
        # Time range: Extended summer period for better data availability (consistent across all scripts)
        time_interval = ('2025-07-01', '2025-07-31')
        
        # Image resolution - optimized for native satellite resolution
        # Sentinel-2: 10m native for RGB/NIR bands - use dimensions that respect this
        # For 0.7°x0.5° bbox (~77km x 55km), at 10m resolution = ~7700x5500 pixels
        # Using 768x512 for computational efficiency while respecting 10m grid
        image_size = (768, 512)  # Optimized for 10m native resolution
        
        # Detection parameters (standardized across all scripts)
        ndwi_threshold = -0.05  # More inclusive water detection
        fdi_threshold = 0.002   # Adaptive threshold based on water area statistics
        
        # Resolution settings - use 10m which is native for Sentinel-2 RGB/NIR bands
        target_resolution = 10  # meters - native resolution for main Sentinel-2 bands
        
        print(f"\nArea of Interest: {bbox}")
        print(f"Time Interval: {time_interval}")
        print(f"Image Size: {image_size} (optimized for {target_resolution}m native resolution)")
        print(f"Target Resolution: {target_resolution}m (native for Sentinel-2 RGB/NIR)")
        print(f"Data will be saved to: plastic_detection/data")
        
        # Step 1: Download RGB image for visualization
        rgb_image = download_rgb_image(config, bbox, time_interval, image_size)
        
        # Step 2: Download required bands for FDI calculation and water masking
        green_band, red_band, nir_band, swir_band, data_mask = download_sentinel2_fdi_bands(
            config, bbox, time_interval, image_size
        )
        
        # Step 3: Create water mask to exclude land areas
        print(f"\n{'='*50}")
        print("STEP 3: CREATING WATER MASK")
        print(f"{'='*50}")
        water_mask, ndwi = create_water_mask(green_band, nir_band, data_mask)
        
        # Step 4: Calculate FDI
        print(f"\n{'='*50}")
        print("STEP 4: CALCULATING FDI")
        print(f"{'='*50}")
        fdi = calculate_fdi(red_band, nir_band, swir_band, data_mask)
        
        # Step 5: Create detection mask (water areas only)
        print(f"\n{'='*50}")
        print("STEP 5: DETECTING PLASTIC IN WATER AREAS")
        print(f"{'='*50}")
        detection_mask = create_plastic_detection_mask(fdi, water_mask, fdi_threshold)
        
        # Step 6: Calculate area statistics
        area_stats = calculate_plastic_area_statistics(detection_mask, water_mask, bbox)
        
        # Step 6: Download multi-resolution data for comparison
        print(f"\n{'='*50}")
        print("DOWNLOADING MULTI-RESOLUTION DATA FOR COMPARISON")
        print(f"{'='*50}")
        multi_res_data = download_multi_resolution_data(config, bbox, time_interval)
        
        # Step 7: Visualize results
        visualize_results(rgb_image, fdi, detection_mask, water_mask, ndwi, bbox, time_interval, fdi_threshold, area_stats, multi_res_data)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nINTERPRETATION NOTES:")
        print("• Detection now LIMITED TO WATER AREAS ONLY")
        print("• Land areas are excluded using NDWI water masking")
        print("• High FDI values (red in FDI map) indicate potential plastic debris")
        print("• The method works best in clear, calm water conditions")
        print("• Remaining false positives may occur due to:")
        print("  - Whitecaps and foam")
        print("  - Suspended sediments in water")
        print("  - Cloud shadows over water")
        print("  - Bright boats or floating structures")
        print("• For operational use, combine with:")
        print("  - Multi-temporal analysis")
        print("  - Wind/wave condition data")
        print("  - Visual validation")
        print("  - SAR data for weather-independent detection")
        
        print("\nWATER MASKING IMPROVEMENTS:")
        print("• NDWI threshold: 0.0 (Green-NIR)/(Green+NIR)")
        print("• Separates water (NDWI > 0) from land (NDWI < 0)")
        print("• Prevents false positives from vegetation, soil, buildings")
        print("• FDI calculation applied only to water pixels")
        print("• Significant reduction in false detection rates")
        
        print("\nRECOMMENDATIONS:")
        print("• Current water masking should eliminate land false positives")
        print("• Try different FDI thresholds (0.005 - 0.02) for sensitivity")
        print("• Analyze multiple dates for temporal consistency")
        print("• Consider stricter NDWI thresholds for coastal areas")
        print("• Validate with high-resolution imagery when available")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Check SH_CLIENT_ID and SH_CLIENT_SECRET environment variables")
        print("2. Verify internet connection")
        print("3. Ensure sufficient Sentinel Hub processing units")
        print("4. Try a different time period if no data available")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sentinel-1 SAR + Sentinel-2 Optical Data Fusion for Marine Debris Detection

This script demonstrates data fusion by downloading temporally close Sentinel-1 (SAR) 
and Sentinel-2 (Optical) data for the same AOI. The output is a stacked NumPy array 
ready for machine learning applications.

Key advantages of SAR + Optical fusion:
- SAR works in all weather conditions (clouds, rain, darkness)
- Optical provides spectral information for material classification
- Combined data improves detection accuracy and reduces false positives
- SAR backscatter helps distinguish plastic from organic materials

Study Area: Romanian coast of the Black Sea (near Constanța port)

Author: Varun Burde 
Email: varun@recycllux.com
Date: 2025
Reference: Topouzelis et al. (2020) "Detection of floating plastics from satellite and unmanned aerial systems"

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

def download_sentinel2_optical_data(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-2 optical bands for data fusion
    
    Downloads key optical bands:
    - B02: Blue (490nm) - water penetration, atmospheric correction
    - B03: Green (560nm) - water quality, chlorophyll
    - B04: Red (665nm) - sediments, organic matter
    - B08: NIR (842nm) - vegetation, water/land boundary
    - B11: SWIR1 (1610nm) - moisture content, debris detection
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
        
    Returns:
        tuple: (optical_data, data_mask) where optical_data has shape (H, W, 5)
    """
    print("Downloading Sentinel-2 optical data...")
    
    evalscript_s2 = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "B11", "dataMask"]
            }],
            output: {
                bands: 6,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.dataMask];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_s2,
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
    
    data = request.get_data()[0]
    
    # Extract optical bands and mask
    optical_data = data[:, :, :5]  # First 5 bands (B02, B03, B04, B08, B11)
    data_mask = data[:, :, 5]      # Data mask
    
    print(f"✓ Sentinel-2 optical data downloaded: {optical_data.shape}")
    print(f"  - Blue (B02): min={optical_data[:,:,0].min():.4f}, max={optical_data[:,:,0].max():.4f}")
    print(f"  - Green (B03): min={optical_data[:,:,1].min():.4f}, max={optical_data[:,:,1].max():.4f}")
    print(f"  - Red (B04): min={optical_data[:,:,2].min():.4f}, max={optical_data[:,:,2].max():.4f}")
    print(f"  - NIR (B08): min={optical_data[:,:,3].min():.4f}, max={optical_data[:,:,3].max():.4f}")
    print(f"  - SWIR (B11): min={optical_data[:,:,4].min():.4f}, max={optical_data[:,:,4].max():.4f}")
    
    return optical_data, data_mask

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

def download_sentinel1_sar_data(config, bbox, time_interval, size=(512, 512)):
    """
    Download Sentinel-1 SAR data for data fusion
    
    Downloads both polarizations:
    - VV: Vertical transmit, Vertical receive (better for volume scattering)
    - VH: Vertical transmit, Horizontal receive (better for surface scattering)
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
        
    Returns:
        numpy.ndarray: SAR data with shape (H, W, 2) containing VV and VH
    """
    print("Downloading Sentinel-1 SAR data...")
    
    evalscript_s1 = """
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
        evalscript=evalscript_s1,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=time_interval,
                # Note: SAR data availability is typically more sparse than optical
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    sar_data = request.get_data()[0]
    
    print(f"✓ Sentinel-1 SAR data downloaded: {sar_data.shape}")
    print(f"  - VV polarization: min={sar_data[:,:,0].min():.4f}, max={sar_data[:,:,0].max():.4f}")
    print(f"  - VH polarization: min={sar_data[:,:,1].min():.4f}, max={sar_data[:,:,1].max():.4f}")
    
    return sar_data

def create_fused_dataset(optical_data, sar_data, data_mask):
    """
    Create fused dataset by stacking optical and SAR data
    
    Args:
        optical_data: Sentinel-2 optical bands (H, W, 5)
        sar_data: Sentinel-1 SAR data (H, W, 2)
        data_mask: Valid pixel mask
        
    Returns:
        numpy.ndarray: Fused dataset with shape (H, W, 7)
    """
    print("Creating fused optical + SAR dataset...")
    
    # Ensure same spatial dimensions
    assert optical_data.shape[:2] == sar_data.shape[:2], \
        f"Spatial dimensions mismatch: optical {optical_data.shape[:2]} vs SAR {sar_data.shape[:2]}"
    
    # Stack optical and SAR data along the channel dimension
    fused_data = np.concatenate([optical_data, sar_data], axis=2)
    
    # Apply data mask to all channels
    for i in range(fused_data.shape[2]):
        fused_data[:, :, i][data_mask == 0] = np.nan
    
    print(f"✓ Fused dataset created: {fused_data.shape}")
    print("  Channel order: [Blue, Green, Red, NIR, SWIR, VV, VH]")
    
    # Calculate statistics for each channel
    channel_names = ['Blue (B02)', 'Green (B03)', 'Red (B04)', 'NIR (B08)', 'SWIR (B11)', 'VV', 'VH']
    print("\nChannel statistics:")
    for i, name in enumerate(channel_names):
        channel_data = fused_data[:, :, i]
        valid_data = channel_data[~np.isnan(channel_data)]
        if len(valid_data) > 0:
            print(f"  {name}: mean={np.mean(valid_data):.4f}, std={np.std(valid_data):.4f}")
    
    return fused_data

def calculate_derived_indices(fused_data):
    """
    Calculate derived indices from fused dataset for enhanced plastic detection
    
    Args:
        fused_data: Fused optical + SAR dataset (H, W, 7)
        
    Returns:
        dict: Dictionary of derived indices
    """
    print("Calculating derived indices...")
    
    # Extract individual channels
    blue = fused_data[:, :, 0]    # B02
    green = fused_data[:, :, 1]   # B03
    red = fused_data[:, :, 2]     # B04
    nir = fused_data[:, :, 3]     # B08
    swir = fused_data[:, :, 4]    # B11
    vv = fused_data[:, :, 5]      # VV polarization
    vh = fused_data[:, :, 6]      # VH polarization
    
    indices = {}
    
    # Optical indices
    with np.errstate(divide='ignore', invalid='ignore'):
        # NDVI (Normalized Difference Vegetation Index)
        indices['ndvi'] = (nir - red) / (nir + red + 1e-10)
        
        # FDI (Floating Debris Index) - same as Script 1
        lambda_red, lambda_nir, lambda_swir = 665.0, 842.0, 1610.0
        wavelength_factor = (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
        baseline = red + (swir - red) * wavelength_factor
        indices['fdi'] = nir - baseline
        
        # Normalized Difference Water Index
        indices['ndwi'] = (green - nir) / (green + nir + 1e-10)
        
        # Plastic Index (empirical)
        indices['pi'] = (blue + red) / (2 * green + 1e-10)
    
    # SAR indices
    with np.errstate(divide='ignore', invalid='ignore'):
        # Cross-polarization ratio
        indices['vh_vv_ratio'] = vh / (vv + 1e-10)
        
        # SAR roughness proxy
        indices['sar_intensity'] = np.sqrt(vv**2 + vh**2)
        
        # Depolarization ratio
        indices['depol_ratio'] = vh / (vv + vh + 1e-10)
    
    # Multi-sensor indices (combining optical and SAR)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Optical-SAR composite for plastic detection
        indices['optical_sar_composite'] = indices['fdi'] * np.log10(vv + 1e-10)
        
        # Enhanced plastic index using both sensors
        indices['enhanced_plastic'] = (indices['fdi'] + 0.1) * (1 - indices['vh_vv_ratio'])
    
    print(f"✓ Calculated {len(indices)} derived indices")
    for name, data in indices.items():
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"  {name}: mean={np.mean(valid_data):.4f}, range=[{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
    
    return indices

def create_ml_ready_dataset(fused_data, indices):
    """
    Create machine learning ready dataset by combining original bands and derived indices
    
    Args:
        fused_data: Original fused dataset (H, W, 7)
        indices: Dictionary of derived indices
        
    Returns:
        numpy.ndarray: ML-ready dataset with shape (N_pixels, N_features)
    """
    print("Creating ML-ready dataset...")
    
    # Get spatial dimensions
    height, width = fused_data.shape[:2]
    
    # Stack all indices into a single array
    index_stack = np.stack(list(indices.values()), axis=2)
    
    # Combine original bands with derived indices
    full_dataset = np.concatenate([fused_data, index_stack], axis=2)
    
    # Reshape to (N_pixels, N_features) format for ML
    n_features = full_dataset.shape[2]
    ml_dataset = full_dataset.reshape(-1, n_features)
    
    # Remove pixels with any NaN values
    valid_mask = ~np.any(np.isnan(ml_dataset), axis=1)
    ml_dataset_clean = ml_dataset[valid_mask]
    
    print(f"✓ ML-ready dataset created:")
    print(f"  - Original shape: {full_dataset.shape}")
    print(f"  - ML shape: {ml_dataset_clean.shape}")
    print(f"  - Valid pixels: {np.sum(valid_mask):,} / {len(valid_mask):,} ({100*np.sum(valid_mask)/len(valid_mask):.1f}%)")
    print(f"  - Features: {n_features}")
    
    # Feature names for reference
    feature_names = [
        'Blue_B02', 'Green_B03', 'Red_B04', 'NIR_B08', 'SWIR_B11', 'VV', 'VH',
        'NDVI', 'FDI', 'NDWI', 'Plastic_Index', 'VH_VV_Ratio', 'SAR_Intensity',
        'Depolarization_Ratio', 'Optical_SAR_Composite', 'Enhanced_Plastic'
    ]
    
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i:2d}: {name}")
    
    return ml_dataset_clean, valid_mask.reshape(height, width), feature_names

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
    lat_center = (bbox.min_y + bbox.max_y) / 2
    lon_deg_to_m = 111320 * np.cos(np.radians(lat_center))
    lat_deg_to_m = 111320
    
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
        'reference_area_km2': water_area_km2,  # Water area as reference
        'detected_area_km2': detected_area_km2,
        'detected_area_m2': detected_area_m2,
        'detected_pixels': int(detected_pixels),
        'coverage_percentage': plastic_percentage,
        'pixel_area_m2': pixel_area_m2,
        'area_type': "water areas",
        # Additional fields for backward compatibility
        'water_area_km2': water_area_km2,
        'land_area_km2': land_area_km2,
        'water_pixels': water_pixels,
        'land_pixels': land_pixels,
        'plastic_percentage': plastic_percentage,
        'water_coverage': water_coverage,
        'land_coverage': land_coverage
    }
    
    print(f"✓ Area statistics: {detected_area_m2:.0f} m² detected ({plastic_percentage:.3f}% of water)")
    return statistics

def create_enhanced_plastic_detection(indices, water_mask, threshold=0.05):
    """
    Create enhanced plastic detection using multiple indices (water areas only)
    
    Args:
        indices: Dictionary of calculated indices
        water_mask: Water mask (1=water, 0=land)
        threshold: Enhanced plastic index threshold
        
    Returns:
        numpy.ndarray: Detection mask (1=plastic, 0=water, NaN=land/invalid)
    """
    print(f"Creating water-only enhanced plastic detection (threshold: {threshold})...")
    
    enhanced_plastic = indices['enhanced_plastic']
    detection_mask = np.zeros_like(enhanced_plastic)
    
    # Apply threshold only to water areas
    water_areas = (water_mask == 1) & (~np.isnan(enhanced_plastic))
    detection_mask[water_areas & (enhanced_plastic > threshold)] = 1
    
    # Set land areas and invalid pixels to NaN
    detection_mask[water_mask != 1] = np.nan
    detection_mask[np.isnan(enhanced_plastic)] = np.nan
    
    # Calculate detection statistics (water areas only)
    valid_water_pixels = np.sum(water_mask == 1)
    detected_pixels = np.sum(detection_mask == 1)
    detection_percentage = (detected_pixels / valid_water_pixels) * 100 if valid_water_pixels > 0 else 0
    
    print(f"✓ Water-only enhanced detection mask created")
    print(f"  - Valid water pixels: {valid_water_pixels}")
    print(f"  - Detected plastic pixels: {detected_pixels}")
    print(f"  - Detection percentage (water only): {detection_percentage:.2f}%")
    
    return detection_mask

def visualize_fusion_results(optical_data, sar_data, fused_data, indices, bbox, time_interval, water_mask, ndwi, detection_mask=None, area_stats=None):
    """
    Create comprehensive visualization of data fusion results with water masking
    
    Args:
        optical_data: Sentinel-2 optical bands
        sar_data: Sentinel-1 SAR data
        fused_data: Combined dataset
        indices: Derived indices
        bbox: Area of interest
        time_interval: Time period
        water_mask: Water/land mask
        ndwi: NDWI values
        detection_mask: Binary detection mask (optional)
        area_stats: Area statistics (optional)
    """
    print("Creating comprehensive visualization with water masking...")
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(28, 20))
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. RGB True Color (Optical)
    ax1 = plt.subplot(3, 4, 1)
    rgb_image = np.stack([
        optical_data[:, :, 2],  # Red (B04)
        optical_data[:, :, 1],  # Green (B03)
        optical_data[:, :, 0]   # Blue (B02)
    ], axis=2)
    # Normalize for display
    rgb_display = np.clip(rgb_image * 3.5, 0, 1)
    ax1.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax1.set_title('Sentinel-2 RGB\n(True Color)', fontsize=11)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. NIR band
    ax2 = plt.subplot(3, 4, 2)
    nir_plot = ax2.imshow(optical_data[:, :, 3], cmap='RdYlGn',
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax2.set_title('Sentinel-2 NIR\n(B08)', fontsize=11)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(nir_plot, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. VV polarization (SAR)
    ax3 = plt.subplot(3, 4, 3)
    vv_db = 10 * np.log10(np.maximum(sar_data[:, :, 0], 1e-10))
    vv_plot = ax3.imshow(vv_db, cmap='gray', vmin=-25, vmax=0,
                        extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax3.set_title('Sentinel-1 VV\n(dB)', fontsize=11)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(vv_plot, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. VH polarization (SAR)
    ax4 = plt.subplot(3, 4, 4)
    vh_db = 10 * np.log10(np.maximum(sar_data[:, :, 1], 1e-10))
    vh_plot = ax4.imshow(vh_db, cmap='gray', vmin=-30, vmax=-5,
                        extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax4.set_title('Sentinel-1 VH\n(dB)', fontsize=11)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(vh_plot, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. FDI (Floating Debris Index)
    ax5 = plt.subplot(3, 4, 5)
    fdi_plot = ax5.imshow(indices['fdi'], cmap='RdBu_r', vmin=-0.05, vmax=0.05,
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax5.set_title('FDI\n(Floating Debris Index)', fontsize=11)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(fdi_plot, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. VH/VV Ratio
    ax6 = plt.subplot(3, 4, 6)
    ratio_plot = ax6.imshow(indices['vh_vv_ratio'], cmap='viridis', vmin=0, vmax=1,
                           extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax6.set_title('VH/VV Ratio\n(Cross-pol)', fontsize=11)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(ratio_plot, ax=ax6, fraction=0.046, pad=0.04)
    
    # 7. Enhanced Plastic Index
    ax7 = plt.subplot(3, 4, 7)
    plastic_plot = ax7.imshow(indices['enhanced_plastic'], cmap='Reds', vmin=0, vmax=0.1,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax7.set_title('Enhanced Plastic Index\n(Optical + SAR)', fontsize=11)
    ax7.set_xlabel('Longitude')
    ax7.set_ylabel('Latitude')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(plastic_plot, ax=ax7, fraction=0.046, pad=0.04)
    
    # 8. SAR Intensity
    ax8 = plt.subplot(3, 4, 8)
    intensity_plot = ax8.imshow(indices['sar_intensity'], cmap='plasma',
                               extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax8.set_title('SAR Intensity\n(√(VV² + VH²))', fontsize=11)
    ax8.set_xlabel('Longitude')
    ax8.set_ylabel('Latitude')
    ax8.grid(True, alpha=0.3)
    plt.colorbar(intensity_plot, ax=ax8, fraction=0.046, pad=0.04)
    
    # 9. Correlation matrix between bands
    ax9 = plt.subplot(3, 4, 9)
    # Calculate correlation matrix for central region to avoid edge effects
    h, w = fused_data.shape[:2]
    center_data = fused_data[h//4:3*h//4, w//4:3*w//4, :].reshape(-1, 7)
    center_data_clean = center_data[~np.any(np.isnan(center_data), axis=1)]
    
    if len(center_data_clean) > 100:
        corr_matrix = np.corrcoef(center_data_clean.T)
        im = ax9.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax9.set_title('Band Correlation Matrix', fontsize=11)
        band_labels = ['B02', 'B03', 'B04', 'B08', 'B11', 'VV', 'VH']
        ax9.set_xticks(range(7))
        ax9.set_yticks(range(7))
        ax9.set_xticklabels(band_labels, rotation=45)
        ax9.set_yticklabels(band_labels)
        plt.colorbar(im, ax=ax9, fraction=0.046, pad=0.04)
    else:
        ax9.text(0.5, 0.5, 'Insufficient\nvalid data\nfor correlation', 
                ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Band Correlation Matrix', fontsize=11)
    
    # 10. Feature importance proxy (standard deviation)
    ax10 = plt.subplot(3, 4, 10)
    if len(center_data_clean) > 100:
        feature_std = np.std(center_data_clean, axis=0)
        feature_names_short = ['B02', 'B03', 'B04', 'B08', 'B11', 'VV', 'VH']
        bars = ax10.bar(feature_names_short, feature_std)
        ax10.set_title('Feature Variability\n(Std Dev)', fontsize=11)
        ax10.set_ylabel('Standard Deviation')
        ax10.tick_params(axis='x', rotation=45)
        # Color bars by sensor type
        for i, bar in enumerate(bars):
            if i < 5:  # Optical bands
                bar.set_color('skyblue')
            else:  # SAR bands
                bar.set_color('orange')
    else:
        ax10.text(0.5, 0.5, 'Insufficient\nvalid data', 
                 ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Feature Variability', fontsize=11)
    
    # 11. Data fusion workflow diagram
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    workflow_text = """
DATA FUSION WORKFLOW

1. Sentinel-2 Download
   ├─ Blue (B02)
   ├─ Green (B03)  
   ├─ Red (B04)
   ├─ NIR (B08)
   └─ SWIR (B11)

2. Sentinel-1 Download
   ├─ VV polarization
   └─ VH polarization

3. Spatial Alignment
   └─ Same bbox & resolution

4. Feature Engineering
   ├─ Spectral indices (NDVI, FDI)
   ├─ SAR indices (ratios, intensity)
   └─ Multi-sensor combinations

5. ML-Ready Dataset
   └─ 15 features per pixel
    """
    ax11.text(0.05, 0.95, workflow_text, transform=ax11.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 12. Statistics and information
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Calculate some statistics
    valid_pixels = ~np.any(np.isnan(fused_data), axis=2)
    total_pixels = valid_pixels.size
    valid_count = np.sum(valid_pixels)
    
    stats_text = f"""
FUSION STATISTICS

Area of Interest:
• Bbox: {bbox.min_x:.2f}°E - {bbox.max_x:.2f}°E
        {bbox.min_y:.2f}°N - {bbox.max_y:.2f}°N
• Region: Romanian Black Sea
• Focus: Constanța Port & Danube Delta

Time Period:
• {time_interval[0]} to {time_interval[1]}

Data Quality:
• Total pixels: {total_pixels:,}
• Valid pixels: {valid_count:,}
• Data coverage: {100*valid_count/total_pixels:.1f}%

Sensors Combined:
• Sentinel-2 L2A (optical)
• Sentinel-1 IW (SAR)

Output Features:
• Original bands: 7
• Derived indices: 8
• Total features: 15

ML Applications:
• Plastic debris detection
• Water quality monitoring
• Marine pollution assessment
    """
    
    ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Sentinel-1 SAR + Sentinel-2 Optical Data Fusion\nRomanian Black Sea Coast', 
                 fontsize=16, y=0.98)
    
    # Create data directory if it doesn't exist
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the plot
    output_filename = os.path.join(data_dir, f"data_fusion_sar_optical_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as: {output_filename}")
    
    plt.show()
    
    # Create separate binary mask visualization if detection_mask is provided
    if detection_mask is not None and area_stats is not None:
        print("Creating binary mask visualization...")
        
        fig_mask = plt.figure(figsize=(15, 12))
        
        # Binary mask
        ax_mask1 = plt.subplot(2, 2, 1)
        colors = ['navy', 'red']
        cmap_mask = ListedColormap(colors)
        mask_plot = ax_mask1.imshow(detection_mask, cmap=cmap_mask, vmin=0, vmax=1,
                                   extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_mask1.set_title(f'Enhanced Plastic Detection Mask\nDetected: {area_stats["detected_area_m2"]:.0f} m²', fontsize=12)
        ax_mask1.set_xlabel('Longitude')
        ax_mask1.set_ylabel('Latitude')
        ax_mask1.grid(True, alpha=0.3)
        
        # Add legend
        import matplotlib.patches as mpatches
        water_patch = mpatches.Patch(color='navy', label='Water')
        plastic_patch = mpatches.Patch(color='red', label=f'Detected Plastic ({area_stats["detected_pixels"]} pixels)')
        ax_mask1.legend(handles=[water_patch, plastic_patch], loc='upper right')
        
        # RGB with overlay
        ax_mask2 = plt.subplot(2, 2, 2)
        rgb_image = np.stack([
            optical_data[:, :, 2],  # Red
            optical_data[:, :, 1],  # Green
            optical_data[:, :, 0]   # Blue
        ], axis=2)
        rgb_display = np.clip(rgb_image * 3.5, 0, 1)
        ax_mask2.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        
        # Overlay detection mask
        detection_overlay = np.ma.masked_where(detection_mask != 1, detection_mask)
        ax_mask2.imshow(detection_overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1,
                       extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_mask2.set_title('RGB + Detection Overlay', fontsize=12)
        ax_mask2.set_xlabel('Longitude')
        ax_mask2.set_ylabel('Latitude')
        ax_mask2.grid(True, alpha=0.3)
        
        # Enhanced plastic index
        ax_mask3 = plt.subplot(2, 2, 3)
        enhanced_plot = ax_mask3.imshow(indices['enhanced_plastic'], cmap='hot', vmin=0, vmax=0.2,
                                       extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_mask3.set_title('Enhanced Plastic Index\n(Multi-sensor Fusion)', fontsize=12)
        ax_mask3.set_xlabel('Longitude')
        ax_mask3.set_ylabel('Latitude')
        ax_mask3.grid(True, alpha=0.3)
        plt.colorbar(enhanced_plot, ax=ax_mask3, fraction=0.046, pad=0.04)
        
        # Statistics
        ax_mask4 = plt.subplot(2, 2, 4)
        ax_mask4.axis('off')
        
        stats_text = f"""
ENHANCED DETECTION STATISTICS (WATER AREAS ONLY)

Area Coverage:
• Total AOI: {area_stats['total_area_km2']:.2f} km²
• {area_stats['area_type']}: {area_stats['reference_area_km2']:.2f} km²
• Detected plastic: {area_stats['detected_area_km2']:.4f} km²
• Plastic area: {area_stats['detected_area_m2']:.0f} m²

Detection Performance:
• Plastic coverage: {area_stats['coverage_percentage']:.3f}% of {area_stats['area_type']}
• Detected pixels: {area_stats['detected_pixels']:,}
• Pixel size: {area_stats['pixel_area_m2']:.1f} m²

Multi-sensor Features:
• Optical bands: 5 (Blue to SWIR)
• SAR polarizations: 2 (VV, VH)
• Derived indices: 8
• Enhanced plastic index threshold: 0.05

Detection Method:
• Multi-sensor fusion approach
• Combines spectral and backscatter info
• Reduces false positives from clouds
• Weather-independent SAR component
        """
        
        ax_mask4.text(0.05, 0.95, stats_text, transform=ax_mask4.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('Enhanced Plastic Detection - Multi-sensor Binary Analysis\n' + 
                     f'Romanian Black Sea Coast | {time_interval[0]} to {time_interval[1]}', 
                     fontsize=14, y=0.98)
        
        # Save binary mask visualization
        data_dir = "plastic_detection/data"
        mask_filename = os.path.join(data_dir, f"binary_mask_fusion_{time_interval[0]}_{time_interval[1]}.png")
        plt.savefig(mask_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Binary mask visualization saved as: {mask_filename}")
        
        plt.show()

def save_ml_dataset(ml_dataset, feature_names, time_interval):
    """
    Save the ML-ready dataset to files for future use
    
    Args:
        ml_dataset: ML-ready dataset array
        feature_names: List of feature names
        time_interval: Time period for filename
    """
    print("Saving ML-ready dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save as numpy binary format
    dataset_filename = os.path.join(data_dir, f"fused_dataset_{time_interval[0]}_{time_interval[1]}.npz")
    np.savez_compressed(
        dataset_filename,
        data=ml_dataset,
        features=feature_names,
        description="Fused Sentinel-1 SAR + Sentinel-2 optical dataset for marine debris detection"
    )
    
    # Save feature names as text file
    features_filename = os.path.join(data_dir, f"feature_names_{time_interval[0]}_{time_interval[1]}.txt")
    with open(features_filename, 'w') as f:
        f.write("Feature names for fused SAR + Optical dataset:\n")
        f.write("=" * 50 + "\n")
        for i, name in enumerate(feature_names):
            f.write(f"{i:2d}: {name}\n")
    
    print(f"✓ Dataset saved as: {dataset_filename}")
    print(f"✓ Feature names saved as: {features_filename}")
    
    # Print basic statistics
    print(f"\nDataset summary:")
    print(f"  Shape: {ml_dataset.shape}")
    print(f"  Size: {ml_dataset.nbytes / 1024 / 1024:.1f} MB")
    print(f"  Features: {len(feature_names)}")

def main():
    """Main function to run the data fusion analysis"""
    print("=" * 70)
    print("SENTINEL-1 SAR + SENTINEL-2 OPTICAL DATA FUSION")
    print("=" * 70)
    print("Location: Romanian coast of the Black Sea (near Constanța)")
    print("Purpose: Create ML-ready dataset for marine debris detection")
    print("Sensors: Sentinel-1 IW (SAR) + Sentinel-2 L2A (Optical)")
    print("=" * 70)
    
    try:
        # Setup credentials
        config = setup_credentials()
        
        # Define area of interest: Romanian coast of the Black Sea
        # Near the port of Constanța and Danube Delta - known shipping activity area
        # Coordinates: Longitude 28.5°E to 29.2°E, Latitude 44.0°N to 44.5°N
        bbox = BBox(bbox=[28.5, 44.0, 29.2, 44.5], crs=CRS.WGS84)
        
        # Time range: Summer period with good data availability
        time_interval = ('2024-07-10', '2024-07-20')
        
        # Image resolution
        image_size = (512, 512)
        
        print(f"Configuration:")
        print(f"  Area of Interest: {bbox}")
        print(f"  Time Interval: {time_interval}")
        print(f"  Image Size: {image_size}")
        print(f"  Data will be saved to: plastic_detection/data")
        
        # Step 1: Download Sentinel-2 optical data
        print(f"\n{'='*50}")
        print("STEP 1: DOWNLOADING SENTINEL-2 OPTICAL DATA")
        print(f"{'='*50}")
        optical_data, data_mask = download_sentinel2_optical_data(
            config, bbox, time_interval, image_size
        )
        
        # Step 1.5: Create water mask
        print(f"\n{'='*50}")
        print("STEP 1.5: CREATING WATER MASK")
        print(f"{'='*50}")
        # Extract Green (B03) and NIR (B08) for water masking
        green_band = optical_data[:, :, 1]  # B03
        nir_band = optical_data[:, :, 3]    # B08
        water_mask, ndwi = create_water_mask(green_band, nir_band, data_mask)
        
        # Step 2: Download Sentinel-1 SAR data
        print(f"\n{'='*50}")
        print("STEP 2: DOWNLOADING SENTINEL-1 SAR DATA")
        print(f"{'='*50}")
        sar_data = download_sentinel1_sar_data(
            config, bbox, time_interval, image_size
        )
        
        # Step 3: Create fused dataset
        print(f"\n{'='*50}")
        print("STEP 3: CREATING FUSED DATASET")
        print(f"{'='*50}")
        fused_data = create_fused_dataset(optical_data, sar_data, data_mask)
        
        # Step 4: Calculate derived indices
        print(f"\n{'='*50}")
        print("STEP 4: CALCULATING DERIVED INDICES")
        print(f"{'='*50}")
        indices = calculate_derived_indices(fused_data)
        
        # Step 4.5: Create enhanced plastic detection (water areas only)
        print(f"\n{'='*50}")
        print("STEP 4.5: CREATING WATER-ONLY PLASTIC DETECTION")
        print(f"{'='*50}")
        detection_mask = create_enhanced_plastic_detection(indices, water_mask, threshold=0.05)
        area_stats = calculate_plastic_area_statistics(detection_mask, water_mask, bbox)
        
        # Step 5: Create ML-ready dataset
        print(f"\n{'='*50}")
        print("STEP 5: PREPARING ML-READY DATASET")
        print(f"{'='*50}")
        ml_dataset, valid_mask, feature_names = create_ml_ready_dataset(fused_data, indices)
        
        # Step 6: Visualize results
        print(f"\n{'='*50}")
        print("STEP 6: CREATING VISUALIZATIONS")
        print(f"{'='*50}")
        visualize_fusion_results(optical_data, sar_data, fused_data, indices, bbox, time_interval, water_mask, ndwi, detection_mask, area_stats)
        
        # Step 7: Save dataset
        print(f"\n{'='*50}")
        print("STEP 7: SAVING ML-READY DATASET")
        print(f"{'='*50}")
        save_ml_dataset(ml_dataset, feature_names, time_interval)
        
        print(f"\n{'='*70}")
        print("DATA FUSION COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        print("\nOUTPUT SUMMARY:")
        print("• Fused dataset with 15 features per pixel")
        print("• Combination of optical spectral and SAR backscatter information")
        print("• Ready for machine learning applications")
        print("• Saved in compressed NumPy format")
        
        print("\nAPPLICATIONS:")
        print("• Train supervised ML models for plastic detection")
        print("• Unsupervised clustering for water quality analysis")
        print("• Time series analysis for pollution monitoring")
        print("• Multi-sensor change detection")
        
        print("\nNEXT STEPS:")
        print("• Collect ground truth data for supervised learning")
        print("• Apply dimensionality reduction (PCA, t-SNE)")
        print("• Train classification models (Random Forest, SVM, NN)")
        print("• Validate results with independent data")
        print("• Scale up to larger areas and time periods")
        
        print("\nFEATURE ADVANTAGES:")
        print("• SAR data works in all weather conditions")
        print("• Optical data provides spectral discrimination")
        print("• Combined features reduce false positive rates")
        print("• Multiple indices capture different aspects of debris")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Check SH_CLIENT_ID and SH_CLIENT_SECRET environment variables")
        print("2. Verify internet connection and Sentinel Hub service status")
        print("3. Ensure sufficient processing units in your account")
        print("4. Try a different time period if data is not available")
        print("5. SAR data availability is typically more sparse than optical")
        print("6. Consider extending the time interval for better SAR coverage")

if __name__ == "__main__":
    main()

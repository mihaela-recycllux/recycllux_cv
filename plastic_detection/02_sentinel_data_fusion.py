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
    
    # Validate SAR data quality
    vv_data = sar_data[:, :, 0]
    vh_data = sar_data[:, :, 1]
    
    # Check for common SAR data issues
    vv_valid = np.sum(vv_data > 1e-6)
    vh_valid = np.sum(vh_data > 1e-6)
    total_pixels = vv_data.size
    
    vv_coverage = (vv_valid / total_pixels) * 100
    vh_coverage = (vh_valid / total_pixels) * 100
    
    print(f"  - VV polarization: min={np.nanmin(vv_data):.6f}, max={np.nanmax(vv_data):.6f}, coverage={vv_coverage:.1f}%")
    print(f"  - VH polarization: min={np.nanmin(vh_data):.6f}, max={np.nanmax(vh_data):.6f}, coverage={vh_coverage:.1f}%")
    
    # Warning for poor data quality
    if vv_coverage < 50 or vh_coverage < 50:
        print(f"  ⚠️  WARNING: Low SAR data coverage detected!")
        print(f"     This may be due to:")
        print(f"     - No SAR acquisitions in the specified time period")
        print(f"     - Different temporal sampling between SAR and optical data")
        print(f"     - Processing issues in the SAR data")
        print(f"     Consider extending the time interval or checking data availability")
    
    # Check for unusual patterns that might indicate artifacts
    if np.all(vv_data == vv_data.flat[0]) or np.all(vh_data == vh_data.flat[0]):
        print(f"  ⚠️  WARNING: Constant SAR values detected - possible data issue!")
    
    # Check for reasonable SAR value ranges (typical for Sentinel-1)
    if np.nanmax(vv_data) > 10 or np.nanmax(vh_data) > 10:
        print(f"  ⚠️  WARNING: Unusually high SAR values detected - possible calibration issue!")
    
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
    
    # Debug: Print basic band statistics
    print(f"  Input data statistics:")
    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR', 'VV', 'VH']
    for i, name in enumerate(band_names):
        data = fused_data[:, :, i]
        valid_data = data[data > 0] if name in ['VV', 'VH'] else data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"    {name}: mean={np.mean(valid_data):.4f}, std={np.std(valid_data):.4f}, range=[{np.min(valid_data):.4f}, {np.max(valid_data):.4f}]")
    
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
        # Cross-polarization ratio - handle invalid values properly
        indices['vh_vv_ratio'] = np.where((vv > 1e-6) & (vh > 1e-6), vh / vv, np.nan)
        
        # SAR roughness proxy - only calculate for valid values
        indices['sar_intensity'] = np.where((vv > 1e-6) & (vh > 1e-6), np.sqrt(vv**2 + vh**2), np.nan)
        
        # Depolarization ratio - handle invalid values
        indices['depol_ratio'] = np.where((vv > 1e-6) & (vh > 1e-6), vh / (vv + vh), np.nan)
    
    # Multi-sensor indices (combining optical and SAR)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Optical-SAR composite for plastic detection - handle invalid values
        indices['optical_sar_composite'] = np.where(vv > 1e-6, 
                                                   indices['fdi'] * np.log10(vv), 
                                                   np.nan)
        
        # Enhanced plastic index using both sensors (FIXED FORMULA)
        # Plastic detection based on:
        # 1. Positive FDI (floating objects reflect more red than NIR)
        # 2. Low depolarization (smooth plastic surfaces)
        # 3. Moderate backscatter intensity
        
        # Clip VH/VV ratio to reasonable range (0-2) to avoid extreme values
        vh_vv_clipped = np.clip(indices['vh_vv_ratio'], 0, 2)
        
        # Surface smoothness indicator (higher for smoother surfaces like plastic)
        smoothness = np.where(~np.isnan(vh_vv_clipped), 
                             1 / (1 + vh_vv_clipped),  # Decreases with increasing vh/vv ratio
                             0)
        
        # Combine positive FDI with surface smoothness
        # Scale FDI to positive range [0, 1] for combination
        fdi_scaled = np.clip((indices['fdi'] + 0.1) / 0.2, 0, 1)  # Shift and scale FDI
        
        indices['enhanced_plastic'] = fdi_scaled * smoothness
        
        # Debug the enhanced plastic calculation
        print(f"  Enhanced plastic debug:")
        vh_vv_valid = vh_vv_clipped[~np.isnan(vh_vv_clipped)]
        smoothness_valid = smoothness[~np.isnan(smoothness)]
        fdi_scaled_valid = fdi_scaled[~np.isnan(fdi_scaled)]
        enhanced_valid = indices['enhanced_plastic'][~np.isnan(indices['enhanced_plastic'])]
        
        if len(vh_vv_valid) > 0:
            print(f"    VH/VV ratio (clipped): mean={np.mean(vh_vv_valid):.4f}, range=[{np.min(vh_vv_valid):.4f}, {np.max(vh_vv_valid):.4f}]")
        if len(smoothness_valid) > 0:
            print(f"    Smoothness factor: mean={np.mean(smoothness_valid):.4f}, range=[{np.min(smoothness_valid):.4f}, {np.max(smoothness_valid):.4f}]")
        if len(fdi_scaled_valid) > 0:
            print(f"    FDI scaled: mean={np.mean(fdi_scaled_valid):.4f}, range=[{np.min(fdi_scaled_valid):.4f}, {np.max(fdi_scaled_valid):.4f}]")
        if len(enhanced_valid) > 0:
            print(f"    Enhanced plastic final: mean={np.mean(enhanced_valid):.4f}, range=[{np.min(enhanced_valid):.4f}, {np.max(enhanced_valid):.4f}]")
    
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

def create_enhanced_plastic_detection(indices, water_mask, threshold=0.01):
    """
    Create enhanced plastic detection using multiple indices (water areas only)
    
    Args:
        indices: Dictionary of calculated indices
        water_mask: Water mask (1=water, 0=land)
        threshold: Enhanced plastic index threshold (lowered to 0.01)
        
    Returns:
        numpy.ndarray: Detection mask (1=plastic, 0=water, NaN=land/invalid)
    """
    print(f"Creating water-only enhanced plastic detection (threshold: {threshold})...")
    
    enhanced_plastic = indices['enhanced_plastic']
    
    # Calculate adaptive threshold based on water area statistics
    water_areas = (water_mask == 1) & (~np.isnan(enhanced_plastic))
    water_enhanced_values = enhanced_plastic[water_areas]
    if len(water_enhanced_values) > 0:
        print(f"  Enhanced plastic in water areas ({len(water_enhanced_values)} pixels):")
        print(f"    Range: {np.min(water_enhanced_values):.4f} to {np.max(water_enhanced_values):.4f}")
        print(f"    Mean: {np.mean(water_enhanced_values):.4f}, Std: {np.std(water_enhanced_values):.4f}")
        print(f"    Percentiles: 50th={np.percentile(water_enhanced_values, 50):.4f}, 90th={np.percentile(water_enhanced_values, 90):.4f}")
        print(f"    Percentiles: 95th={np.percentile(water_enhanced_values, 95):.4f}, 99th={np.percentile(water_enhanced_values, 99):.4f}")
        
        # Count pixels above different thresholds
        print(f"    Pixels above thresholds:")
        for thresh in [0.0001, 0.001, 0.01, 0.05, 0.1]:
            count = np.sum(water_enhanced_values > thresh)
            pct = (count / len(water_enhanced_values)) * 100
            print(f"      > {thresh}: {count} pixels ({pct:.2f}%)")
        
        # Use adaptive threshold: use 95th percentile but with minimum threshold
        adaptive_threshold = max(np.percentile(water_enhanced_values, 95), 0.001)
        if adaptive_threshold != threshold:
            print(f"  Using adaptive threshold: {adaptive_threshold:.4f} (95th percentile, min 0.001)")
        else:
            print(f"  Using fixed threshold: {threshold:.4f}")
        threshold = adaptive_threshold
    
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
    
    # Create main visualization figure with better spacing
    fig1 = plt.figure(figsize=(18, 12))
    gs1 = fig1.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # 1. RGB True Color (Optical)
    ax1 = fig1.add_subplot(gs1[0, 0])
    rgb_image = np.stack([
        optical_data[:, :, 2],  # Red (B04)
        optical_data[:, :, 1],  # Green (B03)
        optical_data[:, :, 0]   # Blue (B02)
    ], axis=2)
    # Normalize for display
    rgb_display = np.clip(rgb_image * 3.5, 0, 1)
    ax1.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax1.set_title('Sentinel-2 RGB\n(True Color)', fontsize=11, pad=15)
    ax1.set_xlabel('Longitude', fontsize=10)
    ax1.set_ylabel('Latitude', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. NIR band
    ax2 = fig1.add_subplot(gs1[0, 1])
    nir_plot = ax2.imshow(optical_data[:, :, 3], cmap='RdYlGn',
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax2.set_title('Sentinel-2 NIR\n(B08)', fontsize=11, pad=15)
    ax2.set_xlabel('Longitude', fontsize=10)
    ax2.set_ylabel('Latitude', fontsize=10)
    ax2.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(nir_plot, ax=ax2, shrink=0.8)
    cbar1.set_label('Reflectance', fontsize=9)
    
    # 3. VV polarization (SAR)
    ax3 = fig1.add_subplot(gs1[0, 2])
    # Check for valid SAR data before processing
    vv_data = sar_data[:, :, 0]
    if np.all(vv_data <= 0) or np.all(np.isnan(vv_data)):
        # No valid SAR data - show warning
        ax3.text(0.5, 0.5, 'No Valid\nSAR VV Data\nAvailable', 
                ha='center', va='center', transform=ax3.transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax3.set_title('Sentinel-1 VV\n(No Data)', fontsize=11, pad=15)
    else:
        # Filter out invalid values and convert to dB with robust outlier removal
        vv_valid = np.where(vv_data > 1e-10, vv_data, np.nan)
        
        # Remove extreme outliers using IQR method
        vv_clean = vv_valid[~np.isnan(vv_valid)]
        if len(vv_clean) > 0:
            q25, q75 = np.percentile(vv_clean, [25, 75])
            iqr = q75 - q25
            lower_bound = max(q25 - 1.5 * iqr, 1e-6)  # Don't go below reasonable SAR minimum
            upper_bound = min(q75 + 1.5 * iqr, 1.0)   # Don't go above reasonable SAR maximum
            vv_clipped = np.where((vv_valid >= lower_bound) & (vv_valid <= upper_bound), vv_valid, np.nan)
        else:
            vv_clipped = vv_valid
        
        # Convert to dB
        vv_db = 10 * np.log10(vv_clipped)
        
        # Debug: Print SAR statistics
        vv_stats = vv_db[~np.isnan(vv_db)]
        if len(vv_stats) > 0:
            print(f"VV dB range: {np.min(vv_stats):.1f} to {np.max(vv_stats):.1f} dB")
            print(f"VV dB mean: {np.mean(vv_stats):.1f} dB, std: {np.std(vv_stats):.1f} dB")
        
        # Use data-driven dB range for better visualization
        vv_stats = vv_db[~np.isnan(vv_db)]
        if len(vv_stats) > 0:
            vmin_vv = np.percentile(vv_stats, 2)   # 2nd percentile for better contrast
            vmax_vv = np.percentile(vv_stats, 98)  # 98th percentile for better contrast
            # Ensure reasonable bounds
            vmin_vv = max(vmin_vv, -60)  # Don't go below -60 dB
            vmax_vv = min(vmax_vv, -5)   # Don't go above -5 dB
        else:
            vmin_vv, vmax_vv = -30, -10
        vv_plot = ax3.imshow(vv_db, cmap='viridis', vmin=vmin_vv, vmax=vmax_vv,
                            extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax3.set_title(f'Sentinel-1 VV\n(dB: {vmin_vv:.1f} to {vmax_vv:.1f})', fontsize=11, pad=15)
        cbar2 = plt.colorbar(vv_plot, ax=ax3, shrink=0.8)
        cbar2.set_label('VV (dB)', fontsize=9)
    ax3.set_xlabel('Longitude', fontsize=10)
    ax3.set_ylabel('Latitude', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. VH polarization (SAR)
    ax4 = fig1.add_subplot(gs1[0, 3])
    # Check for valid SAR data before processing
    vh_data = sar_data[:, :, 1]
    if np.all(vh_data <= 0) or np.all(np.isnan(vh_data)):
        # No valid SAR data - show warning
        ax4.text(0.5, 0.5, 'No Valid\nSAR VH Data\nAvailable', 
                ha='center', va='center', transform=ax4.transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax4.set_title('Sentinel-1 VH\n(No Data)', fontsize=11, pad=15)
    else:
        # Filter out invalid values and convert to dB with robust outlier removal
        vh_valid = np.where(vh_data > 1e-10, vh_data, np.nan)
        
        # Remove extreme outliers using IQR method
        vh_clean = vh_valid[~np.isnan(vh_valid)]
        if len(vh_clean) > 0:
            q25, q75 = np.percentile(vh_clean, [25, 75])
            iqr = q75 - q25
            lower_bound = max(q25 - 1.5 * iqr, 1e-6)  # Don't go below reasonable SAR minimum
            upper_bound = min(q75 + 1.5 * iqr, 1.0)   # Don't go above reasonable SAR maximum
            vh_clipped = np.where((vh_valid >= lower_bound) & (vh_valid <= upper_bound), vh_valid, np.nan)
        else:
            vh_clipped = vh_valid
        
        # Convert to dB
        vh_db = 10 * np.log10(vh_clipped)
        
        # Debug: Print SAR statistics
        vh_stats = vh_db[~np.isnan(vh_db)]
        if len(vh_stats) > 0:
            print(f"VH dB range: {np.min(vh_stats):.1f} to {np.max(vh_stats):.1f} dB")
            print(f"VH dB mean: {np.mean(vh_stats):.1f} dB, std: {np.std(vh_stats):.1f} dB")
        
        # Use data-driven dB range for better visualization (VH typically lower than VV)
        vh_stats = vh_db[~np.isnan(vh_db)]
        if len(vh_stats) > 0:
            vmin_vh = np.percentile(vh_stats, 2)   # 2nd percentile for better contrast
            vmax_vh = np.percentile(vh_stats, 98)  # 98th percentile for better contrast
            # Ensure reasonable bounds
            vmin_vh = max(vmin_vh, -65)  # Don't go below -65 dB
            vmax_vh = min(vmax_vh, -10)  # Don't go above -10 dB
        else:
            vmin_vh, vmax_vh = -35, -15
        vh_plot = ax4.imshow(vh_db, cmap='plasma', vmin=vmin_vh, vmax=vmax_vh,
                            extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax4.set_title(f'Sentinel-1 VH\n(dB: {vmin_vh:.1f} to {vmax_vh:.1f})', fontsize=11, pad=15)
        cbar3 = plt.colorbar(vh_plot, ax=ax4, shrink=0.8)
        cbar3.set_label('VH (dB)', fontsize=9)
    ax4.set_xlabel('Longitude', fontsize=10)
    ax4.set_ylabel('Latitude', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. FDI (Floating Debris Index) - water areas only
    ax5 = fig1.add_subplot(gs1[1, 0])
    # Mask FDI to show only water areas
    fdi_water_only = indices['fdi'].copy()
    fdi_water_only[water_mask != 1] = np.nan  # Hide land areas
    
    fdi_plot = ax5.imshow(fdi_water_only, cmap='RdBu_r', vmin=-0.05, vmax=0.05,
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax5.set_title('FDI (Water Areas Only)\n(Floating Debris Index)', fontsize=11, pad=15)
    ax5.set_xlabel('Longitude', fontsize=10)
    ax5.set_ylabel('Latitude', fontsize=10)
    ax5.grid(True, alpha=0.3)
    cbar4 = plt.colorbar(fdi_plot, ax=ax5, shrink=0.8)
    cbar4.set_label('FDI Value (Water Only)', fontsize=9)
    
    # 6. VH/VV Ratio
    ax6 = fig1.add_subplot(gs1[1, 1])
    # Check for valid ratio calculation
    vv_data = sar_data[:, :, 0]
    vh_data = sar_data[:, :, 1]
    if np.all(vv_data <= 0) or np.all(vh_data <= 0) or np.all(np.isnan(indices['vh_vv_ratio'])):
        # No valid SAR data for ratio calculation
        ax6.text(0.5, 0.5, 'No Valid\nSAR Data\nfor Ratio', 
                ha='center', va='center', transform=ax6.transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        ax6.set_title('VH/VV Ratio\n(No Data)', fontsize=11, pad=15)
    else:
        # Use adaptive scaling for ratio
        ratio_valid = indices['vh_vv_ratio'][~np.isnan(indices['vh_vv_ratio'])]
        if len(ratio_valid) > 0:
            vmin_ratio = np.percentile(ratio_valid, 1)
            vmax_ratio = np.percentile(ratio_valid, 99)
        else:
            vmin_ratio, vmax_ratio = 0, 1
        ratio_plot = ax6.imshow(indices['vh_vv_ratio'], cmap='viridis', 
                               vmin=vmin_ratio, vmax=vmax_ratio,
                               extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax6.set_title(f'VH/VV Ratio\n({vmin_ratio:.2f} to {vmax_ratio:.2f})', fontsize=11, pad=15)
        cbar5 = plt.colorbar(ratio_plot, ax=ax6, shrink=0.8)
        cbar5.set_label('Ratio', fontsize=9)
    ax6.set_xlabel('Longitude', fontsize=10)
    ax6.set_ylabel('Latitude', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # 7. Enhanced Plastic Index
    ax7 = fig1.add_subplot(gs1[1, 2])
    plastic_plot = ax7.imshow(indices['enhanced_plastic'], cmap='Reds', vmin=0, vmax=0.1,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax7.set_title('Enhanced Plastic Index\n(Optical + SAR)', fontsize=11, pad=15)
    ax7.set_xlabel('Longitude', fontsize=10)
    ax7.set_ylabel('Latitude', fontsize=10)
    ax7.grid(True, alpha=0.3)
    cbar6 = plt.colorbar(plastic_plot, ax=ax7, shrink=0.8)
    cbar6.set_label('Enhanced Plastic', fontsize=9)
    
    # 8. SAR Intensity
    ax8 = fig1.add_subplot(gs1[1, 3])
    # Scale SAR intensity for better visualization
    sar_valid = indices['sar_intensity'][~np.isnan(indices['sar_intensity'])]
    if len(sar_valid) > 0:
        vmin_sar = np.percentile(sar_valid, 2)
        vmax_sar = np.percentile(sar_valid, 98)
    else:
        vmin_sar, vmax_sar = 0, 1
    intensity_plot = ax8.imshow(indices['sar_intensity'], cmap='plasma', 
                               vmin=vmin_sar, vmax=vmax_sar,
                               extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax8.set_title('SAR Intensity\n(√(VV² + VH²))', fontsize=11, pad=15)
    ax8.set_xlabel('Longitude', fontsize=10)
    ax8.set_ylabel('Latitude', fontsize=10)
    ax8.grid(True, alpha=0.3)
    cbar7 = plt.colorbar(intensity_plot, ax=ax8, shrink=0.8)
    cbar7.set_label('Intensity', fontsize=9)
    
    # 9. Band correlation matrix
    ax9 = fig1.add_subplot(gs1[2, 0])
    # Calculate correlation matrix for central region to avoid edge effects
    h, w = fused_data.shape[:2]
    center_data = fused_data[h//4:3*h//4, w//4:3*w//4, :].reshape(-1, 7)
    center_data_clean = center_data[~np.any(np.isnan(center_data), axis=1)]
    
    if len(center_data_clean) > 100:
        corr_matrix = np.corrcoef(center_data_clean.T)
        im = ax9.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax9.set_title('Band Correlation Matrix', fontsize=11, pad=15)
        band_labels = ['B02', 'B03', 'B04', 'B08', 'B11', 'VV', 'VH']
        ax9.set_xticks(range(7))
        ax9.set_yticks(range(7))
        ax9.set_xticklabels(band_labels, rotation=45, fontsize=9)
        ax9.set_yticklabels(band_labels, fontsize=9)
        cbar8 = plt.colorbar(im, ax=ax9, shrink=0.8)
        cbar8.set_label('Correlation', fontsize=9)
    else:
        ax9.text(0.5, 0.5, 'Insufficient\nvalid data\nfor correlation', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=10)
        ax9.set_title('Band Correlation Matrix', fontsize=11, pad=15)
    
    # 10. Feature importance proxy (standard deviation)
    ax10 = fig1.add_subplot(gs1[2, 1])
    if len(center_data_clean) > 100:
        feature_std = np.std(center_data_clean, axis=0)
        feature_names_short = ['B02', 'B03', 'B04', 'B08', 'B11', 'VV', 'VH']
        bars = ax10.bar(feature_names_short, feature_std)
        ax10.set_title('Feature Variability\n(Std Dev)', fontsize=11, pad=15)
        ax10.set_ylabel('Standard Deviation', fontsize=10)
        ax10.tick_params(axis='x', rotation=45, labelsize=9)
        # Color bars by sensor type
        for i, bar in enumerate(bars):
            if i < 5:  # Optical bands
                bar.set_color('skyblue')
            else:  # SAR bands
                bar.set_color('orange')
    else:
        ax10.text(0.5, 0.5, 'Insufficient\nvalid data', 
                 ha='center', va='center', transform=ax10.transAxes, fontsize=10)
        ax10.set_title('Feature Variability', fontsize=11, pad=15)
    
    # 11. Water mask visualization
    ax11 = fig1.add_subplot(gs1[2, 2])
    water_colors = ['brown', 'blue']  # brown=land, blue=water
    water_cmap = ListedColormap(water_colors)
    water_plot = ax11.imshow(water_mask, cmap=water_cmap, vmin=0, vmax=1,
                           extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax11.set_title('Water/Land Mask\n(NDWI Based)', fontsize=11, pad=15)
    ax11.set_xlabel('Longitude', fontsize=10)
    ax11.set_ylabel('Latitude', fontsize=10)
    ax11.grid(True, alpha=0.3)
    
    # Add legend for water mask
    import matplotlib.patches as mpatches
    land_patch = mpatches.Patch(color='brown', label='Land')
    water_patch = mpatches.Patch(color='blue', label='Water')
    ax11.legend(handles=[land_patch, water_patch], loc='upper right', fontsize=9)
    
    # 12. Statistics and information
    ax12 = fig1.add_subplot(gs1[2, 3])
    ax12.axis('off')
    
    # Calculate some statistics
    valid_pixels = ~np.any(np.isnan(fused_data), axis=2)
    total_pixels = valid_pixels.size
    valid_count = np.sum(valid_pixels)
    water_coverage = np.sum(water_mask == 1) / total_pixels * 100
    
    stats_text = f"""FUSION STATISTICS

Area of Interest:
• Bbox: {bbox.min_x:.2f}°E - {bbox.max_x:.2f}°E
        {bbox.min_y:.2f}°N - {bbox.max_y:.2f}°N
• Region: Romanian Black Sea

Time Period: {time_interval[0]} to {time_interval[1]}

Data Quality:
• Total pixels: {total_pixels:,}
• Valid pixels: {valid_count:,}
• Coverage: {100*valid_count/total_pixels:.1f}%
• Water: {water_coverage:.1f}%

Sensors:
• Sentinel-2 L2A (optical)
• Sentinel-1 IW (SAR)

Features:
• Original bands: 7
• Derived indices: 8
• Total features: 15

Applications:
• Plastic debris detection
• Water quality monitoring
• Marine pollution assessment"""
    
    ax12.text(0.02, 0.98, stats_text, transform=ax12.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.3))
    
    plt.suptitle('Sentinel-1 SAR + Sentinel-2 Optical Data Fusion\nRomanian Black Sea Coast', 
                 fontsize=14, y=0.98)
    
    # Save the main plot
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    output_filename = os.path.join(data_dir, f"data_fusion_main_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Main fusion visualization saved as: {output_filename}")
    plt.show()
    
    # Create separate binary mask visualization if detection_mask is provided
    if detection_mask is not None and area_stats is not None:
        print("Creating enhanced detection visualization...")
        
        fig2 = plt.figure(figsize=(16, 10))
        gs2 = fig2.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        # Binary mask
        ax_mask1 = fig2.add_subplot(gs2[0, 0])
        colors = ['navy', 'red']
        cmap_mask = ListedColormap(colors)
        mask_plot = ax_mask1.imshow(detection_mask, cmap=cmap_mask, vmin=0, vmax=1,
                                   extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_mask1.set_title(f'Enhanced Plastic Detection Mask\nDetected: {area_stats["detected_area_m2"]:.0f} m²', fontsize=12, pad=15)
        ax_mask1.set_xlabel('Longitude', fontsize=10)
        ax_mask1.set_ylabel('Latitude', fontsize=10)
        ax_mask1.grid(True, alpha=0.3)
        
        # Add legend
        water_patch = mpatches.Patch(color='navy', label='Water')
        plastic_patch = mpatches.Patch(color='red', label=f'Detected Plastic ({area_stats["detected_pixels"]} pixels)')
        ax_mask1.legend(handles=[water_patch, plastic_patch], loc='upper right', fontsize=10)
        
        # RGB with overlay
        ax_mask2 = fig2.add_subplot(gs2[0, 1])
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
        ax_mask2.set_title('RGB + Detection Overlay', fontsize=12, pad=15)
        ax_mask2.set_xlabel('Longitude', fontsize=10)
        ax_mask2.set_ylabel('Latitude', fontsize=10)
        ax_mask2.grid(True, alpha=0.3)
        
        # Enhanced plastic index
        ax_mask3 = fig2.add_subplot(gs2[0, 2])
        enhanced_plot = ax_mask3.imshow(indices['enhanced_plastic'], cmap='hot', vmin=0, vmax=0.2,
                                       extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_mask3.set_title('Enhanced Plastic Index\n(Multi-sensor Fusion)', fontsize=12, pad=15)
        ax_mask3.set_xlabel('Longitude', fontsize=10)
        ax_mask3.set_ylabel('Latitude', fontsize=10)
        ax_mask3.grid(True, alpha=0.3)
        cbar_enh = plt.colorbar(enhanced_plot, ax=ax_mask3, shrink=0.8)
        cbar_enh.set_label('Enhanced Plastic', fontsize=9)
        
        # Water mask visualization
        ax_water = fig2.add_subplot(gs2[1, 0])
        colors_water = ['brown', 'blue']
        cmap_water = ListedColormap(colors_water)
        water_plot = ax_water.imshow(water_mask, cmap=cmap_water, vmin=0, vmax=1,
                                    extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
        ax_water.set_title('Water Mask (NDWI-based)\nBrown=Land, Blue=Water', fontsize=11, pad=15)
        ax_water.set_xlabel('Longitude', fontsize=10)
        ax_water.set_ylabel('Latitude', fontsize=10)
        ax_water.grid(True, alpha=0.3)
        
        # Add legend for water mask  
        land_patch = mpatches.Patch(color='brown', label='Land')
        water_patch_mask = mpatches.Patch(color='blue', label='Water')
        ax_water.legend(handles=[land_patch, water_patch_mask], loc='upper right', fontsize=10)
        
        # SAR vs Optical scatter plot
        ax_scatter = fig2.add_subplot(gs2[1, 1])
        if len(center_data_clean) > 1000:
            # Sample data for faster plotting
            sample_idx = np.random.choice(len(center_data_clean), 1000, replace=False)
            sample_data = center_data_clean[sample_idx]
            
            ax_scatter.scatter(sample_data[:, 2], sample_data[:, 5], alpha=0.5, s=10)  # Red vs VV
            ax_scatter.set_xlabel('Red Band (B04)', fontsize=10)
            ax_scatter.set_ylabel('VV Polarization', fontsize=10)
            ax_scatter.set_title('Optical vs SAR\nCorrelation', fontsize=11, pad=15)
            ax_scatter.grid(True, alpha=0.3)
        else:
            ax_scatter.text(0.5, 0.5, 'Insufficient\ndata for\nscatter plot', 
                           ha='center', va='center', transform=ax_scatter.transAxes, fontsize=10)
            ax_scatter.set_title('Optical vs SAR Correlation', fontsize=11, pad=15)
        
        # Statistics
        ax_stats = fig2.add_subplot(gs2[1, 2])
        ax_stats.axis('off')
        
        stats_text = f"""ENHANCED DETECTION STATISTICS

Area Coverage:
• Total AOI: {area_stats['total_area_km2']:.2f} km²
• {area_stats['area_type']}: {area_stats['reference_area_km2']:.2f} km²
• Detected plastic: {area_stats['detected_area_km2']:.4f} km²
• Plastic area: {area_stats['detected_area_m2']:.0f} m²

Detection Performance:
• Coverage: {area_stats['coverage_percentage']:.3f}%
• Detected pixels: {area_stats['detected_pixels']:,}
• Pixel size: {area_stats['pixel_area_m2']:.1f} m²

Multi-sensor Features:
• Optical bands: 5 (Blue to SWIR)
• SAR polarizations: 2 (VV, VH)
• Derived indices: 8
• Enhanced plastic index threshold: 0.05

Detection Method:
• Multi-sensor fusion approach
• Combines spectral & backscatter
• Reduces false positives
• Weather-independent SAR"""
        
        ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes, fontsize=8,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8, pad=0.3))
        
        plt.suptitle('Enhanced Plastic Detection - Multi-sensor Binary Analysis\n' + 
                     f'Romanian Black Sea Coast | {time_interval[0]} to {time_interval[1]}', 
                     fontsize=14, y=0.98)
        
        # Save enhanced detection visualization
        enhanced_filename = os.path.join(data_dir, f"enhanced_detection_fusion_{time_interval[0]}_{time_interval[1]}.png")
        plt.savefig(enhanced_filename, dpi=300, bbox_inches='tight')
        print(f"✓ Enhanced detection visualization saved as: {enhanced_filename}")
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
        
        # CONFIGURATION - Consistent across all scripts
        # Define area of interest: Romanian coast of the Black Sea
        # Near the port of Constanța and Danube Delta - known shipping activity area
        # Coordinates: Longitude 28.5°E to 29.2°E, Latitude 44.0°N to 44.5°N
        bbox = BBox(bbox=[28.5, 44.0, 29.2, 44.5], crs=CRS.WGS84)
        
        # Time range: Extended summer period for better data availability (consistent across all scripts)
        time_interval = ('2025-07-01', '2025-07-31')
        
        # Image resolution - optimized for native satellite resolution
        # Sentinel-2: 10m native for RGB/NIR, Sentinel-1: resampled to 10m
        # For 0.7°x0.5° bbox (~77km x 55km), at 10m resolution = ~7700x5500 pixels  
        # Using 768x512 for computational efficiency while respecting 10m grid
        image_size = (768, 512)  # Optimized for 10m native resolution
        
        # Detection parameters (consistent across all scripts)
        # Detection parameters (standardized across all scripts)
        enhanced_plastic_threshold = 0.001  # Adaptive threshold using 95th percentile
        ndwi_threshold = -0.05              # More inclusive water detection
        
        # Resolution settings - use 10m which is native for both sensors
        target_resolution = 10  # meters - native for Sentinel-2 RGB/NIR and processed Sentinel-1
        
        print(f"Configuration:")
        print(f"  Area of Interest: {bbox}")
        print(f"  Time Interval: {time_interval}")
        print(f"  Image Size: {image_size} (optimized for {target_resolution}m native resolution)")
        print(f"  Target Resolution: {target_resolution}m (native for both Sentinel-1 & Sentinel-2)")
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
        detection_mask = create_enhanced_plastic_detection(indices, water_mask, enhanced_plastic_threshold)
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

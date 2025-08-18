#!/usr/bin/env python3
"""
Comprehensive Plastic Debris Detection using Multi-sensor Fusion and Machine Learning

This script combines the FDI detection method (Script 1) with multi-sensor data fusion (Script 2)
to perform comprehensive plastic debris detection. It implements multiple detection algorithms
and provides confidence-based results.

Key Features:
1. Downloads both Sentinel-1 SAR and Sentinel-2 optical data
2. Calculates Floating Debris Index (FDI) for initial detection
3. Applies machine learning clustering for refined detection
4. Combines multiple detection methods for improved accuracy
5. Provides confidence scores and uncertainty estimation

Study Area: Romanian coast of the Black Sea (near Constanța port)

Author: Varun Burde 
Email: varun@recycllux.com
Date: 2025
Reference: Combined methodology from Biermann et al. (2020) and Topouzelis et al. (2020)

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datetime as dt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

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
    load_dotenv()
    
    config = SHConfig()
    config.sh_client_id = os.environ.get('SH_CLIENT_ID')
    config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')
    
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Sentinel Hub credentials not found! "
            "Please set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables."
        )
    
    print("✓ Sentinel Hub credentials loaded successfully")
    return config

def download_multi_sensor_data(config, bbox, time_interval, size=(512, 512)):
    """
    Download both Sentinel-1 SAR and Sentinel-2 optical data
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        size: Output image size in pixels
        
    Returns:
        tuple: (optical_data, sar_data, data_mask)
    """
    print("Downloading multi-sensor satellite data...")
    
    # Evalscript for Sentinel-2 optical data (including RGB for visualization)
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
    
    # Download Sentinel-2 data
    request_s2 = SentinelHubRequest(
        evalscript=evalscript_s2,
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
    
    s2_data = request_s2.get_data()[0]
    optical_data = s2_data[:, :, :5]  # First 5 bands
    data_mask = s2_data[:, :, 5]      # Data mask
    
    # Evalscript for Sentinel-1 SAR data
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
    
    # Download Sentinel-1 data
    request_s1 = SentinelHubRequest(
        evalscript=evalscript_s1,
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
    
    sar_data = request_s1.get_data()[0]
    
    print(f"✓ Multi-sensor data downloaded:")
    print(f"  - Optical data shape: {optical_data.shape}")
    print(f"  - SAR data shape: {sar_data.shape}")
    print(f"  - Data mask coverage: {np.mean(data_mask)*100:.1f}%")
    
    return optical_data, sar_data, data_mask

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

def calculate_comprehensive_indices(optical_data, sar_data, data_mask):
    """
    Calculate comprehensive set of indices for plastic detection
    
    Args:
        optical_data: Sentinel-2 bands [Blue, Green, Red, NIR, SWIR]
        sar_data: Sentinel-1 polarizations [VV, VH]
        data_mask: Valid pixel mask
        
    Returns:
        dict: Dictionary of calculated indices
    """
    print("Calculating comprehensive indices...")
    
    # Extract optical bands
    blue = optical_data[:, :, 0]    # B02
    green = optical_data[:, :, 1]   # B03  
    red = optical_data[:, :, 2]     # B04
    nir = optical_data[:, :, 3]     # B08
    swir = optical_data[:, :, 4]    # B11
    
    # Extract SAR bands
    vv = sar_data[:, :, 0]
    vh = sar_data[:, :, 1]
    
    indices = {}
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # OPTICAL INDICES
        
        # 1. Floating Debris Index (FDI) - Primary plastic detection index
        lambda_red, lambda_nir, lambda_swir = 665.0, 842.0, 1610.0
        wavelength_factor = (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
        baseline = red + (swir - red) * wavelength_factor
        indices['fdi'] = nir - baseline
        
        # 2. Normalized Difference Vegetation Index (NDVI)
        indices['ndvi'] = (nir - red) / (nir + red + 1e-10)
        
        # 3. Normalized Difference Water Index (NDWI)
        indices['ndwi'] = (green - nir) / (green + nir + 1e-10)
        
        # 4. Enhanced Vegetation Index (EVI)
        indices['evi'] = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        
        # 5. Simple Ratio Index (SR)
        indices['sr'] = nir / (red + 1e-10)
        
        # 6. Atmospherically Resistant Vegetation Index (ARVI)
        rb = red - 2*(blue - red)
        indices['arvi'] = (nir - rb) / (nir + rb + 1e-10)
        
        # 7. Plastic Index (empirical combination)
        indices['plastic_index'] = (blue + red) / (2 * green + 1e-10)
        
        # 8. Modified FDI for better sensitivity
        indices['fdi_modified'] = indices['fdi'] / (red + 1e-10)
        
        # SAR INDICES
        
        # 9. Cross-polarization ratio
        indices['vh_vv_ratio'] = vh / (vv + 1e-10)
        
        # 10. SAR intensity
        indices['sar_intensity'] = np.sqrt(vv**2 + vh**2)
        
        # 11. Depolarization ratio
        indices['depol_ratio'] = vh / (vv + vh + 1e-10)
        
        # 12. Radar Vegetation Index (RVI)
        indices['rvi'] = 4 * vh / (vv + vh + 1e-10)
        
        # 13. VV in dB scale
        indices['vv_db'] = 10 * np.log10(np.maximum(vv, 1e-10))
        
        # 14. VH in dB scale
        indices['vh_db'] = 10 * np.log10(np.maximum(vh, 1e-10))
        
        # MULTI-SENSOR FUSION INDICES
        
        # 15. Optical-SAR composite for plastic detection
        indices['optical_sar_plastic'] = indices['fdi'] * (1 - indices['vh_vv_ratio'])
        
        # 16. Enhanced plastic index using both sensors
        indices['enhanced_plastic'] = (indices['fdi'] + 0.1) * np.exp(-indices['vh_vv_ratio'])
        
        # 17. Water quality proxy
        indices['water_quality'] = indices['ndwi'] * indices['sar_intensity']
        
        # 18. Surface roughness indicator
        indices['surface_roughness'] = indices['vh_vv_ratio'] * indices['sar_intensity']
    
    # Apply data mask to all indices
    for key, value in indices.items():
        indices[key][data_mask == 0] = np.nan
    
    print(f"✓ Calculated {len(indices)} indices")
    
    # Print statistics for key indices
    key_indices = ['fdi', 'vh_vv_ratio', 'optical_sar_plastic', 'enhanced_plastic']
    for key in key_indices:
        valid_data = indices[key][~np.isnan(indices[key])]
        if len(valid_data) > 0:
            print(f"  {key}: mean={np.mean(valid_data):.4f}, std={np.std(valid_data):.4f}")
    
    return indices

def detect_plastic_fdi_method(indices, water_mask, threshold=0.01):
    """
    Detect plastic using traditional FDI thresholding method (water areas only)
    
    Args:
        indices: Dictionary of calculated indices
        water_mask: Water mask (1=water, 0=land)
        threshold: FDI threshold for detection
        
    Returns:
        numpy.ndarray: Binary detection mask (1=plastic, 0=water, NaN=land)
    """
    print(f"Applying FDI threshold method (threshold: {threshold}, water areas only)...")
    
    fdi = indices['fdi']
    detection_mask = np.zeros_like(fdi)
    
    # Apply threshold only to water areas
    water_areas = (water_mask == 1) & (~np.isnan(fdi))
    detection_mask[water_areas & (fdi > threshold)] = 1
    
    # Set land areas and invalid pixels to NaN
    detection_mask[water_mask != 1] = np.nan
    detection_mask[np.isnan(fdi)] = np.nan

    valid_water_pixels = np.sum(water_mask == 1)
    detected_pixels = np.sum(detection_mask == 1)
    detection_rate = (detected_pixels / valid_water_pixels) * 100 if valid_water_pixels > 0 else 0

    print(f"✓ FDI detection: {detected_pixels} pixels ({detection_rate:.2f}% of water)")

    return detection_mask

def detect_plastic_ml_clustering(indices, data_mask, water_mask, n_clusters=3):
    """
    Detect plastic using unsupervised machine learning clustering (water areas only)
    
    Args:
        indices: Dictionary of calculated indices
        data_mask: Binary mask of valid data areas
        water_mask: Binary mask of water areas (True for water pixels)
        n_clusters: Number of clusters for K-means
        
    Returns:
        tuple: (cluster_labels, plastic_cluster_id, confidence_scores)
    """
    print(f"Applying ML clustering method ({n_clusters} clusters, water areas only)...")
    
    # Select most relevant indices for clustering
    key_indices = ['fdi', 'optical_sar_plastic', 'enhanced_plastic', 'vh_vv_ratio', 
                   'ndwi', 'plastic_index', 'sar_intensity', 'surface_roughness']
    
    # Stack indices for ML processing
    feature_stack = np.stack([indices[key] for key in key_indices], axis=2)
    height, width, n_features = feature_stack.shape
    
    # Create combined mask for valid water pixels only
    # Handle data_mask and water_mask properly
    data_mask_bool = (data_mask == 1)
    water_mask_bool = (water_mask == 1)
    valid_water_mask = data_mask_bool & water_mask_bool
    
    # Reshape to (n_pixels, n_features)
    features_reshaped = feature_stack.reshape(-1, n_features)
    valid_water_mask_flat = valid_water_mask.flatten()
    
    # Remove invalid pixels and keep only water pixels
    nan_mask = ~np.any(np.isnan(features_reshaped), axis=1)
    pixel_mask = nan_mask & valid_water_mask_flat
    features_valid = features_reshaped[pixel_mask]
    
    if len(features_valid) < 100:
        print("Warning: Insufficient valid water data for clustering")
        return np.zeros((height, width)), 0, np.zeros((height, width))
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels_valid = kmeans.fit_predict(features_scaled)
    
    # Map back to spatial coordinates (only for water pixels)
    cluster_labels_full = np.full(len(features_reshaped), -1)
    cluster_labels_full[pixel_mask] = cluster_labels_valid
    cluster_labels_spatial = cluster_labels_full.reshape(height, width)
    
    # Apply water mask to cluster results
    cluster_labels_spatial[water_mask != 1] = -1
    
    # Identify plastic cluster (highest mean FDI and enhanced_plastic values)
    cluster_scores = []
    for i in range(n_clusters):
        cluster_mask = cluster_labels_valid == i
        if np.sum(cluster_mask) > 0:
            fdi_mean = np.mean(features_valid[cluster_mask, 0])  # FDI is first feature
            enhanced_mean = np.mean(features_valid[cluster_mask, 2])  # Enhanced plastic is third
            score = fdi_mean + enhanced_mean
            cluster_scores.append(score)
        else:
            cluster_scores.append(-np.inf)
    
    plastic_cluster_id = np.argmax(cluster_scores)
    
    # Calculate confidence scores based on distance to cluster centers
    distances = kmeans.transform(features_scaled)
    min_distances = np.min(distances, axis=1)
    confidence_scores_valid = 1 / (1 + min_distances)  # Higher confidence = closer to cluster center
    
    confidence_scores_full = np.zeros(len(features_reshaped))
    confidence_scores_full[pixel_mask] = confidence_scores_valid
    confidence_scores_spatial = confidence_scores_full.reshape(height, width)
    
    plastic_pixels = np.sum(cluster_labels_spatial == plastic_cluster_id)
    total_valid_pixels = np.sum(pixel_mask)
    detection_rate = (plastic_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
    
    print(f"✓ ML clustering: {plastic_pixels} pixels ({detection_rate:.2f}%) in plastic cluster")
    print(f"  Plastic cluster ID: {plastic_cluster_id}")
    print(f"  Cluster sizes: {[np.sum(cluster_labels_valid == i) for i in range(n_clusters)]}")
    
    return cluster_labels_spatial, plastic_cluster_id, confidence_scores_spatial

def detect_plastic_anomaly_detection(indices, data_mask, water_mask):
    """
    Detect plastic using anomaly detection (Isolation Forest) in water areas only
    
    Args:
        indices: Dictionary of calculated indices
        data_mask: Binary mask of valid data areas
        water_mask: Binary mask of water areas (True for water pixels)
        
    Returns:
        tuple: (anomaly_scores, anomaly_mask)
    """
    print("Applying anomaly detection method (water areas only)...")
    
    # Select features for anomaly detection
    key_indices = ['fdi', 'optical_sar_plastic', 'enhanced_plastic', 'plastic_index']
    
    # Stack indices
    feature_stack = np.stack([indices[key] for key in key_indices], axis=2)
    height, width, n_features = feature_stack.shape
    
    # Create combined mask for valid water pixels only
    # Handle data_mask and water_mask properly
    data_mask_bool = (data_mask == 1)
    water_mask_bool = (water_mask == 1)
    valid_water_mask = data_mask_bool & water_mask_bool
    
    # Reshape and clean data (water pixels only)
    features_reshaped = feature_stack.reshape(-1, n_features)
    valid_water_mask_flat = valid_water_mask.flatten()
    nan_mask = ~np.any(np.isnan(features_reshaped), axis=1)
    pixel_mask = nan_mask & valid_water_mask_flat
    features_valid = features_reshaped[pixel_mask]
    
    if len(features_valid) < 100:
        print("Warning: Insufficient valid water data for anomaly detection")
        return np.zeros((height, width)), np.zeros((height, width))
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_valid)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = iso_forest.fit_predict(features_scaled)
    anomaly_scores_valid = iso_forest.score_samples(features_scaled)
    
    # Map back to spatial coordinates (water pixels only)
    anomaly_scores_full = np.zeros(len(features_reshaped))
    anomaly_scores_full[pixel_mask] = anomaly_scores_valid
    anomaly_scores_spatial = anomaly_scores_full.reshape(height, width)
    
    anomaly_labels_full = np.zeros(len(features_reshaped))
    anomaly_labels_full[pixel_mask] = anomaly_labels
    anomaly_mask_spatial = anomaly_labels_full.reshape(height, width)
    anomaly_mask_spatial = (anomaly_mask_spatial == -1).astype(float)  # -1 indicates anomaly
    
    # Apply water mask to results
    anomaly_scores_spatial[water_mask != 1] = 0
    anomaly_mask_spatial[water_mask != 1] = 0
    
    anomaly_pixels = np.sum(anomaly_mask_spatial == 1)
    total_water_pixels = np.sum((water_mask == 1) & (data_mask == 1))
    detection_rate = (anomaly_pixels / total_water_pixels) * 100 if total_water_pixels > 0 else 0
    
    print(f"✓ Anomaly detection: {anomaly_pixels} pixels ({detection_rate:.2f}%) classified as anomalies in water areas")
    
    return anomaly_scores_spatial, anomaly_mask_spatial

def create_ensemble_detection(fdi_mask, ml_clusters, plastic_cluster_id, anomaly_mask, confidence_scores, water_mask):
    """
    Create ensemble detection by combining multiple methods (water areas only)
    
    Args:
        fdi_mask: Binary mask from FDI thresholding
        ml_clusters: Cluster labels from ML clustering
        plastic_cluster_id: ID of the plastic cluster
        anomaly_mask: Binary mask from anomaly detection
        confidence_scores: Confidence scores from ML clustering
        water_mask: Binary mask of water areas (True for water pixels)
        
    Returns:
        tuple: (ensemble_mask, ensemble_confidence)
    """
    print("Creating ensemble detection (water areas only)...")
    
    # Create ML binary mask (only for the plastic cluster)
    ml_mask = (ml_clusters == plastic_cluster_id).astype(float)
    
    # Apply water mask to all individual detections
    fdi_mask_water = fdi_mask.copy()
    fdi_mask_water[water_mask != 1] = 0
    
    ml_mask_water = ml_mask.copy()
    ml_mask_water[water_mask != 1] = 0
    
    anomaly_mask_water = anomaly_mask.copy()
    anomaly_mask_water[water_mask != 1] = 0
    
    # Combine detection methods (water areas only)
    # Each method contributes to the ensemble score
    ensemble_score = np.zeros_like(fdi_mask_water)
    
    # FDI contribution (weight: 0.4)
    fdi_contribution = np.nan_to_num(fdi_mask_water, 0) * 0.4
    
    # ML clustering contribution (weight: 0.4)
    ml_contribution = np.nan_to_num(ml_mask_water, 0) * 0.4
    
    # Anomaly detection contribution (weight: 0.2)
    anomaly_contribution = np.nan_to_num(anomaly_mask_water, 0) * 0.2
    
    # Combine contributions
    ensemble_score = fdi_contribution + ml_contribution + anomaly_contribution
    
    # Create ensemble confidence by combining individual confidences (water areas only)
    ensemble_confidence = np.zeros_like(confidence_scores)
    ensemble_confidence[water_mask == 1] = confidence_scores[water_mask == 1] * ensemble_score[water_mask == 1]
    ensemble_confidence[water_mask != 1] = 0
    
    # Create final binary mask (ensemble detection > 0.5, water areas only)
    ensemble_mask = (ensemble_score > 0.5).astype(float)
    ensemble_mask[water_mask != 1] = 0
    
    # Calculate statistics for water areas
    total_water = np.sum(water_mask == 1)
    ensemble_detections = np.sum(ensemble_mask == 1)
    ensemble_rate = (ensemble_detections / total_water) * 100 if total_water > 0 else 0
    
    print(f"✓ Ensemble detection: {ensemble_detections} pixels ({ensemble_rate:.2f}%) in water areas")
    print(f"  Method agreement statistics (water areas only):")
    print(f"  - FDI only: {np.sum(fdi_mask_water == 1)} pixels")
    print(f"  - ML only: {np.sum(ml_mask_water == 1)} pixels") 
    print(f"  - Anomaly only: {np.sum(anomaly_mask_water == 1)} pixels")
    print(f"  - All methods agree: {np.sum((fdi_mask_water == 1) & (ml_mask_water == 1) & (anomaly_mask_water == 1))} pixels")
    
    return ensemble_mask, ensemble_confidence

def calculate_plastic_area_statistics(detection_mask, bbox, water_mask=None):
    """
    Calculate plastic area statistics for any detection mask (water areas only if water_mask provided)
    
    Args:
        detection_mask: Binary detection mask
        bbox: Bounding box for area calculation
        water_mask: Optional binary mask of water areas (True for water pixels)
        
    Returns:
        dict: Dictionary containing area statistics
    """
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
    
    # If water mask is provided, calculate statistics for water areas only
    if water_mask is not None:
        water_pixels = np.sum(water_mask == 1)
        water_area_m2 = water_pixels * pixel_area_m2
        water_area_km2 = water_area_m2 / 1e6

        # Count detections in water areas only using logical AND
        detection_mask_water = np.logical_and(detection_mask == 1, water_mask == 1)
        detected_pixels = np.sum(detection_mask_water)
        coverage_percentage = (detected_pixels / water_pixels * 100) if water_pixels > 0 else 0

        area_type = "water areas"
        reference_area_m2 = water_area_m2
        reference_area_km2 = water_area_km2
    else:
        # Original calculation for total area (ignore NaNs in detection_mask)
        detected_pixels = np.nansum(detection_mask)
        total_pixels = image_height * image_width
        coverage_percentage = (detected_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        area_type = "total area"
        reference_area_m2 = total_area_m2
        reference_area_km2 = total_area_km2
    
    # Calculate detected areas
    detected_area_m2 = detected_pixels * pixel_area_m2
    detected_area_km2 = detected_area_m2 / 1e6
    
    return {
        'total_area_km2': total_area_km2,
        'reference_area_km2': reference_area_km2,
        'detected_area_km2': detected_area_km2,
        'detected_area_m2': detected_area_m2,
        'detected_pixels': int(detected_pixels),
        'coverage_percentage': coverage_percentage,
        'pixel_area_m2': pixel_area_m2,
        'area_type': area_type
    }

def create_multi_resolution_analysis(config, bbox, time_interval):
    """
    Perform multi-resolution analysis for comprehensive detection
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        
    Returns:
        dict: Multi-resolution analysis results
    """
    print("Performing multi-resolution analysis...")
    
    resolutions = {
        'low': (256, 256),
        'medium': (512, 512), 
        'high': (1024, 1024)
    }
    
    multi_res_results = {}
    
    for res_name, size in resolutions.items():
        print(f"  Processing {res_name} resolution ({size[0]}x{size[1]})...")
        
        try:
            # Download multi-sensor data for this resolution
            optical_data, sar_data, data_mask = download_multi_sensor_data(
                config, bbox, time_interval, size
            )
            
            # Calculate indices
            indices = calculate_comprehensive_indices(optical_data, sar_data, data_mask)
            
            # Create water mask for this resolution
            water_mask_res, _ = create_water_mask(
                optical_data[:, :, 1], optical_data[:, :, 3], data_mask
            )
            # Apply different detection methods
            fdi_mask = detect_plastic_fdi_method(indices, water_mask_res, threshold=0.01)
            ml_clusters, plastic_cluster_id, confidence_scores = detect_plastic_ml_clustering(
                indices, data_mask, water_mask_res
            )
            anomaly_scores, anomaly_mask = detect_plastic_anomaly_detection(
                indices, data_mask, water_mask_res
            )
            ensemble_mask, ensemble_confidence = create_ensemble_detection(
                fdi_mask, ml_clusters, plastic_cluster_id,
                anomaly_mask, confidence_scores, water_mask_res
            )
            
            # Calculate area statistics for each method
            area_stats = {
                'fdi': calculate_plastic_area_statistics(fdi_mask, bbox, water_mask_res),
                'ml': calculate_plastic_area_statistics(
                    (ml_clusters == plastic_cluster_id).astype(float), bbox, water_mask_res
                ),
                'anomaly': calculate_plastic_area_statistics(anomaly_mask, bbox, water_mask_res),
                'ensemble': calculate_plastic_area_statistics(ensemble_mask, bbox, water_mask_res)
            }
            
            multi_res_results[res_name] = {
                'size': size,
                'optical_data': optical_data,
                'sar_data': sar_data,
                'indices': indices,
                'fdi_mask': fdi_mask,
                'ensemble_mask': ensemble_mask,
                'ensemble_confidence': ensemble_confidence,
                'area_stats': area_stats
            }
            
        except Exception as e:
            print(f"    Warning: Failed to process {res_name} resolution: {e}")
            continue
    
    print(f"✓ Multi-resolution analysis completed for {len(multi_res_results)} resolutions")
    return multi_res_results

def visualize_comprehensive_results(optical_data, sar_data, indices, detections, bbox, time_interval, multi_res_results=None):
    """
    Create comprehensive visualization of all detection results
    
    Args:
        optical_data: Sentinel-2 optical bands
        sar_data: Sentinel-1 SAR data
        indices: Dictionary of calculated indices
        detections: Dictionary of detection results
        bbox: Area of interest
        time_interval: Time period
    """
    print("Creating comprehensive visualization...")
    
    # Create large figure with multiple subplots
    fig = plt.figure(figsize=(28, 20))
    
    # 1. RGB True Color
    ax1 = plt.subplot(4, 5, 1)
    rgb_image = np.stack([
        optical_data[:, :, 2],  # Red
        optical_data[:, :, 1],  # Green  
        optical_data[:, :, 0]   # Blue
    ], axis=2)
    rgb_display = np.clip(rgb_image * 3.5, 0, 1)
    ax1.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax1.set_title('Sentinel-2 RGB\n(True Color)', fontsize=11)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. FDI Index
    ax2 = plt.subplot(4, 5, 2)
    fdi_plot = ax2.imshow(indices['fdi'], cmap='RdBu_r', vmin=-0.05, vmax=0.05,
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax2.set_title('FDI\n(Floating Debris Index)', fontsize=11)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(fdi_plot, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Enhanced Plastic Index
    ax3 = plt.subplot(4, 5, 3)
    plastic_plot = ax3.imshow(indices['enhanced_plastic'], cmap='Reds', vmin=0, vmax=0.15,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax3.set_title('Enhanced Plastic Index\n(Multi-sensor)', fontsize=11)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(plastic_plot, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. SAR VV (dB)
    ax4 = plt.subplot(4, 5, 4)
    vv_db = 10 * np.log10(np.maximum(sar_data[:, :, 0], 1e-10))
    vv_plot = ax4.imshow(vv_db, cmap='gray', vmin=-25, vmax=0,
                        extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax4.set_title('Sentinel-1 VV\n(dB)', fontsize=11)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(vv_plot, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. VH/VV Ratio
    ax5 = plt.subplot(4, 5, 5)
    ratio_plot = ax5.imshow(indices['vh_vv_ratio'], cmap='viridis', vmin=0, vmax=1,
                           extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax5.set_title('VH/VV Ratio\n(Cross-polarization)', fontsize=11)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(ratio_plot, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. FDI Detection
    ax6 = plt.subplot(4, 5, 6)
    colors = ['blue', 'red']
    cmap = ListedColormap(colors)
    fdi_det_plot = ax6.imshow(detections['fdi_mask'], cmap=cmap, vmin=0, vmax=1,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax6.set_title('FDI Detection\n(Threshold Method)', fontsize=11)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    ax6.grid(True, alpha=0.3)
    
    # 7. ML Clustering
    ax7 = plt.subplot(4, 5, 7)
    ml_mask = (detections['ml_clusters'] == detections['plastic_cluster_id']).astype(float)
    ml_mask[detections['ml_clusters'] == -1] = np.nan
    ml_plot = ax7.imshow(ml_mask, cmap=cmap, vmin=0, vmax=1,
                        extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax7.set_title('ML Clustering\n(K-means)', fontsize=11)
    ax7.set_xlabel('Longitude')
    ax7.set_ylabel('Latitude')
    ax7.grid(True, alpha=0.3)
    
    # 8. Anomaly Detection
    ax8 = plt.subplot(4, 5, 8)
    anomaly_plot = ax8.imshow(detections['anomaly_mask'], cmap=cmap, vmin=0, vmax=1,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax8.set_title('Anomaly Detection\n(Isolation Forest)', fontsize=11)
    ax8.set_xlabel('Longitude')
    ax8.set_ylabel('Latitude')
    ax8.grid(True, alpha=0.3)
    
    # 9. Ensemble Detection
    ax9 = plt.subplot(4, 5, 9)
    ensemble_plot = ax9.imshow(detections['ensemble_mask'], cmap=cmap, vmin=0, vmax=1,
                              extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax9.set_title('Ensemble Detection\n(Combined Methods)', fontsize=11)
    ax9.set_xlabel('Longitude')
    ax9.set_ylabel('Latitude')
    ax9.grid(True, alpha=0.3)
    
    # Add legend for detection plots
    import matplotlib.patches as mpatches
    water_patch = mpatches.Patch(color='blue', label='Water')
    plastic_patch = mpatches.Patch(color='red', label='Potential Plastic')
    ax9.legend(handles=[water_patch, plastic_patch], loc='upper right', fontsize=8)
    
    # 10. Confidence Map
    ax10 = plt.subplot(4, 5, 10)
    conf_plot = ax10.imshow(detections['ensemble_confidence'], cmap='hot', vmin=0, vmax=1,
                           extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax10.set_title('Detection Confidence\n(0=Low, 1=High)', fontsize=11)
    ax10.set_xlabel('Longitude')
    ax10.set_ylabel('Latitude')
    ax10.grid(True, alpha=0.3)
    plt.colorbar(conf_plot, ax=ax10, fraction=0.046, pad=0.04)
    
    # 11. RGB with Ensemble Overlay
    ax11 = plt.subplot(4, 5, 11)
    ax11.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    overlay = np.ma.masked_where(detections['ensemble_mask'] != 1, detections['ensemble_mask'])
    ax11.imshow(overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1,
               extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax11.set_title('RGB + Detection Overlay\n(Red = Detected Plastic)', fontsize=11)
    ax11.set_xlabel('Longitude')
    ax11.set_ylabel('Latitude')
    ax11.grid(True, alpha=0.3)
    
    # 12. Detection Method Comparison
    ax12 = plt.subplot(4, 5, 12)
    methods = ['FDI', 'ML\nCluster', 'Anomaly', 'Ensemble']
    detection_counts = [
        np.sum(detections['fdi_mask'] == 1),
        np.sum(ml_mask == 1),
        np.sum(detections['anomaly_mask'] == 1),
        np.sum(detections['ensemble_mask'] == 1)
    ]
    bars = ax12.bar(methods, detection_counts, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax12.set_title('Detection Count\nComparison', fontsize=11)
    ax12.set_ylabel('Detected Pixels')
    ax12.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, detection_counts):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(count), ha='center', va='bottom', fontsize=9)
    
    # 13. Index Correlation Heatmap
    ax13 = plt.subplot(4, 5, 13)
    key_indices = ['fdi', 'enhanced_plastic', 'vh_vv_ratio', 'ndwi', 'plastic_index']
    correlation_data = []
    for idx_name in key_indices:
        idx_data = indices[idx_name]
        valid_data = idx_data[~np.isnan(idx_data)].flatten()
        correlation_data.append(valid_data[:1000] if len(valid_data) > 1000 else valid_data)
    
    # Pad arrays to same length
    max_len = max(len(arr) for arr in correlation_data)
    correlation_matrix = np.zeros((len(key_indices), len(key_indices)))
    
    for i in range(len(key_indices)):
        for j in range(len(key_indices)):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                data1 = correlation_data[i]
                data2 = correlation_data[j]
                min_len = min(len(data1), len(data2))
                if min_len > 10:
                    correlation_matrix[i, j] = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]
    
    im = ax13.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax13.set_title('Index Correlation\nMatrix', fontsize=11)
    ax13.set_xticks(range(len(key_indices)))
    ax13.set_yticks(range(len(key_indices)))
    ax13.set_xticklabels([idx.upper() for idx in key_indices], rotation=45, fontsize=8)
    ax13.set_yticklabels([idx.upper() for idx in key_indices], fontsize=8)
    plt.colorbar(im, ax=ax13, fraction=0.046, pad=0.04)
    
    # 14. Detection Quality Metrics
    ax14 = plt.subplot(4, 5, 14)
    ax14.axis('off')
    
    # Calculate agreement metrics
    fdi_det = detections['fdi_mask'] == 1
    ml_det = ml_mask == 1
    anomaly_det = detections['anomaly_mask'] == 1
    ensemble_det = detections['ensemble_mask'] == 1
    
    # Method agreement
    fdi_ml_agree = np.sum(fdi_det & ml_det)
    fdi_anomaly_agree = np.sum(fdi_det & anomaly_det)
    ml_anomaly_agree = np.sum(ml_det & anomaly_det)
    all_agree = np.sum(fdi_det & ml_det & anomaly_det)
    
    metrics_text = f"""
DETECTION QUALITY METRICS

Method Agreement:
• FDI ∩ ML: {fdi_ml_agree} pixels
• FDI ∩ Anomaly: {fdi_anomaly_agree} pixels
• ML ∩ Anomaly: {ml_anomaly_agree} pixels
• All methods: {all_agree} pixels

Detection Rates:
• FDI: {np.sum(fdi_det)} pixels
• ML Clustering: {np.sum(ml_det)} pixels
• Anomaly Detection: {np.sum(anomaly_det)} pixels
• Ensemble: {np.sum(ensemble_det)} pixels

Confidence Statistics:
• Mean confidence: {np.nanmean(detections['ensemble_confidence']):.3f}
• High confidence (>0.7): {np.sum(detections['ensemble_confidence'] > 0.7)} pixels
• Low confidence (<0.3): {np.sum(detections['ensemble_confidence'] < 0.3)} pixels

Data Quality:
• Valid pixels: {np.sum(~np.isnan(detections['fdi_mask']))} / {detections['fdi_mask'].size}
• Coverage: {100 * np.sum(~np.isnan(detections['fdi_mask'])) / detections['fdi_mask'].size:.1f}%
    """
    
    ax14.text(0.05, 0.95, metrics_text, transform=ax14.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 15. Algorithm Information
    ax15 = plt.subplot(4, 5, 15)
    ax15.axis('off')
    
    algorithm_text = f"""
ALGORITHM SUMMARY

Data Sources:
• Sentinel-2 L2A (Optical)
• Sentinel-1 IW (SAR)
• Time: {time_interval[0]} to {time_interval[1]}

Detection Methods:
1. FDI Thresholding
   └─ Baseline: Linear interpolation
   └─ Threshold: 0.01

2. ML Clustering
   └─ Algorithm: K-means (3 clusters)
   └─ Features: 8 key indices
   └─ Preprocessing: StandardScaler

3. Anomaly Detection
   └─ Algorithm: Isolation Forest
   └─ Contamination: 10%
   └─ Features: 4 plastic indices

4. Ensemble Method
   └─ Weights: FDI(40%), ML(40%), Anomaly(20%)
   └─ Threshold: 0.5 for final detection

Indices Calculated: 18
• Optical: 8 indices
• SAR: 6 indices  
• Multi-sensor: 4 indices

Study Area:
• Romanian Black Sea Coast
• Near Constanța Port & Danube Delta
• Bbox: {bbox.min_x:.1f}°E - {bbox.max_x:.1f}°E
        {bbox.min_y:.1f}°N - {bbox.max_y:.1f}°N
    """
    
    ax15.text(0.05, 0.95, algorithm_text, transform=ax15.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 16-20: Index detail plots
    detail_indices = ['fdi', 'optical_sar_plastic', 'vh_vv_ratio', 'plastic_index', 'sar_intensity']
    detail_titles = ['FDI Detail', 'Optical-SAR Composite', 'VH/VV Ratio Detail', 
                    'Plastic Index Detail', 'SAR Intensity Detail']
    
    for i, (idx_name, title) in enumerate(zip(detail_indices, detail_titles)):
        ax = plt.subplot(4, 5, 16 + i)
        idx_data = indices[idx_name]
        valid_data = idx_data[~np.isnan(idx_data)]
        
        if len(valid_data) > 0:
            ax.hist(valid_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{title}\nDistribution', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Comprehensive Plastic Debris Detection - Multi-sensor Fusion Analysis\n' + 
                 f'Romanian Black Sea Coast | {time_interval[0]} to {time_interval[1]}', 
                 fontsize=16, y=0.98)
    
    # Create data directory if it doesn't exist
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save the comprehensive plot
    output_filename = os.path.join(data_dir, f"comprehensive_plastic_detection_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive visualization saved as: {output_filename}")
    
    plt.show()
    
    # Create additional binary mask visualizations
    print("Creating binary mask visualizations...")
    
    # Calculate area statistics for all methods
    fdi_area_stats = calculate_plastic_area_statistics(detections['fdi_mask'], bbox)
    ml_mask = (detections['ml_clusters'] == detections['plastic_cluster_id']).astype(float)
    # Set invalid (unassigned) cluster pixels to 0 instead of NaN for visualization
    ml_mask[detections['ml_clusters'] == -1] = 0
    ml_area_stats = calculate_plastic_area_statistics(ml_mask, bbox)
    anomaly_area_stats = calculate_plastic_area_statistics(detections['anomaly_mask'], bbox)
    ensemble_area_stats = calculate_plastic_area_statistics(detections['ensemble_mask'], bbox)
    
    # Create binary mask comparison figure
    fig_masks = plt.figure(figsize=(20, 16))
    
    # Define colors for binary masks
    colors = ['navy', 'red']
    cmap_binary = ListedColormap(colors)
    
    # 1. FDI Binary Mask
    ax1 = plt.subplot(3, 3, 1)
    fdi_plot = ax1.imshow(detections['fdi_mask'], cmap=cmap_binary, vmin=0, vmax=1,
                         extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax1.set_title(f'FDI Detection\n{fdi_area_stats["detected_area_m2"]:.0f} m² ({fdi_area_stats["coverage_percentage"]:.3f}%)', fontsize=12)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. ML Clustering Binary Mask
    ax2 = plt.subplot(3, 3, 2)
    ml_plot = ax2.imshow(ml_mask, cmap=cmap_binary, vmin=0, vmax=1,
                        extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax2.set_title(f'ML Clustering\n{ml_area_stats["detected_area_m2"]:.0f} m² ({ml_area_stats["coverage_percentage"]:.3f}%)', fontsize=12)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Anomaly Detection Binary Mask
    ax3 = plt.subplot(3, 3, 3)
    anomaly_plot = ax3.imshow(detections['anomaly_mask'], cmap=cmap_binary, vmin=0, vmax=1,
                             extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax3.set_title(f'Anomaly Detection\n{anomaly_area_stats["detected_area_m2"]:.0f} m² ({anomaly_area_stats["coverage_percentage"]:.3f}%)', fontsize=12)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Ensemble Binary Mask
    ax4 = plt.subplot(3, 3, 4)
    ensemble_plot = ax4.imshow(detections['ensemble_mask'], cmap=cmap_binary, vmin=0, vmax=1,
                              extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax4.set_title(f'Ensemble Detection\n{ensemble_area_stats["detected_area_m2"]:.0f} m² ({ensemble_area_stats["coverage_percentage"]:.3f}%)', fontsize=12)
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.grid(True, alpha=0.3)
    
    # 5. RGB with Ensemble Overlay
    ax5 = plt.subplot(3, 3, 5)
    rgb_image = np.stack([
        optical_data[:, :, 2],  # Red
        optical_data[:, :, 1],  # Green
        optical_data[:, :, 0]   # Blue
    ], axis=2)
    rgb_display = np.clip(rgb_image * 3.5, 0, 1)
    ax5.imshow(rgb_display, extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    
    # Overlay ensemble detection
    detection_overlay = np.ma.masked_where(detections['ensemble_mask'] != 1, detections['ensemble_mask'])
    ax5.imshow(detection_overlay, cmap='Reds', alpha=0.8, vmin=0, vmax=1,
              extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax5.set_title('RGB + Ensemble Overlay', fontsize=12)
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    ax5.grid(True, alpha=0.3)
    
    # 6. Confidence Map
    ax6 = plt.subplot(3, 3, 6)
    conf_plot = ax6.imshow(detections['ensemble_confidence'], cmap='hot', vmin=0, vmax=1,
                          extent=[bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y])
    ax6.set_title('Detection Confidence\n(0=Low, 1=High)', fontsize=12)
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(conf_plot, ax=ax6, fraction=0.046, pad=0.04)
    
    # 7. Area Comparison Chart
    ax7 = plt.subplot(3, 3, 7)
    methods = ['FDI', 'ML Cluster', 'Anomaly', 'Ensemble']
    areas = [fdi_area_stats['detected_area_m2'], ml_area_stats['detected_area_m2'], 
             anomaly_area_stats['detected_area_m2'], ensemble_area_stats['detected_area_m2']]
    
    bars = ax7.bar(methods, areas, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax7.set_title('Detected Area Comparison', fontsize=12)
    ax7.set_ylabel('Detected Area (m²)')
    ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, area in zip(bars, areas):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(areas)*0.01,
                f'{area:.0f} m²', ha='center', va='bottom', fontsize=9)
    
    # 8. Multi-resolution comparison (if available)
    if multi_res_results and len(multi_res_results) > 0:
        ax8 = plt.subplot(3, 3, 8)
        res_names = list(multi_res_results.keys())
        ensemble_areas = [multi_res_results[name]['area_stats']['ensemble']['detected_area_m2'] 
                         for name in res_names if 'area_stats' in multi_res_results[name]]
        
        if ensemble_areas:
            bars8 = ax8.bar(res_names, ensemble_areas, color=['lightcoral', 'gold', 'lightsteelblue'])
            ax8.set_title('Area by Resolution\n(Ensemble Method)', fontsize=12)
            ax8.set_ylabel('Detected Area (m²)')
            ax8.set_xlabel('Resolution')
            
            # Add value labels
            for bar, area in zip(bars8, ensemble_areas):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ensemble_areas)*0.01,
                        f'{area:.0f} m²', ha='center', va='bottom', fontsize=9)
        else:
            ax8.text(0.5, 0.5, 'Multi-resolution\ndata not available', 
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Multi-Resolution Analysis', fontsize=12)
    else:
        ax8 = plt.subplot(3, 3, 8)
        ax8.text(0.5, 0.5, 'Multi-resolution\nanalysis not performed', 
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Multi-Resolution Analysis', fontsize=12)
    
    # 9. Combined Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    combined_stats_text = f"""
COMPREHENSIVE AREA STATISTICS

Detection Method Results:
• FDI: {fdi_area_stats['detected_area_m2']:.0f} m²
• ML Clustering: {ml_area_stats['detected_area_m2']:.0f} m²
• Anomaly Detection: {anomaly_area_stats['detected_area_m2']:.0f} m²
• Ensemble: {ensemble_area_stats['detected_area_m2']:.0f} m²

Ensemble Performance:
• Total valid area: {ensemble_area_stats['reference_area_km2']:.2f} km²
• Plastic coverage: {ensemble_area_stats['coverage_percentage']:.3f}%
• Confidence mean: {np.nanmean(detections['ensemble_confidence']):.3f}
• High confidence (>0.7): {np.sum(detections['ensemble_confidence'] > 0.7)} pixels

Method Agreement:
• All methods agree: {np.sum((detections['fdi_mask'] == 1) & (ml_mask == 1) & (detections['anomaly_mask'] == 1))} pixels
• At least 2 agree: {np.sum(((detections['fdi_mask'] == 1).astype(int) + (ml_mask == 1).astype(int) + (detections['anomaly_mask'] == 1).astype(int)) >= 2)} pixels

Quality Metrics:
• Pixel resolution: {ensemble_area_stats['pixel_area_m2']:.1f} m²
• Data coverage: {ensemble_area_stats['coverage_percentage']:.1f}%
• Detection reliability: Multi-method validation
    """
    
    ax9.text(0.05, 0.95, combined_stats_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add legend for binary masks
    import matplotlib.patches as mpatches
    water_patch = mpatches.Patch(color='navy', label='Water')
    plastic_patch = mpatches.Patch(color='red', label='Detected Plastic')
    fig_masks.legend(handles=[water_patch, plastic_patch], loc='upper center', 
                    bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.suptitle('Binary Mask Analysis - Comprehensive Plastic Detection Results\n' + 
                 f'Romanian Black Sea Coast | {time_interval[0]} to {time_interval[1]}', 
                 fontsize=16, y=0.98)
    
    # Save binary mask analysis
    data_dir = "plastic_detection/data"
    binary_filename = os.path.join(data_dir, f"binary_masks_comprehensive_{time_interval[0]}_{time_interval[1]}.png")
    plt.savefig(binary_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Binary mask analysis saved as: {binary_filename}")
    
    plt.show()

def save_detection_results(detections, indices, time_interval):
    """
    Save all detection results to files
    
    Args:
        detections: Dictionary of detection results
        indices: Dictionary of calculated indices
        time_interval: Time period for filename
    """
    print("Saving detection results...")
    
    # Create data directory if it doesn't exist
    data_dir = "plastic_detection/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save detection masks
    detection_filename = os.path.join(data_dir, f"plastic_detection_results_{time_interval[0]}_{time_interval[1]}.npz")
    np.savez_compressed(
        detection_filename,
        fdi_mask=detections['fdi_mask'],
        ml_clusters=detections['ml_clusters'],
        plastic_cluster_id=detections['plastic_cluster_id'],
        anomaly_mask=detections['anomaly_mask'],
        anomaly_scores=detections['anomaly_scores'],
        ensemble_mask=detections['ensemble_mask'],
        ensemble_confidence=detections['ensemble_confidence'],
        description="Comprehensive plastic detection results using multiple methods"
    )
    
    # Save indices
    indices_filename = os.path.join(data_dir, f"calculated_indices_{time_interval[0]}_{time_interval[1]}.npz")
    np.savez_compressed(indices_filename, **indices)
    
    print(f"✓ Detection results saved as: {detection_filename}")
    print(f"✓ Indices saved as: {indices_filename}")

def main():
    """Main function to run comprehensive plastic detection"""
    print("=" * 80)
    print("COMPREHENSIVE PLASTIC DEBRIS DETECTION")
    print("Multi-sensor Fusion + Machine Learning + Ensemble Methods")
    print("=" * 80)
    print("Location: Romanian coast of the Black Sea (near Constanța)")
    print("Sensors: Sentinel-1 SAR + Sentinel-2 Optical")
    print("Methods: FDI + ML Clustering + Anomaly Detection + Ensemble")
    print("=" * 80)
    
    try:
        # Setup
        config = setup_credentials()
        # Define area of interest: Romanian coast of the Black Sea
        # Near the port of Constanța and Danube Delta - major shipping and river confluence area
        # Coordinates: Longitude 28.5°E to 29.2°E, Latitude 44.0°N to 44.5°N
        bbox = BBox(bbox=[28.5, 44.0, 29.2, 44.5], crs=CRS.WGS84)
        time_interval = ('2024-07-10', '2024-07-20')
        image_size = (512, 512)
        
        print(f"Configuration:")
        print(f"  Area: {bbox}")
        print(f"  Time: {time_interval}")
        print(f"  Size: {image_size}")
        print(f"  Data will be saved to: plastic_detection/data")
        
        # Step 1: Download data
        print(f"\n{'='*60}")
        print("STEP 1: DOWNLOADING MULTI-SENSOR DATA")
        print(f"{'='*60}")
        optical_data, sar_data, data_mask = download_multi_sensor_data(
            config, bbox, time_interval, image_size
        )
        
        # Step 1.5: Create water mask
        print(f"\n{'='*60}")
        print("STEP 1.5: CREATING WATER MASK")
        print(f"{'='*60}")
        # Extract Green (B03) and NIR (B08) from optical data for water masking
        green_band = optical_data[:, :, 1]  # B03
        nir_band = optical_data[:, :, 3]    # B08  
        water_mask, ndwi = create_water_mask(green_band, nir_band, data_mask)
        
        # Step 2: Calculate indices
        print(f"\n{'='*60}")
        print("STEP 2: CALCULATING COMPREHENSIVE INDICES")
        print(f"{'='*60}")
        indices = calculate_comprehensive_indices(optical_data, sar_data, data_mask)
        
        # Step 3: Apply detection methods (water areas only)
        print(f"\n{'='*60}")
        print("STEP 3: APPLYING DETECTION METHODS (WATER AREAS ONLY)")
        print(f"{'='*60}")
        
        # Method 1: FDI thresholding
        fdi_mask = detect_plastic_fdi_method(indices, water_mask, threshold=0.01)
        
        # Method 2: ML clustering
        ml_clusters, plastic_cluster_id, confidence_scores = detect_plastic_ml_clustering(indices, data_mask, water_mask)
        
        # Method 3: Anomaly detection
        anomaly_scores, anomaly_mask = detect_plastic_anomaly_detection(indices, data_mask, water_mask)
        
        # Step 4: Create ensemble detection
        print(f"\n{'='*60}")
        print("STEP 4: CREATING ENSEMBLE DETECTION")
        print(f"{'='*60}")
        ensemble_mask, ensemble_confidence = create_ensemble_detection(
            fdi_mask, ml_clusters, plastic_cluster_id, anomaly_mask, confidence_scores, water_mask
        )
        
        # Organize detection results
        detections = {
            'fdi_mask': fdi_mask,
            'ml_clusters': ml_clusters,
            'plastic_cluster_id': plastic_cluster_id,
            'anomaly_mask': anomaly_mask,
            'anomaly_scores': anomaly_scores,
            'ensemble_mask': ensemble_mask,
            'ensemble_confidence': ensemble_confidence
        }
        
        # Step 5: Visualization
        print(f"\n{'='*60}")
        print("STEP 5: MULTI-RESOLUTION ANALYSIS")
        print(f"{'='*60}")
        multi_res_results = create_multi_resolution_analysis(config, bbox, time_interval)
        
        # Step 6: Calculate area statistics (water areas only)
        print(f"\n{'='*60}")
        print("STEP 6: CALCULATING AREA STATISTICS (WATER AREAS ONLY)")
        print(f"{'='*60}")
        
        # Calculate statistics for each detection method
        fdi_stats = calculate_plastic_area_statistics(fdi_mask, bbox, water_mask)
        ml_mask_binary = (ml_clusters == plastic_cluster_id).astype(float)
        ml_stats = calculate_plastic_area_statistics(ml_mask_binary, bbox, water_mask)
        anomaly_stats = calculate_plastic_area_statistics(anomaly_mask, bbox, water_mask)
        ensemble_stats = calculate_plastic_area_statistics(ensemble_mask, bbox, water_mask)
        water_stats = calculate_plastic_area_statistics(water_mask.astype(float), bbox)
        
        print(f"Water Coverage Analysis:")
        print(f"  Total image area: {water_stats['total_area_km2']:.3f} km²")
        print(f"  Water area: {water_stats['detected_area_km2']:.3f} km² ({water_stats['coverage_percentage']:.1f}%)")
        print(f"  Land area: {water_stats['total_area_km2'] - water_stats['detected_area_km2']:.3f} km²")
        
        print(f"\nPlastic Detection Results (Water Areas Only):")
        print(f"  FDI Method: {fdi_stats['detected_area_km2']:.6f} km² ({fdi_stats['coverage_percentage']:.2f}% of water)")
        print(f"  ML Clustering: {ml_stats['detected_area_km2']:.6f} km² ({ml_stats['coverage_percentage']:.2f}% of water)")
        print(f"  Anomaly Detection: {anomaly_stats['detected_area_km2']:.6f} km² ({anomaly_stats['coverage_percentage']:.2f}% of water)")
        print(f"  Ensemble Method: {ensemble_stats['detected_area_km2']:.6f} km² ({ensemble_stats['coverage_percentage']:.2f}% of water)")
        
        # Add statistics to detections dictionary
        detections['water_mask'] = water_mask
        detections['ndwi'] = ndwi
        detections['area_statistics'] = {
            'fdi': fdi_stats,
            'ml': ml_stats,
            'anomaly': anomaly_stats,
            'ensemble': ensemble_stats,
            'water': water_stats
        }
        
        # Step 7: Comprehensive Visualization
        print(f"\n{'='*60}")
        print("STEP 7: CREATING COMPREHENSIVE VISUALIZATION")
        print(f"{'='*60}")
        visualize_comprehensive_results(optical_data, sar_data, indices, detections, bbox, time_interval, multi_res_results)
        
        # Step 8: Save results
        print(f"\n{'='*60}")
        print("STEP 8: SAVING RESULTS")
        print(f"{'='*60}")
        save_detection_results(detections, indices, time_interval)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE PLASTIC DETECTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        # Final summary
        print("\nFINAL DETECTION SUMMARY:")
        print(f"  FDI Method: {np.sum(fdi_mask == 1)} pixels detected")
        print(f"  ML Clustering: {np.sum(ml_clusters == plastic_cluster_id)} pixels detected")
        print(f"  Anomaly Detection: {np.sum(anomaly_mask == 1)} pixels detected")
        print(f"  Ensemble Method: {np.sum(ensemble_mask == 1)} pixels detected")
        print(f"  Mean Confidence: {np.nanmean(ensemble_confidence):.3f}")
        
        print("\nMETHOD ADVANTAGES:")
        print("• FDI: Well-established, physically-based index")
        print("• ML Clustering: Unsupervised, finds natural groupings")
        print("• Anomaly Detection: Identifies unusual patterns")
        print("• Ensemble: Combines strengths, reduces false positives")
        
        print("\nRECOMMENDATIONS:")
        print("• Focus on high-confidence detections (>0.7)")
        print("• Validate with high-resolution imagery")
        print("• Consider temporal analysis for confirmation")
        print("• Use ensemble results for operational decisions")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Check credentials and internet connection")
        print("2. Verify data availability for the time period")
        print("3. Ensure sufficient processing units")
        print("4. Try different time range if needed")
        print("5. Install required ML packages: sklearn")

if __name__ == "__main__":
    main()

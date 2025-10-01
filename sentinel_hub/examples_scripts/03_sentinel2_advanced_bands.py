#!/usr/bin/env python3
"""
Advanced Sentinel-2 Band Analysis Script

This script demonstrates advanced band combinations and spectral indices
for various applications like agriculture, water detection, urban analysis, etc.

Spectral bands and their applications:
- B01 (443nm): Coastal aerosol, atmospheric correction
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

def plot_index_with_colorbar(data, title, cmap='RdYlGn', vmin=None, vmax=None):
    """Plot an index with appropriate colorbar"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if vmin is None:
        vmin = np.percentile(data, 2)
    if vmax is None:
        vmax = np.percentile(data, 98)
    
    im = ax.imshow(data.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, label='Index Value')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def calculate_vegetation_indices(config, bbox, time_interval, size=(512, 512)):
    """
    Calculate various vegetation indices from Sentinel-2 data
    
    Vegetation Indices:
    - NDVI: (NIR - Red) / (NIR + Red)
    - EVI: 2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))
    - SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    - NDRE: (NIR - RedEdge) / (NIR + RedEdge)
    """
    print("Calculating vegetation indices...")
    
    evalscript_vegetation = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B04", "B05", "B08"]
            }],
            output: {
                bands: 4,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        var blue = sample.B02;
        var red = sample.B04;
        var redEdge = sample.B05;
        var nir = sample.B08;
        
        // NDVI
        var ndvi = (nir - red) / (nir + red);
        
        // EVI
        var evi = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1));
        
        // SAVI (Soil Adjusted Vegetation Index)
        var savi = ((nir - red) / (nir + red + 0.5)) * 1.5;
        
        // NDRE (Normalized Difference Red Edge)
        var ndre = (nir - redEdge) / (nir + redEdge);
        
        return [ndvi, evi, savi, ndre];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_vegetation,
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
    
    # Extract individual indices
    ndvi = data[:, :, 0]
    evi = data[:, :, 1]
    savi = data[:, :, 2]
    ndre = data[:, :, 3]
    
    # Plot all indices
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    indices = [
        (ndvi, "NDVI (Normalized Difference Vegetation Index)", (-1, 1)),
        (evi, "EVI (Enhanced Vegetation Index)", (-1, 1)),
        (savi, "SAVI (Soil Adjusted Vegetation Index)", (-1, 1)),
        (ndre, "NDRE (Normalized Difference Red Edge)", (-1, 1))
    ]
    
    for i, (index_data, title, vlim) in enumerate(indices):
        row, col = i // 2, i % 2
        im = axes[row, col].imshow(index_data, cmap='RdYlGn', vmin=vlim[0], vmax=vlim[1])
        axes[row, col].set_title(title)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        plt.colorbar(im, ax=axes[row, col])
    
    plt.tight_layout()
    plt.show()
    
    return {"NDVI": ndvi, "EVI": evi, "SAVI": savi, "NDRE": ndre}

def calculate_water_indices(config, bbox, time_interval, size=(512, 512)):
    """
    Calculate water detection indices
    
    Water Indices:
    - NDWI: (Green - NIR) / (Green + NIR)
    - MNDWI: (Green - SWIR1) / (Green + SWIR1)
    - AWEI: 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    """
    print("Calculating water indices...")
    
    evalscript_water = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B03", "B08", "B11", "B12"]
            }],
            output: {
                bands: 3,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        var green = sample.B03;
        var nir = sample.B08;
        var swir1 = sample.B11;
        var swir2 = sample.B12;
        
        // NDWI (Normalized Difference Water Index)
        var ndwi = (green - nir) / (green + nir);
        
        // MNDWI (Modified Normalized Difference Water Index)
        var mndwi = (green - swir1) / (green + swir1);
        
        // AWEI (Automated Water Extraction Index)
        var awei = 4*(green - swir1) - (0.25*nir + 2.75*swir2);
        
        return [ndwi, mndwi, awei];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_water,
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
    
    # Extract individual indices
    ndwi = data[:, :, 0]
    mndwi = data[:, :, 1]
    awei = data[:, :, 2]
    
    # Plot water indices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    water_indices = [
        (ndwi, "NDWI", 'Blues'),
        (mndwi, "MNDWI", 'Blues'),
        (awei, "AWEI", 'Blues')
    ]
    
    for i, (index_data, title, cmap) in enumerate(water_indices):
        im = axes[i].imshow(index_data, cmap=cmap)
        axes[i].set_title(title)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return {"NDWI": ndwi, "MNDWI": mndwi, "AWEI": awei}

def calculate_built_up_indices(config, bbox, time_interval, size=(512, 512)):
    """
    Calculate built-up/urban area indices
    
    Built-up Indices:
    - NDBI: (SWIR1 - NIR) / (SWIR1 + NIR)
    - UI: (SWIR2 - NIR) / (SWIR2 + NIR)
    - BU: (Red + SWIR1) - (NIR + Blue)
    """
    print("Calculating built-up area indices...")
    
    evalscript_builtup = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B04", "B08", "B11", "B12"]
            }],
            output: {
                bands: 3,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        var blue = sample.B02;
        var red = sample.B04;
        var nir = sample.B08;
        var swir1 = sample.B11;
        var swir2 = sample.B12;
        
        // NDBI (Normalized Difference Built-up Index)
        var ndbi = (swir1 - nir) / (swir1 + nir);
        
        // UI (Urban Index)
        var ui = (swir2 - nir) / (swir2 + nir);
        
        // BU (Built-up Index)
        var bu = (red + swir1) - (nir + blue);
        
        return [ndbi, ui, bu];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_builtup,
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
    
    # Extract individual indices
    ndbi = data[:, :, 0]
    ui = data[:, :, 1]
    bu = data[:, :, 2]
    
    # Plot built-up indices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    builtup_indices = [
        (ndbi, "NDBI", 'Reds'),
        (ui, "UI", 'Reds'),
        (bu, "BU", 'Reds')
    ]
    
    for i, (index_data, title, cmap) in enumerate(builtup_indices):
        im = axes[i].imshow(index_data, cmap=cmap)
        axes[i].set_title(title)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return {"NDBI": ndbi, "UI": ui, "BU": bu}

def create_advanced_composites(config, bbox, time_interval, size=(512, 512)):
    """
    Create advanced band composites for different applications
    
    Composites:
    - Agriculture: SWIR1, NIR, Red
    - Geology: SWIR2, SWIR1, Blue
    - Bathymetry: Red, Green, Blue/Green ratio
    - Vegetation analysis: NIR, Red Edge, Red
    """
    print("Creating advanced band composites...")
    
    evalscript_composites = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B05", "B08", "B11", "B12"]
            }],
            output: {
                bands: 12,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        var blue = sample.B02;
        var green = sample.B03;
        var red = sample.B04;
        var redEdge = sample.B05;
        var nir = sample.B08;
        var swir1 = sample.B11;
        var swir2 = sample.B12;
        
        // Agriculture composite (SWIR1, NIR, Red)
        var agr_r = swir1 * 3.5;
        var agr_g = nir * 3.5;
        var agr_b = red * 3.5;
        
        // Geology composite (SWIR2, SWIR1, Blue)
        var geo_r = swir2 * 3.5;
        var geo_g = swir1 * 3.5;
        var geo_b = blue * 3.5;
        
        // Vegetation analysis (NIR, Red Edge, Red)
        var veg_r = nir * 3.5;
        var veg_g = redEdge * 3.5;
        var veg_b = red * 3.5;
        
        // Bathymetry (enhanced water penetration)
        var bath_r = red * 3.5;
        var bath_g = green * 3.5;
        var bath_b = (blue / (green + 0.001)) * 2;
        
        return [
            agr_r, agr_g, agr_b,
            geo_r, geo_g, geo_b,
            veg_r, veg_g, veg_b,
            bath_r, bath_g, bath_b
        ];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_composites,
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
    
    # Extract composites
    agr_composite = np.clip(data[:, :, 0:3], 0, 1)
    geo_composite = np.clip(data[:, :, 3:6], 0, 1)
    veg_composite = np.clip(data[:, :, 6:9], 0, 1)
    bath_composite = np.clip(data[:, :, 9:12], 0, 1)
    
    # Plot composites
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    composites = [
        (agr_composite, "Agriculture Composite\n(SWIR1-NIR-Red)"),
        (geo_composite, "Geology Composite\n(SWIR2-SWIR1-Blue)"),
        (veg_composite, "Vegetation Analysis\n(NIR-RedEdge-Red)"),
        (bath_composite, "Bathymetry Composite\n(Red-Green-Blue/Green)")
    ]
    
    for i, (composite, title) in enumerate(composites):
        row, col = i // 2, i % 2
        axes[row, col].imshow(composite)
        axes[row, col].set_title(title)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return {
        "Agriculture": agr_composite,
        "Geology": geo_composite,
        "Vegetation": veg_composite,
        "Bathymetry": bath_composite
    }

def main():
    """Main function to run advanced band analysis examples"""
    print("=== Advanced Sentinel-2 Band Analysis ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest (example: Mixed landscape area)
    bbox = BBox(bbox=[14.0, 46.0, 14.3, 46.2], crs=CRS.WGS84)
    time_interval = ('2023-07-01', '2023-07-31')
    
    print(f"Area of Interest: {bbox}")
    print(f"Time Interval: {time_interval}")
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
        vegetation_indices = calculate_vegetation_indices(config, bbox, time_interval)
        print("✓ Vegetation indices calculated successfully\n")
        
        # Example 2: Water indices
        print("2. Water Detection Indices")
        water_indices = calculate_water_indices(config, bbox, time_interval)
        print("✓ Water indices calculated successfully\n")
        
        # Example 3: Built-up area indices
        print("3. Built-up Area Indices")
        builtup_indices = calculate_built_up_indices(config, bbox, time_interval)
        print("✓ Built-up indices calculated successfully\n")
        
        # Example 4: Advanced composites
        print("4. Advanced Band Composites")
        composites = create_advanced_composites(config, bbox, time_interval)
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
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()

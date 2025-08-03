#!/usr/bin/env python3
"""
Data Download and Export Script

This script demonstrates how to download satellite data and export it in different formats
for further analysis in GIS software or other applications:
- GeoTIFF export
- NetCDF export  
- Cloud Optimized GeoTIFF (COG)
- CSV data export
- Integration with external tools

Author: Learning Script
Date: 2025
"""

import os
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS as RasterCRS
import xarray as xr

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

def download_and_export_geotiff(config, bbox, time_interval, output_dir="outputs", size=(512, 512)):
    """
    Download satellite data and export as GeoTIFF files
    
    Args:
        config: SHConfig object with credentials
        bbox: Bounding box for the area of interest
        time_interval: Tuple of (start_date, end_date)
        output_dir: Directory to save outputs
        size: Output image size in pixels
    """
    print("Downloading and exporting as GeoTIFF...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evalscript for multiple bands
    evalscript_multi = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04", "B08", "B11", "B12"]
            }],
            output: {
                bands: 6,
                sampleType: "FLOAT32"
            }
        };
    }
    
    function evaluatePixel(sample) {
        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11, sample.B12];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript_multi,
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
    
    # Define band names
    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    
    # Calculate transform for georeferencing
    transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, size[0], size[1])
    
    # Export multi-band GeoTIFF
    multiband_path = os.path.join(output_dir, 'sentinel2_multiband.tif')
    with rasterio.open(
        multiband_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=data.shape[2],
        dtype=data.dtype,
        crs=RasterCRS.from_epsg(4326),
        transform=transform,
        compress='lzw'
    ) as dst:
        for i in range(data.shape[2]):
            dst.write(data[:, :, i], i + 1)
        
        # Add band descriptions
        dst.descriptions = band_names
    
    print(f"✓ Multi-band GeoTIFF saved: {multiband_path}")
    
    # Export individual band GeoTIFFs
    for i, band_name in enumerate(band_names):
        band_path = os.path.join(output_dir, f'sentinel2_{band_name.lower()}.tif')
        with rasterio.open(
            band_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=RasterCRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(data[:, :, i], 1)
            dst.set_band_description(1, band_name)
        
        print(f"✓ {band_name} band saved: {band_path}")
    
    return data, multiband_path

def calculate_and_export_indices(data, bbox, output_dir="outputs", size=(512, 512)):
    """
    Calculate vegetation indices and export them
    
    Args:
        data: Multi-band satellite data array
        bbox: Bounding box for georeferencing
        output_dir: Directory to save outputs
        size: Image size for georeferencing
    """
    print("Calculating and exporting vegetation indices...")
    
    # Extract bands (assuming order: Blue, Green, Red, NIR, SWIR1, SWIR2)
    blue = data[:, :, 0]
    green = data[:, :, 1]
    red = data[:, :, 2]
    nir = data[:, :, 3]
    swir1 = data[:, :, 4]
    swir2 = data[:, :, 5]
    
    # Calculate indices
    indices = {}
    
    # NDVI
    indices['NDVI'] = (nir - red) / (nir + red + 1e-10)
    
    # EVI
    indices['EVI'] = 2.5 * ((nir - red) / (nir + 6*red - 7.5*blue + 1))
    
    # NDWI
    indices['NDWI'] = (green - nir) / (green + nir + 1e-10)
    
    # NDBI
    indices['NDBI'] = (swir1 - nir) / (swir1 + nir + 1e-10)
    
    # Calculate transform
    transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, size[0], size[1])
    
    # Export each index as GeoTIFF
    for index_name, index_data in indices.items():
        index_path = os.path.join(output_dir, f'{index_name.lower()}.tif')
        with rasterio.open(
            index_path,
            'w',
            driver='GTiff',
            height=index_data.shape[0],
            width=index_data.shape[1],
            count=1,
            dtype='float32',
            crs=RasterCRS.from_epsg(4326),
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(index_data.astype('float32'), 1)
            dst.set_band_description(1, index_name)
        
        print(f"✓ {index_name} saved: {index_path}")
    
    return indices

def export_as_netcdf(data, bbox, time_interval, output_dir="outputs", size=(512, 512)):
    """
    Export data as NetCDF format for climate/atmospheric analysis
    
    Args:
        data: Multi-band satellite data array
        bbox: Bounding box for the area of interest
        time_interval: Time interval of the data
        output_dir: Directory to save outputs
        size: Image size
    """
    print("Exporting as NetCDF...")
    
    # Create coordinate arrays
    lons = np.linspace(bbox.min_x, bbox.max_x, size[0])
    lats = np.linspace(bbox.max_y, bbox.min_y, size[1])
    
    # Band names
    band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    
    # Create xarray Dataset
    data_vars = {}
    for i, band in enumerate(band_names):
        data_vars[band] = (['y', 'x'], data[:, :, i])
    
    ds = xr.Dataset(
        data_vars,
        coords={
            'x': lons,
            'y': lats,
        },
        attrs={
            'title': 'Sentinel-2 L2A Data',
            'source': 'Sentinel Hub',
            'time_range': f"{time_interval[0]} to {time_interval[1]}",
            'crs': 'EPSG:4326',
            'bbox': f"{bbox.min_x}, {bbox.min_y}, {bbox.max_x}, {bbox.max_y}"
        }
    )
    
    # Add band attributes
    for band in band_names:
        ds[band].attrs['units'] = 'reflectance'
        ds[band].attrs['long_name'] = f'Sentinel-2 {band.upper()} band'
    
    # Save as NetCDF
    netcdf_path = os.path.join(output_dir, 'sentinel2_data.nc')
    ds.to_netcdf(netcdf_path, engine='netcdf4')
    print(f"✓ NetCDF saved: {netcdf_path}")
    
    return ds, netcdf_path

def export_statistics_csv(data, indices, output_dir="outputs"):
    """
    Export statistical summaries as CSV
    
    Args:
        data: Multi-band satellite data array
        indices: Dictionary of calculated indices
        output_dir: Directory to save outputs
    """
    print("Exporting statistics as CSV...")
    
    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    
    # Calculate statistics for bands
    band_stats = []
    for i, band_name in enumerate(band_names):
        band_data = data[:, :, i]
        valid_data = band_data[~np.isnan(band_data)]
        
        stats = {
            'Variable': band_name,
            'Mean': np.mean(valid_data),
            'Median': np.median(valid_data),
            'Std': np.std(valid_data),
            'Min': np.min(valid_data),
            'Max': np.max(valid_data),
            'Count': len(valid_data)
        }
        band_stats.append(stats)
    
    # Calculate statistics for indices
    for index_name, index_data in indices.items():
        valid_data = index_data[~np.isnan(index_data)]
        
        stats = {
            'Variable': index_name,
            'Mean': np.mean(valid_data),
            'Median': np.median(valid_data),
            'Std': np.std(valid_data),
            'Min': np.min(valid_data),
            'Max': np.max(valid_data),
            'Count': len(valid_data)
        }
        band_stats.append(stats)
    
    # Create DataFrame and save
    df = pd.DataFrame(band_stats)
    csv_path = os.path.join(output_dir, 'satellite_statistics.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Statistics CSV saved: {csv_path}")
    
    # Display statistics
    print("\nSATELLITE DATA STATISTICS:")
    print(df.round(4))
    
    return df, csv_path

def create_quicklook_images(data, indices, output_dir="outputs"):
    """
    Create quicklook images for visualization
    
    Args:
        data: Multi-band satellite data array
        indices: Dictionary of calculated indices
        output_dir: Directory to save outputs
    """
    print("Creating quicklook images...")
    
    # RGB quicklook
    rgb = data[:, :, [2, 1, 0]]  # Red, Green, Blue
    rgb_enhanced = np.clip(rgb * 3.5, 0, 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_enhanced)
    plt.title('Sentinel-2 True Color RGB')
    plt.axis('off')
    rgb_path = os.path.join(output_dir, 'quicklook_rgb.png')
    plt.savefig(rgb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ RGB quicklook saved: {rgb_path}")
    
    # False color quicklook (NIR-Red-Green)
    false_color = data[:, :, [3, 2, 1]]  # NIR, Red, Green
    false_color_enhanced = np.clip(false_color * 3.5, 0, 1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(false_color_enhanced)
    plt.title('Sentinel-2 False Color (NIR-Red-Green)')
    plt.axis('off')
    false_color_path = os.path.join(output_dir, 'quicklook_false_color.png')
    plt.savefig(false_color_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ False color quicklook saved: {false_color_path}")
    
    # NDVI quicklook
    plt.figure(figsize=(10, 10))
    im = plt.imshow(indices['NDVI'], cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    plt.title('NDVI (Normalized Difference Vegetation Index)')
    plt.colorbar(im, label='NDVI Value')
    plt.axis('off')
    ndvi_path = os.path.join(output_dir, 'quicklook_ndvi.png')
    plt.savefig(ndvi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NDVI quicklook saved: {ndvi_path}")

def create_metadata_file(bbox, time_interval, output_dir="outputs"):
    """
    Create metadata file with processing information
    
    Args:
        bbox: Bounding box of the data
        time_interval: Time interval of the data
        output_dir: Directory to save outputs
    """
    print("Creating metadata file...")
    
    metadata = {
        'Processing Information': {
            'Satellite': 'Sentinel-2',
            'Processing Level': 'L2A',
            'Time Range': f"{time_interval[0]} to {time_interval[1]}",
            'Coordinate System': 'EPSG:4326 (WGS84)',
            'Bounding Box': {
                'Min Longitude': bbox.min_x,
                'Max Longitude': bbox.max_x,
                'Min Latitude': bbox.min_y,
                'Max Latitude': bbox.max_y
            }
        },
        'Band Information': {
            'Blue (B02)': '490 nm',
            'Green (B03)': '560 nm', 
            'Red (B04)': '665 nm',
            'NIR (B08)': '842 nm',
            'SWIR1 (B11)': '1610 nm',
            'SWIR2 (B12)': '2190 nm'
        },
        'Calculated Indices': {
            'NDVI': '(NIR - Red) / (NIR + Red)',
            'EVI': '2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))',
            'NDWI': '(Green - NIR) / (Green + NIR)',
            'NDBI': '(SWIR1 - NIR) / (SWIR1 + NIR)'
        },
        'Files Generated': [
            'sentinel2_multiband.tif - All bands in single file',
            'Individual band GeoTIFFs',
            'Index GeoTIFFs (NDVI, EVI, NDWI, NDBI)',
            'sentinel2_data.nc - NetCDF format',
            'satellite_statistics.csv - Statistical summary',
            'Quicklook images (PNG format)'
        ]
    }
    
    # Save as JSON
    import json
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_path}")
    
    return metadata_path

def main():
    """Main function to run data download and export examples"""
    print("=== Satellite Data Download and Export ===\n")
    
    # Setup credentials
    config = setup_credentials()
    if config is None:
        return
    
    # Define area of interest
    bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)
    time_interval = ('2023-07-15', '2023-07-25')
    output_dir = "satellite_outputs"
    
    print(f"Area of Interest: {bbox}")
    print(f"Time Interval: {time_interval}")
    print(f"Output Directory: {output_dir}\n")
    
    print("EXPORT FORMATS:")
    print("- GeoTIFF: Standard format for GIS applications")
    print("- NetCDF: Climate and atmospheric data analysis")
    print("- CSV: Statistical summaries and tabular data")
    print("- PNG: Quicklook images for visualization\n")
    
    try:
        # Example 1: Download and export as GeoTIFF
        print("1. Downloading and Exporting as GeoTIFF")
        data, multiband_path = download_and_export_geotiff(config, bbox, time_interval, output_dir)
        print("✓ GeoTIFF export completed\n")
        
        # Example 2: Calculate and export indices
        print("2. Calculating and Exporting Vegetation Indices")
        indices = calculate_and_export_indices(data, bbox, output_dir)
        print("✓ Index calculation and export completed\n")
        
        # Example 3: Export as NetCDF
        print("3. Exporting as NetCDF")
        dataset, netcdf_path = export_as_netcdf(data, bbox, time_interval, output_dir)
        print("✓ NetCDF export completed\n")
        
        # Example 4: Export statistics as CSV
        print("4. Exporting Statistics as CSV")
        stats_df, csv_path = export_statistics_csv(data, indices, output_dir)
        print("✓ CSV export completed\n")
        
        # Example 5: Create quicklook images
        print("5. Creating Quicklook Images")
        create_quicklook_images(data, indices, output_dir)
        print("✓ Quicklook creation completed\n")
        
        # Example 6: Create metadata file
        print("6. Creating Metadata File")
        metadata_path = create_metadata_file(bbox, time_interval, output_dir)
        print("✓ Metadata creation completed\n")
        
        print("=== Data Export completed successfully! ===")
        print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")
        
        print("\nFILE USAGE GUIDE:")
        print("GeoTIFF FILES:")
        print("- Use in QGIS, ArcGIS, or any GIS software")
        print("- Can be opened with GDAL/rasterio in Python")
        print("- Suitable for spatial analysis and mapping")
        
        print("\nNetCDF FILES:")
        print("- Use with xarray, pandas, or climate analysis tools")
        print("- Compatible with atmospheric/climate models")
        print("- Self-describing format with metadata")
        
        print("\nCSV FILES:")
        print("- Open in Excel, R, or any data analysis tool")
        print("- Contains statistical summaries")
        print("- Good for reporting and documentation")
        
        print("\nQUICKLOOK IMAGES:")
        print("- PNG format for presentations and reports")
        print("- High resolution for publication quality")
        print("- Include both true color and analytical products")
        
        print("\nNext steps:")
        print("- Load GeoTIFFs in your preferred GIS software")
        print("- Use Python libraries like rasterio, xarray for analysis")
        print("- Combine with field data for validation")
        print("- Create time series by repeating for multiple dates")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your credentials and internet connection.")

if __name__ == "__main__":
    main()

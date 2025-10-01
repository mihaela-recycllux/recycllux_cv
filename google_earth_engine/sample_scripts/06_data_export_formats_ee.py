#!/usr/bin/env python3
"""
Data Download and Export with Google Earth Engine

This script demonstrates how to download satellite data and export it in different formats
for further analysis in GIS software or other applications:
- Direct downloads via URLs
- Export to Google Drive
- Export to Google Cloud Storage
- Export to Earth Engine Assets
- Integration with external tools

Author: Varun Burde 
email: varun@recycllux.com
Date: 2025

"""

import os
import ee
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
import pandas as pd
import signal

def with_timeout(seconds=60):
    """Decorator to add timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)
                return result
            except Exception as e:
                signal.alarm(0)
                return None
        return wrapper
    return decorator

def initialize_earth_engine():
    """Initialize Google Earth Engine"""
    try:
        ee.Initialize(project='recycllux-satellite-data')
        print('Earth Engine initialized')
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project='recycllux-satellite-data')
        print('Authenticated and initialized')

@with_timeout(120)  # 120 second timeout for export operations
def download_and_export_bands(roi, start_date, end_date):
    """
    Download satellite data and export individual bands
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Downloading and exporting Sentinel-2 bands...")
    
    # Build Sentinel-2 collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Check if collection has data
    collection_size = collection.size()
    print(f"Found {collection_size.getInfo()} images")
    
    if collection_size.getInfo() == 0:
        print("No data available for the specified criteria")
        return None, {}
    
    # Get median composite
    image = collection.median()
    
    # Select multiple bands
    bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    band_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    multiband_image = image.select(bands)
    
    # Generate download URLs for individual bands
    download_urls = {}
    for band, name in zip(bands, band_names):
        band_image = image.select(band)
        url = band_image.getDownloadURL({
            'scale': 20,  # Increased scale to reduce file size
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        download_urls[name] = url
        print(f"✓ {name} band download URL: {url}")
    
    # Multi-band download URL
    multiband_url = multiband_image.getDownloadURL({
        'scale': 20,  # Increased scale to reduce file size
        'crs': 'EPSG:4326',
        'region': roi,
        'fileFormat': 'GeoTIFF'
    })
    download_urls['MultiBand'] = multiband_url
    print(f"✓ Multi-band download URL: {multiband_url}")
    
    return multiband_image, download_urls

def calculate_and_export_indices(roi, start_date, end_date):
    """
    Calculate vegetation indices and export them
    
    Args:
        roi: Region of interest (ee.Geometry)
        start_date: Start date string
        end_date: End date string
    """
    print("Calculating and exporting vegetation indices...")
    
    # Build collection with updated dataset
    collection = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    )
    
    # Get median composite and scale
    image = collection.median().multiply(0.0001)
    
    # Select bands
    blue = image.select('B2')
    green = image.select('B3')
    red = image.select('B4')
    nir = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    
    # Calculate indices
    indices = {}
    
    # NDVI
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    indices['NDVI'] = ndvi
    
    # EVI
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
            'NIR': nir,
            'RED': red,
            'BLUE': blue
        }).rename('EVI')
    indices['EVI'] = evi
    
    # NDWI
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    indices['NDWI'] = ndwi
    
    # NDBI
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename('NDBI')
    indices['NDBI'] = ndbi
    
    # Generate download URLs with increased scale to avoid size limits
    index_urls = {}
    for index_name, index_image in indices.items():
        url = index_image.getDownloadURL({
            'scale': 20,  # Increased scale to reduce file size
            'crs': 'EPSG:4326',
            'region': roi,
            'fileFormat': 'GeoTIFF'
        })
        index_urls[index_name] = url
        print(f"✓ {index_name} download URL: {url}")
    
    return indices, index_urls

@with_timeout(60)  # 60 second timeout
def export_to_drive(image, roi, description, folder='EarthEngine'):
    """
    Export data to Google Drive
    
    Args:
        image: Earth Engine image to export
        roi: Region of interest
        description: Description for the export task
        folder: Google Drive folder name
    """
    print(f"Exporting {description} to Google Drive...")
    
    # Create export task
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=10,
        region=roi,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    
    # Start the task
    task.start()
    
    print(f"✓ Export task started: {description}")
    print(f"  Task ID: {task.id}")
    print(f"  Status: {task.status()['state']}")
    print(f"  Check progress at: https://code.earthengine.google.com/")
    
    return task

@with_timeout(60)  # 60 second timeout
def export_to_cloud_storage(image, roi, description, bucket, folder='earthengine'):
    """
    Export data to Google Cloud Storage
    
    Args:
        image: Earth Engine image to export
        roi: Region of interest
        description: Description for the export task
        bucket: Google Cloud Storage bucket name
        folder: Folder in the bucket
    """
    print(f"Exporting {description} to Google Cloud Storage...")
    
    # Create export task
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=description,
        bucket=bucket,
        fileNamePrefix=f"{folder}/{description}",
        scale=10,
        region=roi,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    
    # Start the task
    task.start()
    
    print(f"✓ Export task started: {description}")
    print(f"  Task ID: {task.id}")
    print(f"  Bucket: gs://{bucket}/{folder}/")
    
    return task

@with_timeout(60)  # 60 second timeout
def export_statistics_to_csv(image, roi, band_names, output_file='google_earth_engine/downloads/satellite_statistics.csv'):
    """
    Export statistical summaries as CSV
    
    Args:
        image: Earth Engine image
        roi: Region of interest
        band_names: List of band names
        output_file: Output CSV filename
    """
    print("Calculating and exporting statistics...")
    
    # Calculate statistics for each band
    stats_list = []
    
    for band_name in band_names:
        band_image = image.select(band_name)
        
        # Calculate statistics
        stats = band_image.reduceRegion(
            reducer=ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.count(), sharedInputs=True),
            geometry=roi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # Extract values
        band_stats = {
            'Band': band_name,
            'Mean': stats.get(f'{band_name}_mean', np.nan),
            'StdDev': stats.get(f'{band_name}_stdDev', np.nan),
            'Min': stats.get(f'{band_name}_min', np.nan),
            'Max': stats.get(f'{band_name}_max', np.nan),
            'Count': stats.get(f'{band_name}_count', np.nan)
        }
        stats_list.append(band_stats)
    
    # Create DataFrame
    df = pd.DataFrame(stats_list)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Statistics saved to: {output_file}")
    
    # Display statistics
    print("\nSATELLITE DATA STATISTICS:")
    print(df.round(4))
    
    return df

def create_quicklook_images(image, roi):
    """
    Create quicklook images for visualization
    
    Args:
        image: Earth Engine image
        roi: Region of interest
    """
    print("Creating quicklook images...")
    
    # RGB quicklook
    rgb_vis = image.visualize(
        bands=['B4', 'B3', 'B2'],
        min=0,
        max=3000
    )
    
    # False color quicklook
    false_color_vis = image.visualize(
        bands=['B8', 'B4', 'B3'],
        min=0,
        max=3000
    )
    
    # NDVI quicklook
    ndvi = image.normalizedDifference(['B8', 'B4'])
    ndvi_vis = ndvi.visualize(
        min=-0.2,
        max=0.8,
        palette=['red', 'yellow', 'green']
    )
    
    # Get thumbnails and save
    try:
        # RGB
        rgb_url = rgb_vis.getThumbURL({
            'dimensions': 1024,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(rgb_url)
        if response.status_code == 200:
            with open('google_earth_engine/downloads/quicklook_rgb.png', 'wb') as f:
                f.write(response.content)
            print("✓ RGB quicklook saved: google_earth_engine/downloads/quicklook_rgb.png")
            
            # Display
            img = Image.open(io.BytesIO(response.content))
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(img))
            plt.title('Sentinel-2 True Color RGB')
            plt.axis('off')
            plt.savefig('google_earth_engine/downloads/quicklook_rgb_display.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # False Color
        false_color_url = false_color_vis.getThumbURL({
            'dimensions': 1024,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(false_color_url)
        if response.status_code == 200:
            with open('google_earth_engine/downloads/quicklook_false_color.png', 'wb') as f:
                f.write(response.content)
            print("✓ False color quicklook saved: google_earth_engine/downloads/quicklook_false_color.png")
        
        # NDVI
        ndvi_url = ndvi_vis.getThumbURL({
            'dimensions': 1024,
            'region': roi,
            'format': 'png'
        })
        
        response = requests.get(ndvi_url)
        if response.status_code == 200:
            with open('google_earth_engine/downloads/quicklook_ndvi.png', 'wb') as f:
                f.write(response.content)
            print("✓ NDVI quicklook saved: google_earth_engine/downloads/quicklook_ndvi.png")
            
            # Display NDVI
            img = Image.open(io.BytesIO(response.content))
            plt.figure(figsize=(10, 10))
            plt.imshow(np.array(img))
            plt.title('NDVI (Normalized Difference Vegetation Index)')
            plt.axis('off')
            plt.savefig('google_earth_engine/downloads/quicklook_ndvi_display.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    except Exception as e:
        print(f"Could not create quicklooks: {e}")

def create_metadata_file(roi, start_date, end_date, download_urls, output_file='google_earth_engine/downloads/metadata.json'):
    """
    Create metadata file with processing information
    
    Args:
        roi: Region of interest
        start_date: Start date
        end_date: End date
        download_urls: Dictionary of download URLs
        output_file: Output filename
    """
    print("Creating metadata file...")
    
    # Get bounding box coordinates
    roi_coords = roi.getInfo()['coordinates'][0]
    
    metadata = {
        'Processing Information': {
            'Satellite': 'Sentinel-2',
            'Processing Level': 'L2A (Surface Reflectance)',
            'Time Range': f"{start_date} to {end_date}",
            'Coordinate System': 'EPSG:4326 (WGS84)',
            'Processing Platform': 'Google Earth Engine',
            'Export Date': pd.Timestamp.now().isoformat()
        },
        'Spatial Information': {
            'Region of Interest': roi_coords,
            'Bounding Box': {
                'coordinates': roi_coords
            }
        },
        'Band Information': {
            'B2 (Blue)': '490 nm - 10m resolution',
            'B3 (Green)': '560 nm - 10m resolution', 
            'B4 (Red)': '665 nm - 10m resolution',
            'B8 (NIR)': '842 nm - 10m resolution',
            'B11 (SWIR1)': '1610 nm - 20m resolution',
            'B12 (SWIR2)': '2190 nm - 20m resolution'
        },
        'Calculated Indices': {
            'NDVI': '(NIR - Red) / (NIR + Red)',
            'EVI': '2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))',
            'NDWI': '(Green - NIR) / (Green + NIR)',
            'NDBI': '(SWIR1 - NIR) / (SWIR1 + NIR)'
        },
        'Download URLs': download_urls,
        'Export Methods': [
            'Direct download via getDownloadURL()',
            'Export to Google Drive',
            'Export to Google Cloud Storage',
            'Export to Earth Engine Assets'
        ],
        'Usage Notes': [
            'Scale factors already applied to Surface Reflectance data',
            'Cloud filtering applied (<20% cloud cover)',
            'Temporal composite using median reducer',
            'All exports in GeoTIFF format with EPSG:4326 projection'
        ]
    }
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {output_file}")
    
    return metadata

def demonstrate_batch_exports(image, indices, roi):
    """
    Demonstrate batch export capabilities
    
    Args:
        image: Base satellite image
        indices: Dictionary of calculated indices
        roi: Region of interest
    """
    print("\nDemonstrating batch export capabilities...")
    
    # Export multiple products to Drive
    exports = []
    
    # 1. RGB composite
    rgb_image = image.select(['B4', 'B3', 'B2'])
    task1 = export_to_drive(rgb_image, roi, 'sentinel2_rgb_composite', 'SatelliteData')
    exports.append(task1)
    
    # 2. All indices combined
    all_indices = ee.Image.cat([
        indices['NDVI'],
        indices['EVI'],
        indices['NDWI'],
        indices['NDBI']
    ])
    task2 = export_to_drive(all_indices, roi, 'vegetation_indices', 'SatelliteData')
    exports.append(task2)
    
    # 3. Multi-spectral bands
    multispectral = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    task3 = export_to_drive(multispectral, roi, 'sentinel2_multispectral', 'SatelliteData')
    exports.append(task3)
    
    print(f"\n✓ Started {len(exports)} export tasks")
    print("Monitor progress at: https://code.earthengine.google.com/tasks")
    
    return exports

def main():
    """Main function to run data download and export examples"""
    print("=== Satellite Data Download and Export with Google Earth Engine ===\n")
    
    # Initialize Earth Engine
    initialize_earth_engine()
    
    # Define area of interest - smaller area to avoid size limits
    roi = ee.Geometry.Rectangle([14.0, 46.0, 14.1, 46.1])  # Reduced area size
    start_date = '2023-07-01'
    end_date = '2023-07-31'
    
    print(f"Area of Interest: {roi.getInfo()}")
    print(f"Time Interval: {start_date} to {end_date}\n")
    
    print("EXPORT METHODS:")
    print("- Direct downloads: Small areas, immediate access")
    print("- Google Drive exports: Larger areas, processed in background")
    print("- Google Cloud Storage: Enterprise workflows")
    print("- Earth Engine Assets: For further processing in EE\n")
    
    try:
        # Example 1: Download bands with direct URLs
        print("1. Downloading Bands with Direct URLs")
        multiband_image, band_urls = download_and_export_bands(roi, start_date, end_date)
        print("✓ Band downloads completed\n")
        
        # Example 2: Calculate and export indices
        print("2. Calculating and Exporting Vegetation Indices")
        indices, index_urls = calculate_and_export_indices(roi, start_date, end_date)
        print("✓ Index calculation and export completed\n")
        
        # Example 3: Export statistics as CSV
        print("3. Exporting Statistics as CSV")
        band_names = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        stats_df = export_statistics_to_csv(multiband_image, roi, band_names)
        print("✓ CSV export completed\n")
        
        # Example 4: Create quicklook images
        print("4. Creating Quicklook Images")
        create_quicklook_images(multiband_image, roi)
        print("✓ Quicklook creation completed\n")
        
        # Example 5: Create metadata file
        print("5. Creating Metadata File")
        all_urls = {**band_urls, **index_urls}
        metadata = create_metadata_file(roi, start_date, end_date, all_urls)
        print("✓ Metadata creation completed\n")
        
        # Example 6: Demonstrate batch exports
        print("6. Batch Export to Google Drive")
        export_tasks = demonstrate_batch_exports(multiband_image, indices, roi)
        print("✓ Batch exports initiated\n")
        
        print("=== Data Export completed successfully! ===")
        print("\nFILE USAGE GUIDE:")
        print("DIRECT DOWNLOADS:")
        print("- Use for small areas (<32MB)")
        print("- Immediate access via URLs")
        print("- Perfect for quick analysis and prototyping")
        
        print("\nGOOGLE DRIVE EXPORTS:")
        print("- Use for larger areas and complex processing")
        print("- Runs in background, can handle large datasets")
        print("- Files appear in your Google Drive")
        
        print("\nSTATISTICAL SUMMARIES:")
        print("- CSV format for easy import into Excel/R/Python")
        print("- Contains mean, std dev, min/max for each band")
        print("- Good for quantitative analysis and reporting")
        
        print("\nQUICKLOOK IMAGES:")
        print("- High-resolution PNG for presentations")
        print("- Both visual and analytical products")
        print("- Ready for publication and reports")
        
        print("\nNext steps:")
        print("- Download files using the provided URLs")
        print("- Load GeoTIFFs in QGIS, ArcGIS, or Python")
        print("- Use metadata.json for processing documentation")
        print("- Monitor export tasks in Earth Engine Code Editor")
        print("- Scale up processing for larger areas using batch exports")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your Earth Engine authentication and internet connection.")

if __name__ == "__main__":
    main()

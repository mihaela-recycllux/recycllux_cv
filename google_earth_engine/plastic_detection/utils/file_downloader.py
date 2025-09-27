#!/usr/bin/env python3
"""
File downloader utility for satellite data.

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025
"""

import os
import requests
import time
import zipfile
import shutil
import numpy as np
from tqdm import tqdm
from config.settings import Settings

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

def convert_to_8bit_geotiff(input_path, output_path):
    """
    Convert 64-bit scientific GeoTIFF to 8-bit GeoTIFF that can be opened on MacBook.
    
    Args:
        input_path: Path to input 64-bit GeoTIFF
        output_path: Path for output 8-bit GeoTIFF
    
    Returns:
        bool: True if conversion successful
    """
    try:
        if RASTERIO_AVAILABLE:
            # Use rasterio for proper georeferencing
            with rasterio.open(input_path) as src:
                # Read the data
                data = src.read(1).astype(np.float64)
                
                # Handle NaN and infinite values
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Normalize to 0-255 range
                if data.max() > data.min():
                    # Scale to 0-1 first
                    data_normalized = (data - data.min()) / (data.max() - data.min())
                    # Convert to 0-255
                    data_8bit = (data_normalized * 255).astype(np.uint8)
                else:
                    data_8bit = np.zeros_like(data, dtype=np.uint8)
                
                # Copy metadata and update for 8-bit
                profile = src.profile.copy()
                profile.update({
                    'dtype': 'uint8',
                    'count': 1,
                    'compress': 'lzw'  # Good compression for 8-bit
                })
                
                # Write 8-bit GeoTIFF
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(data_8bit, 1)
                
                return True
                
        elif TIFFFILE_AVAILABLE:
            # Fallback to tifffile (without georeferencing)
            data = tifffile.imread(input_path)
            
            # Handle NaN and infinite values
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Normalize to 0-255
            if data.max() > data.min():
                data_normalized = (data - data.min()) / (data.max() - data.min())
                data_8bit = (data_normalized * 255).astype(np.uint8)
            else:
                data_8bit = np.zeros_like(data, dtype=np.uint8)
            
            # Save as regular 8-bit TIFF
            tifffile.imwrite(output_path, data_8bit)
            return True
        
        else:
            print("  ⚠️  No library available for 8-bit conversion (need rasterio or tifffile)")
            return False
            
    except Exception as e:
        print(f"  ❌ Error converting to 8-bit: {e}")
        return False

def download_file_from_url(url, filename, output_dir=None, show_progress=True):
    """
    Download a file from URL and save it to disk.
    Handles Google Earth Engine ZIP files by extracting them automatically.
    
    Args:
        url: Download URL
        filename: Local filename to save as (should end with .tif)
        output_dir: Directory to save file (default: Settings.OUTPUT_DIR)
        show_progress: Whether to show progress bar
    
    Returns:
        str: Path to final GeoTIFF file, or None if failed
    """
    if not output_dir:
        output_dir = Settings.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download to temporary ZIP file first
    temp_zip_path = os.path.join(output_dir, f"temp_{filename}.zip")
    final_tif_path = os.path.join(output_dir, filename)
    
    try:
        print(f"Downloading {filename}...")
        
        # Start the download
        response = requests.get(url, stream=True, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download ZIP file with progress bar
        with open(temp_zip_path, 'wb') as f:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Check if downloaded file is a ZIP (Google Earth Engine format)
        if zipfile.is_zipfile(temp_zip_path):
            print(f"  📦 Extracting GeoTIFF from ZIP archive...")
            
            # Extract ZIP contents
            extract_dir = os.path.join(output_dir, f"temp_extract_{filename}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the main GeoTIFF file(s)
            extracted_files = os.listdir(extract_dir)
            tif_files = [f for f in extracted_files if f.endswith('.tif')]
            
            if len(tif_files) == 1:
                # Single band - just rename it
                src_tif = os.path.join(extract_dir, tif_files[0])
                shutil.move(src_tif, final_tif_path)
                print(f"  ✓ Extracted single-band GeoTIFF")
                
            elif len(tif_files) > 1:
                # Multi-band - for RGB, we could stack them, but for now just use the first one
                # Sort to get consistent ordering (B2, B3, B4 for RGB)
                tif_files.sort()
                src_tif = os.path.join(extract_dir, tif_files[0])  # Use first band
                shutil.move(src_tif, final_tif_path)
                print(f"  ✓ Extracted multi-band GeoTIFF (using {tif_files[0]})")
                
                # Save info about other bands
                bands_info_path = os.path.join(output_dir, f"{filename}_bands.txt")
                with open(bands_info_path, 'w') as f:
                    f.write(f"Available bands for {filename}:\n")
                    for band_file in tif_files:
                        f.write(f"  - {band_file}\n")
                
            else:
                print(f"  ✗ No GeoTIFF files found in ZIP archive")
                return None
            
            # Cleanup
            shutil.rmtree(extract_dir)
            os.remove(temp_zip_path)
            
        else:
            # Not a ZIP file, rename directly
            shutil.move(temp_zip_path, final_tif_path)
            print(f"  ✓ Saved as GeoTIFF directly")
        
        # Verify final file
        if os.path.exists(final_tif_path) and os.path.getsize(final_tif_path) > 0:
            file_size_mb = os.path.getsize(final_tif_path) / (1024 * 1024)
            print(f"✓ Downloaded: {filename} ({file_size_mb:.2f} MB)")
            print(f"  📊 High-resolution GeoTIFF ready for QGIS visualization")
            
            # Add metadata info for QGIS users
            if RASTERIO_AVAILABLE:
                try:
                    with rasterio.open(final_tif_path) as src:
                        print(f"  � Resolution: {src.width} x {src.height} pixels")
                        print(f"  🌍 CRS: {src.crs}")
                        print(f"  🎯 Data type: {src.dtypes[0]} (high precision)")
                except:
                    pass
            
            return final_tif_path
        else:
            print(f"✗ Failed: {filename} (file empty or not created)")
            return None
            
    except requests.exceptions.Timeout:
        print(f"✗ Timeout: {filename} (download took too long)")
        return None
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {filename}: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error downloading {filename}: {e}")
        # Cleanup on error
        for temp_file in [temp_zip_path, final_tif_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        return None

def download_multiple_files(url_dict, output_dir=None):
    """
    Download multiple files from URL dictionary
    
    Args:
        url_dict: Dictionary of {filename: url}
        output_dir: Directory to save files
    
    Returns:
        dict: Dictionary of {filename: filepath} for successfully downloaded files
    """
    downloaded_files = {}
    failed_downloads = []
    
    print(f"\nDownloading {len(url_dict)} files...")
    print("=" * 60)
    
    for filename, url in url_dict.items():
        # Add .tif extension if not present
        if not filename.endswith('.tif'):
            filename = f"{filename}.tif"
            
        filepath = download_file_from_url(url, filename, output_dir)
        
        if filepath:
            downloaded_files[filename] = filepath
        else:
            failed_downloads.append(filename)
        
        # Small delay between downloads
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print(f"Download Summary: {len(downloaded_files)} successful, {len(failed_downloads)} failed")
    
    if failed_downloads:
        print("Failed downloads:")
        for filename in failed_downloads:
            print(f"  ✗ {filename}")
    
    return downloaded_files
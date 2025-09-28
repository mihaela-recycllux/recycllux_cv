#!/usr/bin/env python3
"""
Earth Engine Client - Native resolution satellite data with smart tiling
"""

import ee
import os
import requests
import zipfile
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

try:
    import rasterio
    from rasterio.merge import merge
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class EarthEngineClient:
    """Earth Engine client with native resolution tiling"""
    
    def __init__(self):
        self.initialized = False
        
    def initialize(self):
        """Initialize Earth Engine connection"""
        try:
            ee.Initialize(project='recycllux-satellite-data')
            self.initialized = True
            print("✅ Earth Engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Earth Engine initialization failed: {e}")
            return False
    
    def download_area_data(self, latitude, longitude, area_km):
        """Download satellite data at native resolution using tiling"""
        if not self.initialized:
            return {}
        
        print(f"📏 Downloading {area_km}x{area_km}km at native resolution (10m/pixel)")
        
        # Smart tiling: 5km tiles to stay under Earth Engine limits
        tile_size_km = 5.0
        tiles_per_side = max(1, int(np.ceil(area_km / tile_size_km)))
        total_tiles = tiles_per_side * tiles_per_side
        
        print(f"📦 Tiling strategy: {tiles_per_side}x{tiles_per_side} = {total_tiles} tiles")
        print(f"🔍 Each tile: {tile_size_km}x{tile_size_km}km at 10m resolution")
        
        # Create base region for data availability check
        buffer_deg = self._km_to_degrees(area_km / 2)
        base_region = ee.Geometry.Rectangle([
            longitude - buffer_deg, latitude - buffer_deg,
            longitude + buffer_deg, latitude + buffer_deg
        ])
        
        # Check data availability
        s2_collection = self._get_s2_collection(base_region)
        s1_collection = self._get_s1_collection(base_region)
        
        s2_count = s2_collection.size().getInfo()
        s1_count = s1_collection.size().getInfo()
        
        print(f"📊 Available data: {s2_count} Sentinel-2 images, {s1_count} Sentinel-1 images")
        
        if s2_count == 0:
            print("❌ No Sentinel-2 data available")
            return {}
        
        # Create median composites for consistent processing
        s2_image = s2_collection.median()
        s1_image = s1_collection.median() if s1_count > 0 else None
        
        # Download products with tiling
        results = {}
        
        # Sentinel-2 products
        for product_name, bands in [('rgb', ['B4', 'B3', 'B2']), ('fdi', ['FDI']), ('fai', ['FAI'])]:
            print(f"\n🔄 Processing {product_name.upper()}...")
            merged_file = self._download_and_merge_tiles(
                s2_image, product_name, bands, latitude, longitude, 
                area_km, tile_size_km, tiles_per_side
            )
            if merged_file:
                results[product_name] = merged_file
        
        # Sentinel-1 products (if available)
        if s1_image is not None:
            for product_name, bands in [('vv', ['VV']), ('vh', ['VH'])]:
                print(f"\n🔄 Processing SAR {product_name.upper()}...")
                merged_file = self._download_and_merge_tiles(
                    s1_image, product_name, bands, latitude, longitude,
                    area_km, tile_size_km, tiles_per_side
                )
                if merged_file:
                    results[product_name] = merged_file
        
        print(f"\n✅ Downloaded {len(results)} products: {list(results.keys())}")
        return results
    
    def _get_s2_collection(self, region):
        """Get processed Sentinel-2 collection"""
        return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterDate('2023-01-01', '2024-12-31')
                .filterBounds(region)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(self._mask_clouds_s2)
                .map(self._calculate_indices))
    
    def _get_s1_collection(self, region):
        """Get processed Sentinel-1 collection"""
        return (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterDate('2023-01-01', '2024-12-31')
                .filterBounds(region)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                .map(self._preprocess_s1))
    
    def _download_and_merge_tiles(self, image, product_name, bands, center_lat, center_lon, 
                                 area_km, tile_size_km, tiles_per_side):
        """Download tiles and merge them into final product"""
        
        tile_files = []
        
        # Download each tile
        for i in range(tiles_per_side):
            for j in range(tiles_per_side):
                # Calculate tile bounds
                tile_offset = area_km / tiles_per_side
                start_offset = -area_km / 2
                
                lat_min = center_lat + self._km_to_degrees(start_offset + i * tile_offset)
                lat_max = center_lat + self._km_to_degrees(start_offset + (i + 1) * tile_offset)
                lon_min = center_lon + self._km_to_degrees(start_offset + j * tile_offset)
                lon_max = center_lon + self._km_to_degrees(start_offset + (j + 1) * tile_offset)
                
                # Create tile region
                tile_region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
                
                # Download tile
                tile_file = self._download_single_tile(
                    image, bands, tile_region, f"{product_name}_tile_{i}_{j}"
                )
                
                if tile_file:
                    tile_files.append(tile_file)
                    print(f"  ✅ Tile {i+1},{j+1}")
                else:
                    print(f"  ❌ Tile {i+1},{j+1}")
        
        # Merge tiles if we got any
        if tile_files:
            return self._merge_tile_files(tile_files, product_name, center_lat, center_lon, area_km)
        else:
            print(f"  ❌ No tiles downloaded for {product_name}")
            return None
    
    def _download_single_tile(self, image, bands, region, filename_prefix):
        """Download a single tile"""
        try:
            # Select bands and clip to region
            tile_image = image.select(bands).clip(region)
            
            # Get download URL with native resolution
            url = tile_image.getDownloadURL({
                'scale': 10,  # Native 10m resolution
                'crs': 'EPSG:4326',
                'region': region,
                'fileFormat': 'GeoTIFF',
                'maxPixels': 1e8
            })
            
            # Download file
            return self._download_file_from_url(url, f"{filename_prefix}.tif")
            
        except Exception as e:
            print(f"    ⚠️  Tile download failed: {e}")
            return None
    
    def _merge_tile_files(self, tile_files, product_name, lat, lon, area_km):
        """Merge multiple tile files into one"""
        if not RASTERIO_AVAILABLE or not tile_files:
            return tile_files[0] if tile_files else None
        
        try:
            # Open all tiles
            src_files = []
            for tile_path in tile_files:
                if os.path.exists(tile_path):
                    src_files.append(rasterio.open(tile_path))
            
            if not src_files:
                return None
            
            # Merge tiles
            mosaic, out_transform = merge(src_files)
            
            # Create output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'{product_name}_{area_km}km_{lat:.4f}_{lon:.4f}_{timestamp}.tif'
            output_path = Path('results') / output_filename
            output_path.parent.mkdir(exist_ok=True)
            
            # Write merged file
            profile = src_files[0].profile.copy()
            profile.update({
                'height': mosaic.shape[1] if len(mosaic.shape) > 2 else mosaic.shape[0],
                'width': mosaic.shape[2] if len(mosaic.shape) > 2 else mosaic.shape[1],
                'transform': out_transform,
                'compress': 'lzw'
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                if len(mosaic.shape) == 3:
                    dst.write(mosaic)
                else:
                    dst.write(mosaic, 1)
            
            # Cleanup
            for src in src_files:
                src.close()
            for tile_path in tile_files:
                try:
                    os.remove(tile_path)
                except:
                    pass
            
            # Verify result
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"  🎯 Merged: {size_mb:.1f}MB")
                return str(output_path)
            
        except Exception as e:
            print(f"  ❌ Merge failed: {e}")
            return tile_files[0] if tile_files else None
    
    def _download_file_from_url(self, url, filename):
        """Download file from URL"""
        output_path = Path('results') / filename
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Handle ZIP files from Earth Engine
            if response.headers.get('content-type', '').startswith('application/zip'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_zip.write(chunk)
                
                # Extract GeoTIFF
                with zipfile.ZipFile(tmp_zip.name, 'r') as zip_ref:
                    tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
                    if tif_files:
                        zip_ref.extract(tif_files[0], output_path.parent)
                        extracted = output_path.parent / tif_files[0]
                        extracted.rename(output_path)
                
                os.unlink(tmp_zip.name)
            else:
                # Direct download
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            return str(output_path) if output_path.exists() else None
            
        except Exception as e:
            print(f"    ❌ Download error: {e}")
            return None
    
    def _mask_clouds_s2(self, image):
        """Mask clouds in Sentinel-2"""
        qa = image.select('QA60')
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return image.updateMask(cloud_mask)
    
    def _calculate_indices(self, image):
        """Calculate FDI and FAI indices"""
        # Scale to reflectance
        img = image.multiply(0.0001)
        
        # FDI calculation
        nir = img.select('B8')
        red = img.select('B4') 
        swir = img.select('B11')
        fdi = nir.subtract(red).divide(nir.add(red)).subtract(
              nir.subtract(swir).divide(nir.add(swir)))
        
        # FAI calculation
        fai = nir.subtract(red.add(swir.subtract(red).multiply(
              (832.8 - 664.6) / (1613.7 - 664.6))))
        
        return image.addBands([fdi.rename('FDI'), fai.rename('FAI')])
    
    def _preprocess_s1(self, image):
        """Preprocess Sentinel-1 SAR data"""
        return image.select(['VV', 'VH'])
    
    def _km_to_degrees(self, km):
        """Convert kilometers to degrees (approximate)"""
        return km / 111.0
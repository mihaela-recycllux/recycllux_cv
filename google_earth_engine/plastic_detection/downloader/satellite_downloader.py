#!/usr/bin/env python3
"""
Satellite data downloader for plastic detection project.

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025
"""

import ee
import os
import requests
from datetime import datetime
from config.settings import Settings
from utils.ee_utils import (
    with_timeout, 
    get_safe_download_url, 
    mask_clouds_sentinel2,
    scale_sentinel2,
    calculate_spectral_indices,
    print_collection_stats,
    create_timestamp
)

class SatelliteDownloader:
    """Base class for satellite data downloading"""
    
    def __init__(self, roi, start_date, end_date, output_dir=None):
        self.roi = roi
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir or Settings.OUTPUT_DIR
        self.timestamp = create_timestamp()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_filename(self, product_type, date_str=""):
        """Generate standardized filename"""
        if not date_str:
            date_str = self.timestamp
        return f"{product_type}_{date_str}.tif"

class Sentinel2Downloader(SatelliteDownloader):
    """Download Sentinel-2 optical data"""
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_rgb(self):
        """Download RGB true color image"""
        print("Downloading Sentinel-2 RGB...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Settings.SENTINEL2_CLOUD_THRESHOLD))
            .map(mask_clouds_sentinel2)
            .map(scale_sentinel2)
        )
        
        print_collection_stats(collection, "Sentinel-2 RGB")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-2 data available for RGB")
            return None, None
            
        # Get median composite
        image = collection.median()
        rgb_image = image.select(['B4', 'B3', 'B2'])
        
        # Generate download URL
        filename = self._generate_filename('rgb')
        download_url = get_safe_download_url(
            rgb_image, self.roi, Settings.SENTINEL2_SCALE
        )
        
        if download_url:
            print(f"✓ RGB download URL generated: {filename}")
            return rgb_image, download_url
        return None, None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)  
    def download_false_color(self):
        """Download false color (NIR-Red-Green) image"""
        print("Downloading Sentinel-2 False Color...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Settings.SENTINEL2_CLOUD_THRESHOLD))
            .map(mask_clouds_sentinel2)
            .map(scale_sentinel2)
        )
        
        print_collection_stats(collection, "Sentinel-2 False Color")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-2 data available for false color")
            return None, None
            
        # Get median composite
        image = collection.median()
        false_color_image = image.select(['B8', 'B4', 'B3'])
        
        # Generate download URL
        filename = self._generate_filename('false_color')
        download_url = get_safe_download_url(
            false_color_image, self.roi, Settings.SENTINEL2_SCALE
        )
        
        if download_url:
            print(f"✓ False color download URL generated: {filename}")
            return false_color_image, download_url
        return None, None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_ndvi(self):
        """Download NDVI (Normalized Difference Vegetation Index)"""
        print("Downloading Sentinel-2 NDVI...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Settings.SENTINEL2_CLOUD_THRESHOLD))
            .map(mask_clouds_sentinel2)
            .map(scale_sentinel2)
            .map(calculate_spectral_indices)
        )
        
        print_collection_stats(collection, "Sentinel-2 NDVI")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-2 data available for NDVI")
            return None, None
            
        # Get median composite
        image = collection.median()
        ndvi_image = image.select('NDVI')
        
        # Generate download URL
        filename = self._generate_filename('ndvi')
        download_url = get_safe_download_url(
            ndvi_image, self.roi, Settings.SENTINEL2_SCALE
        )
        
        if download_url:
            print(f"✓ NDVI download URL generated: {filename}")
            return ndvi_image, download_url
        return None, None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_water_indices(self):
        """Download water detection indices (NDWI, MNDWI)"""
        print("Downloading Sentinel-2 Water Indices...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Settings.SENTINEL2_CLOUD_THRESHOLD))
            .map(mask_clouds_sentinel2)
            .map(scale_sentinel2)
            .map(calculate_spectral_indices)
        )
        
        print_collection_stats(collection, "Sentinel-2 Water Indices")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-2 data available for water indices")
            return None, None, None
            
        # Get median composite
        image = collection.median()
        ndwi_image = image.select('NDWI')
        mndwi_image = image.select('MNDWI')
        
        # Generate download URLs
        ndwi_filename = self._generate_filename('ndwi')
        mndwi_filename = self._generate_filename('mndwi')
        
        ndwi_url = get_safe_download_url(
            ndwi_image, self.roi, Settings.SENTINEL2_SCALE
        )
        mndwi_url = get_safe_download_url(
            mndwi_image, self.roi, Settings.SENTINEL2_SCALE
        )
        
        urls = {}
        if ndwi_url:
            urls['ndwi'] = ndwi_url
            print(f"✓ NDWI download URL generated: {ndwi_filename}")
        if mndwi_url:
            urls['mndwi'] = mndwi_url
            print(f"✓ MNDWI download URL generated: {mndwi_filename}")
            
        return ndwi_image, mndwi_image, urls
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_plastic_detection_indices(self):
        """Download indices specifically useful for plastic detection"""
        print("Downloading Plastic Detection Indices...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', Settings.SENTINEL2_CLOUD_THRESHOLD))
            .map(mask_clouds_sentinel2)
            .map(scale_sentinel2)
            .map(calculate_spectral_indices)
        )
        
        print_collection_stats(collection, "Plastic Detection Indices")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-2 data available for plastic detection")
            return None, None
            
        # Get median composite
        image = collection.median()
        fdi_image = image.select('FDI')  # Floating Debris Index
        fai_image = image.select('FAI')  # Floating Algae Index
        
        # Generate download URLs
        fdi_filename = self._generate_filename('fdi')
        fai_filename = self._generate_filename('fai')
        
        fdi_url = get_safe_download_url(
            fdi_image, self.roi, Settings.SENTINEL2_SCALE
        )
        fai_url = get_safe_download_url(
            fai_image, self.roi, Settings.SENTINEL2_SCALE
        )
        
        urls = {}
        if fdi_url:
            urls['fdi'] = fdi_url
            print(f"✓ FDI download URL generated: {fdi_filename}")
        if fai_url:
            urls['fai'] = fai_url
            print(f"✓ FAI download URL generated: {fai_filename}")
            
        return fdi_image, fai_image, urls
    
    # Individual Index Methods (wrappers for composite methods)
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_ndwi(self):
        """Download NDWI index only"""
        print("Downloading NDWI...")
        ndwi_image, _, urls = self.download_water_indices()
        return ndwi_image, urls.get('ndwi') if urls else None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)  
    def download_mndwi(self):
        """Download MNDWI index only"""
        print("Downloading MNDWI...")
        _, mndwi_image, urls = self.download_water_indices()
        return mndwi_image, urls.get('mndwi') if urls else None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_fdi(self):
        """Download FDI (Floating Debris Index) only"""
        print("Downloading FDI...")
        fdi_image, _, urls = self.download_plastic_detection_indices()
        return fdi_image, urls.get('fdi') if urls else None
        
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_fai(self):
        """Download FAI (Floating Algae Index) only"""
        print("Downloading FAI...")
        _, fai_image, urls = self.download_plastic_detection_indices() 
        return fai_image, urls.get('fai') if urls else None

class Sentinel1Downloader(SatelliteDownloader):
    """Download Sentinel-1 SAR data"""
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_vv(self):
        """Download VV polarization"""
        print("Downloading Sentinel-1 VV...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.eq('instrumentMode', Settings.SENTINEL1_INSTRUMENT_MODE))
            .select('VV')
        )
        
        print_collection_stats(collection, "Sentinel-1 VV")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-1 VV data available")
            return None, None
            
        # Get median composite  
        image = collection.median()
        
        # Generate download URL
        filename = self._generate_filename('vv')
        download_url = get_safe_download_url(
            image, self.roi, Settings.SENTINEL1_SCALE
        )
        
        if download_url:
            print(f"✓ VV download URL generated: {filename}")
            return image, download_url
        return None, None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_vh(self):
        """Download VH polarization"""
        print("Downloading Sentinel-1 VH...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', Settings.SENTINEL1_INSTRUMENT_MODE))
            .select('VH')
        )
        
        print_collection_stats(collection, "Sentinel-1 VH")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-1 VH data available")
            return None, None
            
        # Get median composite
        image = collection.median()
        
        # Generate download URL
        filename = self._generate_filename('vh')
        download_url = get_safe_download_url(
            image, self.roi, Settings.SENTINEL1_SCALE
        )
        
        if download_url:
            print(f"✓ VH download URL generated: {filename}")
            return image, download_url
        return None, None
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_both_polarizations(self):
        """Download both VV and VH polarizations"""
        print("Downloading Sentinel-1 VV and VH...")
        
        collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', Settings.SENTINEL1_INSTRUMENT_MODE))
            .select(['VV', 'VH'])
        )
        
        print_collection_stats(collection, "Sentinel-1 VV/VH")
        
        if collection.size().getInfo() == 0:
            print("No Sentinel-1 dual-pol data available")
            return None, None, None
            
        # Get median composite
        image = collection.median()
        vv_image = image.select('VV')
        vh_image = image.select('VH')
        
        # Generate download URLs
        vv_filename = self._generate_filename('vv')
        vh_filename = self._generate_filename('vh')
        
        vv_url = get_safe_download_url(
            vv_image, self.roi, Settings.SENTINEL1_SCALE
        )
        vh_url = get_safe_download_url(
            vh_image, self.roi, Settings.SENTINEL1_SCALE
        )
        
        urls = {}
        if vv_url:
            urls['vv'] = vv_url
            print(f"✓ VV download URL generated: {vv_filename}")
        if vh_url:
            urls['vh'] = vh_url
            print(f"✓ VH download URL generated: {vh_filename}")
            
        return vv_image, vh_image, urls

class LandsatDownloader(SatelliteDownloader):
    """Download Landsat 8/9 data"""
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_rgb(self):
        """Download Landsat RGB image"""
        print("Downloading Landsat RGB...")
        
        # Combine Landsat 8 and 9 collections
        l8_collection = (
            ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUD_COVER', Settings.SENTINEL2_CLOUD_THRESHOLD))
        )
        
        l9_collection = (
            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
            .filter(ee.Filter.lt('CLOUD_COVER', Settings.SENTINEL2_CLOUD_THRESHOLD))
        )
        
        # Merge collections
        collection = l8_collection.merge(l9_collection)
        
        print_collection_stats(collection, "Landsat RGB")
        
        if collection.size().getInfo() == 0:
            print("No Landsat data available")
            return None, None
            
        # Apply scaling and get median
        def scale_landsat(image):
            return image.multiply(0.0000275).add(-0.2).copyProperties(image, ['system:time_start'])
            
        image = collection.map(scale_landsat).median()
        rgb_image = image.select(['SR_B4', 'SR_B3', 'SR_B2'])
        
        # Generate download URL
        filename = self._generate_filename('landsat_rgb')
        download_url = get_safe_download_url(
            rgb_image, self.roi, Settings.LANDSAT_SCALE
        )
        
        if download_url:
            print(f"✓ Landsat RGB download URL generated: {filename}")
            return rgb_image, download_url
        return None, None

class MODISDownloader(SatelliteDownloader):
    """Download MODIS data"""
    
    @with_timeout(Settings.PROCESSING_TIMEOUT)
    def download_rgb(self):
        """Download MODIS RGB image"""
        print("Downloading MODIS RGB...")
        
        collection = (
            ee.ImageCollection('MODIS/061/MOD09A1')
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.roi)
        )
        
        print_collection_stats(collection, "MODIS RGB")
        
        if collection.size().getInfo() == 0:
            print("No MODIS data available")
            return None, None
            
        # Get median composite
        image = collection.median()
        rgb_image = image.select(['sur_refl_b01', 'sur_refl_b04', 'sur_refl_b03'])
        
        # Generate download URL
        filename = self._generate_filename('modis_rgb')
        download_url = get_safe_download_url(
            rgb_image, self.roi, Settings.MODIS_SCALE
        )
        
        if download_url:
            print(f"✓ MODIS RGB download URL generated: {filename}")
            return rgb_image, download_url
        return None, None
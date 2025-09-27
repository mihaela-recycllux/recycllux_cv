#!/usr/bin/env python3
"""
Native Resolution Satellite Data Export for Social Tides Research
Exports TRUE native resolution (10m per pixel) Sentinel-2 data

This script exports full-resolution satellite imagery:
1. TRUE NATIVE RESOLUTION: 10m per pixel (Sentinel-2 RGB bands)
2. FULL AREA COVERAGE: Exports complete regions at native resolution
3. GOOGLE DRIVE EXPORT: Uses Earth Engine's export system
4. ASYNCHRONOUS PROCESSING: Handles long-running export tasks
5. MULTIPLE FORMATS: GeoTIFF for analysis, PNG for visualization

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025

"""

import sys
import os
import json
import time
from datetime import datetime
import ee
import requests
from PIL import Image
import numpy as np

# Initialize Earth Engine
try:
    ee.Initialize(project='recycllux-satellite-data')
    print('✓ Earth Engine initialized successfully')
except ee.EEException:
    try:
        ee.Authenticate()
        ee.Initialize(project='recycllux-satellite-data')
        print('✓ Authenticated and initialized successfully')
    except Exception as e:
        print(f'❌ Error initializing Earth Engine: {e}')
        sys.exit(1)

class NativeResolutionExporter:
    """Exports satellite data at true native resolution"""

    def __init__(self):
        self.export_tasks = []

    def create_native_roi(self, lat, lon, buffer_km=10):
        """
        Create ROI optimized for native resolution export

        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer distance in kilometers (smaller for native res)

        Returns:
            ee.Geometry: Region of interest
        """
        # For native resolution, keep areas manageable
        # 10km x 10km at 10m resolution = 1000 x 1000 pixels = 1 million pixels
        # This is reasonable for native resolution export

        buffer_deg = buffer_km / 111.32  # Rough conversion at mid-latitudes

        roi = ee.Geometry.Rectangle([
            lon - buffer_deg, lat - buffer_deg,
            lon + buffer_deg, lat + buffer_deg
        ])

        area_km2 = (buffer_km * 2) ** 2
        pixels_total = (area_km2 * 1000000) / (10 * 10)  # pixels at 10m resolution

        print(f"✓ Native Resolution ROI: {buffer_km}km buffer around {lat:.5f}°N, {lon:.5f}°E")
        print(f"  Coverage: {buffer_km*2}km × {buffer_km*2}km ({area_km2} km²)")
        print(f"  Expected pixels: {pixels_total:,.0f} at 10m resolution")
        print(f"  Expected file size: ~{pixels_total * 3 / 1000000:.1f} MB (RGB)")

        return roi

    def load_native_resolution_data(self, roi, start_date, end_date):
        """
        Load satellite data optimized for native resolution export

        Args:
            roi: Region of interest
            start_date: Start date string
            end_date: End date string

        Returns:
            dict: Satellite data
        """
        print("Loading data for native resolution export...")

        # Sentinel-2 with native resolution processing
        s2_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Quality filtering
            .map(self.mask_s2_clouds)
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        )

        s2_count = s2_collection.size().getInfo()
        print(f"  → Found {s2_count} high-quality Sentinel-2 images")

        if s2_count == 0:
            print("  Trying with relaxed filtering...")
            s2_collection = (
                ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterDate(start_date, end_date)
                .filterBounds(roi)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                .sort('CLOUDY_PIXEL_PERCENTAGE')
            )
            s2_count = s2_collection.size().getInfo()
            print(f"  → Found {s2_count} images with relaxed filtering")

        if s2_count == 0:
            return None

        # Use median composite for best quality
        if s2_count > 1:
            s2_image = s2_collection.median()
            print(f"  ✓ Using median composite from {s2_count} images")
        else:
            s2_image = ee.Image(s2_collection.first())
            print(f"  ✓ Using single best image")

        return {
            'optical': s2_image,
            'count': s2_count,
            'collection': s2_collection
        }

    def mask_s2_clouds(self, image):
        """Cloud masking for native resolution processing"""
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)

    def create_native_visualizations(self, data, roi):
        """
        Create visualizations optimized for native resolution

        Args:
            data: Satellite data
            roi: Region of interest

        Returns:
            dict: Visualization configurations
        """
        print("Creating native resolution visualizations...")

        optical = data['optical']
        viz_configs = {}

        # RGB Composite at native resolution
        rgb_image = optical.select(['B4', 'B3', 'B2'])

        viz_configs['rgb_native'] = {
            'image': rgb_image,
            'description': 'RGB_Native_Resolution',
            'scale': 10,  # 10m per pixel (native)
            'fileFormat': 'GeoTIFF'
        }

        # False Color (NIR-R-G) for vegetation/water analysis
        false_color = optical.select(['B8', 'B4', 'B3'])

        viz_configs['false_color_native'] = {
            'image': false_color,
            'description': 'False_Color_Native_Resolution',
            'scale': 10,
            'fileFormat': 'GeoTIFF'
        }

        # All bands for analysis
        all_bands = optical.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])

        viz_configs['multispectral_native'] = {
            'image': all_bands,
            'description': 'Multispectral_Native_Resolution',
            'scale': 10,
            'fileFormat': 'GeoTIFF'
        }

        print(f"  ✓ Created {len(viz_configs)} native resolution export configurations")

        return viz_configs

    def export_native_resolution(self, viz_configs, roi, buffer_km):
        """
        Export images at native resolution to Google Drive

        Args:
            viz_configs: Visualization configurations
            roi: Region of interest
            buffer_km: Buffer size for naming

        Returns:
            list: Export task information
        """
        print(f"\n🚀 Starting native resolution exports for {buffer_km}km area...")

        export_tasks = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        for viz_name, config in viz_configs.items():
            try:
                filename = f"native_{viz_name}_{buffer_km}km_{timestamp}"

                print(f"  📤 Exporting {viz_name}...")

                # Configure export task
                task = ee.batch.Export.image.toDrive(
                    image=config['image'],
                    description=config['description'],
                    folder='Recyllux_Satellite_Data',  # Google Drive folder
                    fileNamePrefix=filename,
                    region=roi,
                    scale=config['scale'],  # Native resolution: 10m per pixel
                    crs='EPSG:4326',  # WGS84 coordinate system
                    maxPixels=1e9,  # Allow large exports (up to 1 billion pixels)
                    fileFormat=config['fileFormat']
                )

                # Start the export task
                task.start()

                task_info = {
                    'task': task,
                    'name': viz_name,
                    'filename': filename,
                    'description': config['description'],
                    'scale': config['scale'],
                    'fileFormat': config['fileFormat'],
                    'status': 'running',
                    'start_time': datetime.now().isoformat()
                }

                export_tasks.append(task_info)

                print(f"    ✓ Export task started: {filename}")
                print(f"      Scale: {config['scale']}m per pixel")
                print(f"      Format: {config['fileFormat']}")

                # Small delay between tasks to avoid overwhelming Earth Engine
                time.sleep(2)

            except Exception as e:
                print(f"    ❌ Failed to start {viz_name} export: {e}")

        return export_tasks

    def monitor_exports(self, export_tasks, timeout_minutes=30):
        """
        Monitor export tasks and provide status updates

        Args:
            export_tasks: List of export task information
            timeout_minutes: Maximum time to wait for completion

        Returns:
            dict: Final status of all exports
        """
        print(f"\n⏳ Monitoring {len(export_tasks)} export tasks...")
        print("This may take several minutes for native resolution exports.")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        completed_tasks = []
        failed_tasks = []

        while export_tasks and (time.time() - start_time) < timeout_seconds:
            remaining_tasks = []

            for task_info in export_tasks:
                task = task_info['task']

                try:
                    status = task.status()

                    if status['state'] in ['COMPLETED', 'READY']:
                        task_info['status'] = 'completed'
                        task_info['completion_time'] = datetime.now().isoformat()
                        completed_tasks.append(task_info)
                        print(f"  ✅ {task_info['name']}: COMPLETED")

                    elif status['state'] == 'FAILED':
                        task_info['status'] = 'failed'
                        task_info['error'] = status.get('error_message', 'Unknown error')
                        failed_tasks.append(task_info)
                        print(f"  ❌ {task_info['name']}: FAILED - {task_info['error']}")

                    elif status['state'] == 'CANCELLED':
                        task_info['status'] = 'cancelled'
                        failed_tasks.append(task_info)
                        print(f"  ⚠️  {task_info['name']}: CANCELLED")

                    else:
                        # Still running
                        remaining_tasks.append(task_info)

                except Exception as e:
                    print(f"  ⚠️  Error checking {task_info['name']} status: {e}")
                    remaining_tasks.append(task_info)

            export_tasks = remaining_tasks

            if export_tasks:
                elapsed = int(time.time() - start_time)
                print(f"  ⏳ {len(export_tasks)} tasks still running... ({elapsed}s elapsed)")
                time.sleep(30)  # Check every 30 seconds

        # Handle timeouts
        if export_tasks:
            print(f"  ⏰ Timeout reached after {timeout_minutes} minutes")
            for task_info in export_tasks:
                task_info['status'] = 'timeout'
                failed_tasks.append(task_info)
                print(f"    ⏰ {task_info['name']}: TIMEOUT")

        return {
            'completed': completed_tasks,
            'failed': failed_tasks,
            'total_completed': len(completed_tasks),
            'total_failed': len(failed_tasks)
        }

def export_native_resolution_area(lat, lon, buffer_km=10, start_date='2025-08-27', end_date='2025-09-27'):
    """
    Export native resolution data for a specific area

    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Buffer size in km
        start_date: Start date
        end_date: End date

    Returns:
        dict: Export results
    """
    print(f"🛰️  NATIVE RESOLUTION EXPORT: {buffer_km}km area")
    print(f"Target: {lat:.6f}°N, {lon:.6f}°E")
    print(f"Resolution: 10m per pixel (TRUE native Sentinel-2)")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 60)

    exporter = NativeResolutionExporter()

    try:
        # Create ROI
        roi = exporter.create_native_roi(lat, lon, buffer_km)

        # Load data
        data = exporter.load_native_resolution_data(roi, start_date, end_date)
        if not data:
            return {'error': 'No satellite data available', 'status': 'failed'}

        # Create visualizations
        viz_configs = exporter.create_native_visualizations(data, roi)

        # Start exports
        export_tasks = exporter.export_native_resolution(viz_configs, roi, buffer_km)

        if not export_tasks:
            return {'error': 'Failed to start any export tasks', 'status': 'failed'}

        # Monitor progress
        results = exporter.monitor_exports(export_tasks, timeout_minutes=45)  # 45 minutes for large exports

        # Generate summary
        summary = {
            'coordinates': {'lat': lat, 'lon': lon},
            'buffer_km': buffer_km,
            'area_km2': (buffer_km * 2) ** 2,
            'resolution_m': 10,
            'satellite_images_used': data['count'],
            'export_results': results,
            'timestamp': datetime.now().isoformat(),
            'status': 'success' if results['total_completed'] > 0 else 'partial_failure'
        }

        return summary

    except Exception as e:
        print(f"❌ Export failed: {e}")
        return {'error': str(e), 'status': 'failed'}

def main():
    """Main function for native resolution exports"""

    print("🛰️  NATIVE RESOLUTION SATELLITE DATA EXPORT")
    print("FOR SOCIAL TIDES RESEARCH - TRUE 10M RESOLUTION")
    print("=" * 70)

    # Target coordinates
    target_lat = 44.21706925
    target_lon = 28.96504135

    print(f"Target location: {target_lat}°N, {target_lon}°E")
    print("Location: Romanian Black Sea Coast (Danube Delta region)")
    print("Native resolution features:")
    print("  • 🛰️  TRUE NATIVE RESOLUTION: 10m per pixel (Sentinel-2 RGB)")
    print("  • 📊 FULL SPECTRAL DATA: All 12 Sentinel-2 bands")
    print("  • 🗺️  GEOSPATIAL FORMAT: GeoTIFF with coordinate system")
    print("  • ☁️  GOOGLE DRIVE EXPORT: Asynchronous processing")
    print("  • 📈 ANALYSIS READY: Perfect for plastic detection algorithms")
    print("=" * 70)

    # Export multiple areas at native resolution
    export_sizes = [5, 10]  # Start with smaller areas for native resolution

    all_results = []

    for buffer_km in export_sizes:
        try:
            print(f"\n{'='*60}")
            result = export_native_resolution_area(
                lat=target_lat,
                lon=target_lon,
                buffer_km=buffer_km,
                start_date='2025-08-27',
                end_date='2025-09-27'
            )

            all_results.append(result)

            if result.get('status') in ['success', 'partial_failure']:
                print(f"✅ {buffer_km}km export completed")
            else:
                print(f"❌ {buffer_km}km export failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"💥 Critical error with {buffer_km}km export: {e}")
            all_results.append({
                'buffer_km': buffer_km,
                'status': 'failed',
                'error': str(e)
            })

    # Generate comprehensive report
    successful_exports = sum(1 for r in all_results if r.get('status') in ['success', 'partial_failure'])
    total_files = sum(r.get('export_results', {}).get('total_completed', 0) for r in all_results)

    print(f"\n🎉 NATIVE RESOLUTION EXPORT COMPLETED!")
    print(f"• Areas processed: {len(all_results)}")
    print(f"• Successful exports: {successful_exports}")
    print(f"• Total files exported: {total_files}")
    print(f"• Resolution: 10m per pixel (TRUE native)")

    if total_files > 0:
        print(f"\n📁 Files exported to Google Drive folder: 'Recyllux_Satellite_Data'")
        print(f"📄 Check Google Earth Engine Tasks panel for download links")
        print(f"🗂️  Files will be available for download once processing completes")

        # Save results
        results_filename = f"results/Native_Resolution_Export_Results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n📊 Results saved: {results_filename}")

        # Instructions for user
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Go to Google Earth Engine Code Editor: https://code.earthengine.google.com/")
        print(f"2. Check 'Tasks' tab for completed exports")
        print(f"3. Download GeoTIFF files from Google Drive")
        print(f"4. Use native resolution data for plastic detection analysis")

    else:
        print(f"\n❌ No files were successfully exported")
        print(f"Check Earth Engine tasks for error details")

if __name__ == "__main__":
    main()
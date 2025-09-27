#!/usr/bin/env python3
"""
Main entry point for the plastic detection syste        # Get region geometry
        region_map = {
            'great_pacific'                    try:
                        print(f"  🔄 Generating {product} download URL...")
                        method = getattr(downloader, met            # Create collage - for now just create a simple summary since the visualizer needs EE images
            print("⚠️  Advanced collage creation requires Earth Engine images")
            print("📝 Creating analysis summary instead...")
            collage_path = self._create_simple_collage())
                        result = method()
                        print(f"  DEBUG: result type={type(result)}, result={result is not None}")
                        if result:
                            print(f"  DEBUG: result[0]={result[0] is not None}, result[1]={result[1] is not None}")
                        if result and result[0] is not None and result[1] is not None:  # (image, url) tuple
                            image, url = result
                            filename = f"{product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
                            download_info = {
                                'url': url,
                                'filename': filename,
                                'product': product,
                                'satellite': satellite_name
                            }
                            download_urls.append(download_info)
                            print(f"  ✅ {product} URL generated - added to list")
                        else:
                            print(f"  ⚠️  No data available for {product}")
                    except Exception as e:
                        print(f"  ❌ Error generating {product} URL: {e}")cific_Patch',
            'mediterranean': 'Mediterranean', 
            'caribbean': 'Caribbean',
            'north_atlantic': 'Mediterranean'  # Fallback to Mediterranean
        }
        
        region_key = region_map.get(region, 'Mediterranean')
        if region_key in self.config.PLASTIC_HOTSPOT_COORDS:
            region_coords = self.config.PLASTIC_HOTSPOT_COORDS[region_key]
            # region_coords is already [min_lon, min_lat, max_lon, max_lat]
            region_geometry = create_roi_from_coordinates(*region_coords)
        else:
            print(f"Warning: Region {region} not found, using Mediterranean")
            region_coords = self.config.PLASTIC_HOTSPOT_COORDS['Mediterranean']
            region_geometry = create_roi_from_coordinates(*region_coords)Google Earth Engine.
Comprehensive workflow: Download → Analysis → Visualization
"""

import os
import sys
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings, PlasticDetectionConfig
from downloader.satellite_downloader import Sentinel2Downloader, Sentinel1Downloader, LandsatDownloader, MODISDownloader
from visualizer.visualization import SatelliteVisualizer
from utils.ee_utils import initialize_earth_engine, create_roi_from_coordinates
from utils.file_downloader import download_file_from_url


class PlasticDetectionWorkflow:
    """Main workflow class for plastic detection pipeline."""
    
    def __init__(self, output_dir: str = "./outputs"):
        """Initialize the workflow."""
        self.output_dir = output_dir
        self.downloaded_files = []
        self.analysis_results = {}
        
        # Initialize settings and config
        self.settings = Settings()
        self.config = PlasticDetectionConfig()
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'analysis'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
    
    def initialize_earth_engine(self):
        """Initialize Google Earth Engine."""
        try:
            result = initialize_earth_engine()
            if result:
                print(f"✅ Successfully initialized Earth Engine with project: {self.settings.EE_PROJECT_ID}")
                return True
            else:
                print(f"❌ Failed to initialize Earth Engine")
                return False
        except Exception as e:
            print(f"❌ Failed to initialize Earth Engine: {e}")
            return False
    
    def step1_download_all_images(self, region: str = 'great_pacific', 
                                 start_date: str = '2023-01-01', 
                                 end_date: str = '2023-12-31',
                                 satellites: List[str] = None,
                                 products: List[str] = None) -> bool:
        """
        Step 1: Download all satellite images and save them as GeoTIFF files.
        """
        print("\n" + "="*60)
        print("🚀 STEP 1: DOWNLOADING ALL IMAGES")
        print("="*60)
        
        if satellites is None:
            satellites = ['sentinel2', 'sentinel1']
        if products is None:
            products = ['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai', 'vv', 'vh']
        
        # Get region geometry
        region_map = {
            'great_pacific': 'Great_Pacific_Patch',
            'mediterranean': 'Mediterranean', 
            'caribbean': 'Caribbean',
            'north_atlantic': 'Mediterranean'  # Fallback to Mediterranean
        }
        
        region_key = region_map.get(region, 'Mediterranean')
        if region_key in self.config.PLASTIC_HOTSPOT_COORDS:
            region_coords = self.config.PLASTIC_HOTSPOT_COORDS[region_key]
            # region_coords is already [min_lon, min_lat, max_lon, max_lat]
            region_geometry = create_roi_from_coordinates(*region_coords)
        else:
            print(f"Warning: Region {region} not found, using Mediterranean")
            region_coords = self.config.PLASTIC_HOTSPOT_COORDS['Mediterranean']
            region_geometry = create_roi_from_coordinates(*region_coords)
        
        print(f"📍 Region: {region}")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"🛰️  Satellites: {', '.join(satellites)}")
        print(f"🎯 Products: {', '.join(products)}")
        
        # Initialize downloaders
        downloaders = {}
        if 'sentinel2' in satellites:
            downloaders['sentinel2'] = Sentinel2Downloader(
                region_geometry, start_date, end_date, self.output_dir
            )
        if 'sentinel1' in satellites:
            downloaders['sentinel1'] = Sentinel1Downloader(
                region_geometry, start_date, end_date, self.output_dir
            )
        if 'landsat' in satellites:
            downloaders['landsat'] = LandsatDownloader(
                region_geometry, start_date, end_date, self.output_dir
            )
        if 'modis' in satellites:
            downloaders['modis'] = MODISDownloader(
                region_geometry, start_date, end_date, self.output_dir
            )
        
        # Product method mapping with satellite compatibility
        product_methods = {
            'rgb': 'download_rgb',
            'false_color': 'download_false_color', 
            'ndvi': 'download_ndvi',
            'ndwi': 'download_ndwi',
            'mndwi': 'download_mndwi',
            'fdi': 'download_fdi',
            'fai': 'download_fai',
            'vv': 'download_vv',
            'vh': 'download_vh'
        }
        
        # Define which products are available for each satellite
        satellite_products = {
            'sentinel2': ['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai'],
            'sentinel1': ['vv', 'vh'],
            'landsat': ['rgb'],  # Could add more Landsat products later
            'modis': ['rgb']     # Could add more MODIS products later
        }
        
        download_urls = []
        
        # Generate download URLs
        for satellite_name, downloader in downloaders.items():
            print(f"\n📡 Processing {satellite_name.upper()} data...")
            
            # Filter products for this satellite
            available_products = satellite_products.get(satellite_name, [])
            compatible_products = [p for p in products if p in available_products]
            
            if not compatible_products:
                print(f"  ⚠️  No compatible products for {satellite_name.upper()} from requested: {', '.join(products)}")
                print(f"  📋 Available products: {', '.join(available_products)}")
                continue
            
            print(f"  🎯 Compatible products: {', '.join(compatible_products)}")
            
            for product in compatible_products:
                method_name = product_methods.get(product)
                if method_name and hasattr(downloader, method_name):
                    try:
                        print(f"  🔄 Generating {product} download URL...")
                        method = getattr(downloader, method_name)
                        result = method()
                        if result and result[0] is not None and result[1] is not None:  # (image, url) tuple
                            image, url = result
                            filename = f"{product}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
                            download_info = {
                                'url': url,
                                'filename': filename,
                                'product': product,
                                'satellite': satellite_name
                            }
                            download_urls.append(download_info)
                            print(f"  ✅ {product} URL generated")
                        else:
                            print(f"  ⚠️  No data available for {product}")
                    except Exception as e:
                        print(f"  ❌ Error generating {product} URL: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Download all files
        if download_urls:
            print(f"\n📥 Downloading {len(download_urls)} files...")
            successful_downloads = []
            
            for i, download_info in enumerate(download_urls, 1):
                url = download_info['url']
                filename = download_info['filename']
                filepath = os.path.join(self.output_dir, filename)
                
                print(f"  [{i}/{len(download_urls)}] Downloading {filename}...")
                
                result_path = download_file_from_url(url, filename, self.output_dir)
                if result_path:
                    successful_downloads.append(result_path)
                    self.downloaded_files.append({
                        'filepath': result_path,
                        'filename': filename,
                        'product': download_info.get('product', 'unknown'),
                        'satellite': download_info.get('satellite', 'unknown')
                    })
            
            print(f"\n✅ Successfully downloaded {len(successful_downloads)}/{len(download_urls)} files")
            
            if successful_downloads:
                print(f"\n📋 QGIS VISUALIZATION GUIDE:")
                print(f"  1. Open QGIS Desktop")
                print(f"  2. Drag & drop the .tif files from: {os.path.abspath(self.output_dir)}")
                print(f"  3. Files are in EPSG:4326 coordinate system")
                print(f"  4. Use 'Symbology' to adjust colors and contrast")
                print(f"  5. For plastic detection, focus on FDI and FAI bands")
            
            return len(successful_downloads) > 0
        else:
            print("❌ No download URLs generated")
            return False
    
    def step2_analyze_images(self) -> bool:
        """
        Step 2: Load images, perform analysis, and save analysis results.
        """
        print("\n" + "="*60)
        print("🔬 STEP 2: ANALYZING IMAGES")
        print("="*60)
        
        if not self.downloaded_files:
            print("❌ No downloaded files to analyze")
            return False
        
        print(f"📊 Analyzing {len(self.downloaded_files)} downloaded images...")
        
        for file_info in self.downloaded_files:
            filepath = file_info['filepath']
            filename = file_info['filename']
            product = file_info['product']
            
            if not os.path.exists(filepath):
                print(f"  ⚠️  File not found: {filename}")
                continue
            
            try:
                print(f"  🔍 Analyzing {filename}...")
                
                # Basic file analysis
                file_size = os.path.getsize(filepath)
                analysis = {
                    'filename': filename,
                    'filepath': filepath,
                    'product': product,
                    'file_size_mb': round(file_size / (1024 * 1024), 2),
                    'timestamp': datetime.now().isoformat(),
                    'analysis_type': self._get_analysis_type(product)
                }
                
                # Product-specific analysis with QGIS tips
                if product in ['fdi', 'fai']:
                    analysis['plastic_detection_potential'] = 'high'
                    analysis['analysis_notes'] = f'{product.upper()} is specifically designed for marine plastic detection'
                    analysis['qgis_tips'] = f'Use "Singleband pseudocolor" with red-yellow colormap. Higher values indicate potential plastic debris.'
                elif product in ['ndvi', 'ndwi', 'mndwi']:
                    analysis['environmental_indicator'] = True
                    analysis['analysis_notes'] = f'{product.upper()} useful for water/vegetation analysis'
                    if product == 'ndvi':
                        analysis['qgis_tips'] = 'Use "RdYlGn" colormap. Green areas = vegetation, red/brown = bare soil/water.'
                    else:
                        analysis['qgis_tips'] = 'Use "Blues" colormap. Higher values = water bodies, lower values = land/vegetation.'
                elif product in ['vv', 'vh']:
                    analysis['radar_polarization'] = product.upper()
                    analysis['analysis_notes'] = f'SAR data - {product.upper()} polarization for surface texture analysis'
                    analysis['qgis_tips'] = 'Use "Grayscale" or "Viridis" colormap. Dark areas = smooth surfaces (water), bright = rough surfaces.'
                elif product == 'rgb':
                    analysis['analysis_notes'] = f'True color composite (Red-Green-Blue bands)'
                    analysis['qgis_tips'] = 'Use "Multiband color" with R=Band1, G=Band2, B=Band3. Adjust min/max values for better contrast.'
                elif product == 'false_color':
                    analysis['analysis_notes'] = f'False color composite (NIR-Red-Green bands)'
                    analysis['qgis_tips'] = 'Use "Multiband color". Vegetation appears red, water appears dark blue/black.'
                else:
                    analysis['analysis_notes'] = f'Satellite imagery - {product}'
                    analysis['qgis_tips'] = 'Use "Singleband pseudocolor" and experiment with different colormaps.'
                
                self.analysis_results[filename] = analysis
                print(f"  ✅ Analysis completed for {filename}")
                
            except Exception as e:
                print(f"  ❌ Error analyzing {filename}: {e}")
        
        # Save analysis results
        analysis_file = os.path.join(self.output_dir, 'analysis', f'analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            import json
            with open(analysis_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2)
            print(f"\n✅ Analysis results saved to: {analysis_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving analysis results: {e}")
            return False
    
    def step3_create_visualizations(self) -> bool:
        """
        Step 3: Load all images and create visualization/analysis collage.
        """
        print("\n" + "="*60)
        print("🎨 STEP 3: CREATING VISUALIZATIONS")
        print("="*60)
        
        if not self.downloaded_files:
            print("❌ No downloaded files to visualize")
            return False
        
        try:
            print("🖼️  Creating satellite data collage...")
            
            # Initialize visualizer
            visualizer = SatelliteVisualizer(self.output_dir)
            
            # Create comprehensive collage - skip advanced visualizer for now
            print("⚠️  Advanced collage creation requires Earth Engine images")
            print("📝 Creating analysis summary instead...")
            collage_path = self._create_simple_collage()
            
            # Create analysis report - skip for now since it needs specific format
            print("📊 Creating analysis summary...")
            # report_path = visualizer.create_analysis_report(self.downloaded_files)
            # if report_path and os.path.exists(report_path):
            #     print(f"✅ Analysis report created: {os.path.basename(report_path)}")
            
            # Create summary visualization with analysis
            summary_path = self._create_analysis_summary()
            if summary_path:
                print(f"✅ Analysis summary created: {os.path.basename(summary_path)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False
    
    def _get_analysis_type(self, product: str) -> str:
        """Get analysis type based on product."""
        if product in ['fdi', 'fai']:
            return 'plastic_detection'
        elif product in ['ndvi']:
            return 'vegetation_analysis'
        elif product in ['ndwi', 'mndwi']:
            return 'water_analysis'
        elif product in ['vv', 'vh']:
            return 'radar_analysis'
        else:
            return 'optical_imagery'
    
    def _create_simple_collage(self) -> Optional[str]:
        """Create a simple collage if the main visualizer fails."""
        try:
            # This is a fallback method
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            collage_path = os.path.join(self.output_dir, 'visualizations', f'simple_collage_{timestamp}.png')
            
            # Create a simple text-based visualization summary
            with open(collage_path.replace('.png', '_summary.txt'), 'w') as f:
                f.write("SATELLITE DATA VISUALIZATION SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Total Files: {len(self.downloaded_files)}\n\n")
                
                for file_info in self.downloaded_files:
                    f.write(f"• {file_info['filename']}\n")
                    f.write(f"  Product: {file_info['product']}\n")
                    f.write(f"  Satellite: {file_info['satellite']}\n\n")
            
            return collage_path.replace('.png', '_summary.txt')
        except:
            return None
    
    def _create_analysis_summary(self) -> Optional[str]:
        """Create analysis summary visualization."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = os.path.join(self.output_dir, 'visualizations', f'analysis_summary_{timestamp}.txt')
            
            with open(summary_path, 'w') as f:
                f.write("PLASTIC DETECTION ANALYSIS SUMMARY\n")
                f.write("="*50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
                f.write(f"Total Images Analyzed: {len(self.analysis_results)}\n\n")
                
                # Group by analysis type
                analysis_types = {}
                for result in self.analysis_results.values():
                    atype = result.get('analysis_type', 'unknown')
                    if atype not in analysis_types:
                        analysis_types[atype] = []
                    analysis_types[atype].append(result)
                
                for atype, results in analysis_types.items():
                    f.write(f"\n{atype.upper().replace('_', ' ')}:\n")
                    f.write("-" * 30 + "\n")
                    for result in results:
                        f.write(f"• {result['filename']} ({result['file_size_mb']} MB)\n")
                        f.write(f"  {result.get('analysis_notes', 'No notes')}\n")
                
                f.write(f"\n\nRECOMMENDations FOR PLASTIC DETECTION:\n")
                f.write("-" * 40 + "\n")
                f.write("• FDI (Floating Debris Index) files are most suitable for plastic detection\n")
                f.write("• FAI (Floating Algae Index) helps distinguish plastic from algae\n")
                f.write("• SAR data (VV/VH) provides texture information for surface analysis\n")
                f.write("• NDWI/MNDWI help identify water bodies and potential contamination\n")
            
            return summary_path
        except Exception as e:
            print(f"Error creating analysis summary: {e}")
            return None
    
    def run_complete_workflow(self, **kwargs) -> bool:
        """Run the complete 3-step workflow."""
        print("🌊 PLASTIC DETECTION WORKFLOW - COMPLETE PIPELINE")
        print("="*60)
        
        # Step 1: Download
        success1 = self.step1_download_all_images(**kwargs)
        if not success1:
            print("❌ Workflow failed at Step 1 (Download)")
            return False
        
        # Step 2: Analysis  
        success2 = self.step2_analyze_images()
        if not success2:
            print("❌ Workflow failed at Step 2 (Analysis)")
            return False
        
        # Step 3: Visualization
        success3 = self.step3_create_visualizations()
        if not success3:
            print("❌ Workflow failed at Step 3 (Visualization)")
            return False
        
        print("\n" + "="*60)
        print("🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 All results saved in: {self.output_dir}")
        print(f"📊 Images downloaded: {len(self.downloaded_files)}")
        print(f"🔬 Analysis results: {len(self.analysis_results)}")
        print(f"📈 Check visualizations folder for collages and reports")
        
        return True


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Plastic Detection Workflow using Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow
  python main.py --workflow complete --region great_pacific

  # Run individual steps
  python main.py --workflow download --region mediterranean
  python main.py --workflow analyze
  python main.py --workflow visualize

  # Custom parameters
  python main.py --workflow complete --region great_pacific --satellites sentinel2 sentinel1 --products rgb ndvi fdi fai
        """
    )
    
    parser.add_argument(
        '--workflow',
        choices=['complete', 'download', 'analyze', 'visualize'],
        default='complete',
        help='Workflow step to run'
    )
    
    parser.add_argument(
        '--region',
        choices=['great_pacific', 'mediterranean', 'north_atlantic', 'caribbean'],
        default='great_pacific',
        help='Region to focus on for plastic detection'
    )
    
    parser.add_argument(
        '--satellites',
        nargs='+',
        choices=['sentinel2', 'sentinel1', 'landsat', 'modis'],
        default=['sentinel2', 'sentinel1'],
        help='Satellite missions to use'
    )
    
    parser.add_argument(
        '--products',
        nargs='+',
        choices=['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai', 'vv', 'vh'],
        default=['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai', 'vv', 'vh'],
        help='Products to download'
    )
    
    parser.add_argument(
        '--start-date',
        default='2023-01-01',
        help='Start date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        default='2023-12-31',
        help='End date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./outputs',
        help='Output directory for all files'
    )
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = PlasticDetectionWorkflow(args.output_dir)
    
    # Initialize Earth Engine
    if not workflow.initialize_earth_engine():
        return 1
    
    # Run selected workflow
    try:
        if args.workflow == 'complete':
            success = workflow.run_complete_workflow(
                region=args.region,
                satellites=args.satellites,
                products=args.products,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.workflow == 'download':
            success = workflow.step1_download_all_images(
                region=args.region,
                satellites=args.satellites,
                products=args.products,
                start_date=args.start_date,
                end_date=args.end_date
            )
        elif args.workflow == 'analyze':
            success = workflow.step2_analyze_images()
        elif args.workflow == 'visualize':
            success = workflow.step3_create_visualizations()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

import ee
import os
import sys
import argparse
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings, PlasticDetectionConfig
from utils.ee_utils import initialize_earth_engine, create_roi_from_coordinates, validate_date_format
from downloader.satellite_downloader import (
    Sentinel2Downloader, 
    Sentinel1Downloader, 
    LandsatDownloader, 
    MODISDownloader
)
from visualizer.visualization import SatelliteVisualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Plastic Detection Satellite Data Downloader')
    
    # Region of Interest
    parser.add_argument('--roi', type=str, default='mediterranean',
                       choices=['mediterranean', 'pacific', 'caribbean', 'bengal', 'custom'],
                       help='Predefined region or custom')
    parser.add_argument('--custom-roi', type=float, nargs=4, metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
                       help='Custom ROI coordinates: min_lon min_lat max_lon max_lat')
    
    # Time period
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD). Default: 30 days ago')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD). Default: today')
    
    # Data selection
    parser.add_argument('--satellites', type=str, nargs='+', 
                       default=['sentinel2', 'sentinel1'],
                       choices=['sentinel2', 'sentinel1', 'landsat', 'modis'],
                       help='Satellites to download data from')
    
    # Products selection
    parser.add_argument('--products', type=str, nargs='+',
                       default=['rgb', 'false_color', 'ndvi', 'ndwi', 'fdi', 'vv', 'vh'],
                       help='Products to download')
    
    # Visualization options
    parser.add_argument('--create-collage', action='store_true', default=True,
                       help='Create visualization collage')
    parser.add_argument('--create-report', action='store_true', default=True,
                       help='Create plastic detection report')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for downloads and visualizations')
    
    return parser.parse_args()

def get_roi_from_args(args):
    """Get region of interest from command line arguments"""
    if args.roi == 'custom' and args.custom_roi:
        min_lon, min_lat, max_lon, max_lat = args.custom_roi
        return create_roi_from_coordinates(min_lon, min_lat, max_lon, max_lat)
    else:
        # Use predefined regions
        hotspots = PlasticDetectionConfig.get_plastic_hotspots()
        roi_map = {
            'mediterranean': hotspots['Mediterranean'],
            'pacific': hotspots['Great_Pacific_Patch'],
            'caribbean': hotspots['Caribbean'],
            'bengal': hotspots['Bay_of_Bengal']
        }
        return roi_map.get(args.roi, Settings.get_default_roi())

def get_date_range(args):
    """Get date range from arguments"""
    if args.start_date and args.end_date:
        if validate_date_format(args.start_date) and validate_date_format(args.end_date):
            return args.start_date, args.end_date
        else:
            print("Invalid date format. Using default dates.")
    
    # Default to last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def download_sentinel2_data(downloader, products):
    """Download Sentinel-2 data based on requested products"""
    results = {}
    
    if 'rgb' in products:
        print("\n--- Downloading Sentinel-2 RGB ---")
        image, url = downloader.download_rgb()
        if image and url:
            results['rgb'] = (image, Settings.RGB_VIS_PARAMS, url)
    
    if 'false_color' in products:
        print("\n--- Downloading Sentinel-2 False Color ---")
        image, url = downloader.download_false_color()
        if image and url:
            results['false_color'] = (image, Settings.FALSE_COLOR_VIS_PARAMS, url)
    
    if 'ndvi' in products:
        print("\n--- Downloading Sentinel-2 NDVI ---")
        image, url = downloader.download_ndvi()
        if image and url:
            results['ndvi'] = (image, Settings.NDVI_VIS_PARAMS, url)
    
    if any(p in products for p in ['ndwi', 'mndwi']):
        print("\n--- Downloading Water Indices ---")
        ndwi_img, mndwi_img, urls = downloader.download_water_indices()
        if ndwi_img and 'ndwi' in urls:
            results['ndwi'] = (ndwi_img, Settings.NDWI_VIS_PARAMS, urls['ndwi'])
        if mndwi_img and 'mndwi' in urls:
            results['mndwi'] = (mndwi_img, Settings.NDWI_VIS_PARAMS, urls['mndwi'])
    
    if any(p in products for p in ['fdi', 'fai']):
        print("\n--- Downloading Plastic Detection Indices ---")
        fdi_img, fai_img, urls = downloader.download_plastic_detection_indices()
        if fdi_img and 'fdi' in urls:
            fdi_vis = {'min': -0.5, 'max': 0.5, 'palette': ['blue', 'white', 'red']}
            results['fdi'] = (fdi_img, fdi_vis, urls['fdi'])
        if fai_img and 'fai' in urls:
            fai_vis = {'min': -0.1, 'max': 0.1, 'palette': ['purple', 'white', 'green']}
            results['fai'] = (fai_img, fai_vis, urls['fai'])
    
    return results

def download_sentinel1_data(downloader, products):
    """Download Sentinel-1 data based on requested products"""
    results = {}
    
    if 'vv' in products and 'vh' in products:
        print("\n--- Downloading Sentinel-1 VV/VH ---")
        vv_img, vh_img, urls = downloader.download_both_polarizations()
        if vv_img and 'vv' in urls:
            results['vv'] = (vv_img, Settings.SAR_VIS_PARAMS, urls['vv'])
        if vh_img and 'vh' in urls:
            results['vh'] = (vh_img, Settings.SAR_VIS_PARAMS, urls['vh'])
    else:
        if 'vv' in products:
            print("\n--- Downloading Sentinel-1 VV ---")
            image, url = downloader.download_vv()
            if image and url:
                results['vv'] = (image, Settings.SAR_VIS_PARAMS, url)
        
        if 'vh' in products:
            print("\n--- Downloading Sentinel-1 VH ---")
            image, url = downloader.download_vh()
            if image and url:
                results['vh'] = (image, Settings.SAR_VIS_PARAMS, url)
    
    return results

def download_landsat_data(downloader, products):
    """Download Landsat data"""
    results = {}
    
    if 'rgb' in products or 'landsat_rgb' in products:
        print("\n--- Downloading Landsat RGB ---")
        image, url = downloader.download_rgb()
        if image and url:
            results['landsat_rgb'] = (image, {'min': 0, 'max': 0.3}, url)
    
    return results

def download_modis_data(downloader, products):
    """Download MODIS data"""
    results = {}
    
    if 'rgb' in products or 'modis_rgb' in products:
        print("\n--- Downloading MODIS RGB ---")
        image, url = downloader.download_rgb()
        if image and url:
            results['modis_rgb'] = (image, {'min': 0, 'max': 3000}, url)
    
    return results

def create_visualizations(all_results, roi, visualizer, args):
    """Create visualizations from downloaded data"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    visualizations_created = []
    
    # Create main collage
    if args.create_collage and all_results:
        print("\n--- Creating Main Satellite Data Collage ---")
        
        # Prepare data for collage
        collage_data = []
        for key, (image, vis_params, url) in all_results.items():
            title = key.replace('_', ' ').title()
            collage_data.append((image, title, vis_params))
        
        if collage_data:
            collage_path = visualizer.create_collage(
                collage_data, roi, 
                title="Plastic Detection - Satellite Data Overview",
                figsize=(20, 16)
            )
            visualizations_created.append(collage_path)
    
    # Create plastic detection report
    if args.create_report and all_results:
        print("\n--- Creating Plastic Detection Report ---")
        
        # Prepare detection data
        detection_data = {}
        for key, (image, vis_params, url) in all_results.items():
            detection_data[key] = (image, vis_params)
        
        if detection_data:
            report_path = visualizer.create_plastic_detection_report(
                detection_data, roi,
                title=f"Plastic Detection Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
            )
            visualizations_created.append(report_path)
    
    # Create indices dashboard
    indices_data = {}
    for key, (image, vis_params, url) in all_results.items():
        if key in ['ndvi', 'ndwi', 'mndwi', 'fdi', 'fai']:
            indices_data[key.upper()] = (image, vis_params)
    
    if indices_data:
        print("\n--- Creating Indices Dashboard ---")
        dashboard_path = visualizer.create_index_dashboard(
            indices_data, roi,
            title="Spectral Indices Dashboard"
        )
        visualizations_created.append(dashboard_path)
    
    return visualizations_created

def print_summary(all_results, visualizations, roi, start_date, end_date):
    """Print summary of downloaded data and created visualizations"""
    print("\n" + "="*60)
    print("DOWNLOAD AND PROCESSING SUMMARY")
    print("="*60)
    
    print(f"\nRegion: {roi.bounds().getInfo() if hasattr(roi, 'bounds') else 'Custom'}")
    print(f"Time Period: {start_date} to {end_date}")
    print(f"Total Products Downloaded: {len(all_results)}")
    
    print("\nDownloaded Products:")
    for key, (image, vis_params, url) in all_results.items():
        print(f"  ✓ {key.replace('_', ' ').title()}")
        print(f"    URL: {url}")
    
    print(f"\nVisualizations Created: {len(visualizations)}")
    for viz_path in visualizations:
        print(f"  ✓ {os.path.basename(viz_path)}")
    
    print(f"\nOutput Directory: {Settings.OUTPUT_DIR}")
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)

def main():
    """Main function"""
    print("="*60)
    print("PLASTIC DETECTION SATELLITE DATA DOWNLOADER")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Initialize Earth Engine
    print("\n--- Initializing Google Earth Engine ---")
    if not initialize_earth_engine():
        print("Failed to initialize Earth Engine. Exiting.")
        return 1
    
    # Set up parameters
    roi = get_roi_from_args(args)
    start_date, end_date = get_date_range(args)
    output_dir = args.output_dir or Settings.OUTPUT_DIR
    
    print(f"\nConfiguration:")
    print(f"  Region: {args.roi}")
    print(f"  Time Period: {start_date} to {end_date}")
    print(f"  Satellites: {', '.join(args.satellites)}")
    print(f"  Products: {', '.join(args.products)}")
    print(f"  Output Directory: {output_dir}")
    
    # Initialize downloaders and visualizer
    visualizer = SatelliteVisualizer(output_dir)
    all_results = {}
    
    # Download from each requested satellite
    print("\n" + "="*60)
    print("DOWNLOADING SATELLITE DATA")
    print("="*60)
    
    try:
        if 'sentinel2' in args.satellites:
            print("\n🛰️  SENTINEL-2 DATA DOWNLOAD")
            s2_downloader = Sentinel2Downloader(roi, start_date, end_date, output_dir)
            s2_results = download_sentinel2_data(s2_downloader, args.products)
            all_results.update(s2_results)
        
        if 'sentinel1' in args.satellites:
            print("\n🛰️  SENTINEL-1 DATA DOWNLOAD")
            s1_downloader = Sentinel1Downloader(roi, start_date, end_date, output_dir)
            s1_results = download_sentinel1_data(s1_downloader, args.products)
            all_results.update(s1_results)
        
        if 'landsat' in args.satellites:
            print("\n🛰️  LANDSAT DATA DOWNLOAD")
            landsat_downloader = LandsatDownloader(roi, start_date, end_date, output_dir)
            landsat_results = download_landsat_data(landsat_downloader, args.products)
            all_results.update(landsat_results)
        
        if 'modis' in args.satellites:
            print("\n🛰️  MODIS DATA DOWNLOAD")
            modis_downloader = MODISDownloader(roi, start_date, end_date, output_dir)
            modis_results = download_modis_data(modis_downloader, args.products)
            all_results.update(modis_results)
        
        # Create visualizations
        if all_results:
            visualizations = create_visualizations(all_results, roi, visualizer, args)
        else:
            print("No data downloaded. Skipping visualization creation.")
            visualizations = []
        
        # Print summary
        print_summary(all_results, visualizations, roi, start_date, end_date)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
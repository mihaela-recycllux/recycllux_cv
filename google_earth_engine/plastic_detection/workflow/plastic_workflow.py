#!/usr/bin/env python3
"""
Core workflow class for plastic detection pipeline.
Manages the 3-step process: Download → Analysis → Visualization
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from config.settings import Settings, PlasticDetectionConfig
from downloader.satellite_downloader import Sentinel2Downloader, Sentinel1Downloader, LandsatDownloader, MODISDownloader
from analyzer.plastic_analyzer import PlasticAnalyzer
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
    
    def initialize_earth_engine(self) -> bool:
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
        
        # Default values
        if satellites is None:
            satellites = ['sentinel2']
        if products is None:
            products = ['rgb', 'ndvi', 'fdi', 'fai']
        
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
        for satellite in satellites:
            if satellite.lower() == 'sentinel2':
                downloaders['sentinel2'] = Sentinel2Downloader(region_geometry, start_date, end_date)
            elif satellite.lower() == 'sentinel1':
                downloaders['sentinel1'] = Sentinel1Downloader(region_geometry, start_date, end_date)
            elif satellite.lower() == 'landsat':
                downloaders['landsat'] = LandsatDownloader(region_geometry, start_date, end_date)
            elif satellite.lower() == 'modis':
                downloaders['modis'] = MODISDownloader(region_geometry, start_date, end_date)
        
        # Generate download URLs for all products
        download_urls = []
        
        # Define which products are available for each satellite
        satellite_products = {
            'sentinel2': ['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai'],
            'sentinel1': ['vv', 'vh'],
            'landsat': ['rgb', 'false_color', 'ndvi', 'ndwi'],
            'modis': ['rgb', 'ndvi', 'evi']
        }
        
        # Define method mapping for each product
        product_methods = {
            'rgb': 'download_rgb',
            'false_color': 'download_false_color', 
            'ndvi': 'download_ndvi',
            'ndwi': 'download_ndwi',
            'mndwi': 'download_mndwi',
            'fdi': 'download_fdi',
            'fai': 'download_fai',
            'vv': 'download_vv',
            'vh': 'download_vh',
            'evi': 'download_evi'
        }
        
        for satellite_name, downloader in downloaders.items():
            print(f"\n📡 Processing {satellite_name.upper()} data...")
            
            # Filter products available for this satellite
            available_products = satellite_products.get(satellite_name, [])
            compatible_products = [p for p in products if p in available_products]
            
            if not compatible_products:
                print(f"  ⚠️  No compatible products for {satellite_name}")
                continue
                
            print(f"  🎯 Compatible products: {', '.join(compatible_products)}")
            
            for product in compatible_products:
                method_name = product_methods.get(product)
                if method_name and hasattr(downloader, method_name):
                    try:
                        print(f"  🔄 Generating {product} download URL...")
                        method = getattr(downloader, method_name)
                        result = method()
                        if result and result[0] is not None and result[1] is not None:
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
                else:
                    print(f"  ⚠️  Method {method_name} not available for {product}")
        
        if not download_urls:
            print("❌ No download URLs generated")
            return False
        
        # Download all files
        print(f"\n📥 Downloading {len(download_urls)} files...")
        successful_downloads = 0
        
        for i, download_info in enumerate(download_urls, 1):
            print(f"  [{i}/{len(download_urls)}] Downloading {download_info['filename']}...")
            
            success = download_file_from_url(
                download_info['url'], 
                download_info['filename'],
                self.output_dir
            )
            
            if success:
                self.downloaded_files.append({
                    'filename': download_info['filename'],
                    'filepath': os.path.join(self.output_dir, download_info['filename']),
                    'product': download_info['product'],
                    'satellite': download_info['satellite']
                })
                successful_downloads += 1
        
        print(f"\n✅ Successfully downloaded {successful_downloads}/{len(download_urls)} files")
        
        if successful_downloads > 0:
            print("\n📋 QGIS VISUALIZATION GUIDE:")
            print("  1. Open QGIS Desktop")
            print(f"  2. Drag & drop the .tif files from: {self.output_dir}")
            print("  3. Files are in EPSG:4326 coordinate system")
            print("  4. Use 'Symbology' to adjust colors and contrast")
            print("  5. For plastic detection, focus on FDI and FAI bands")
            
            return True
        
        return False
    
    def step2_analyze_images(self) -> bool:
        """
        Step 2: Load downloaded images and perform analysis.
        """
        print("\n" + "="*60)
        print("🔬 STEP 2: ANALYZING IMAGES")
        print("="*60)
        
        if not self.downloaded_files:
            print("❌ No downloaded files to analyze. Run step1 first.")
            return False
        
        analyzer = PlasticAnalyzer(self.output_dir)
        
        for file_info in self.downloaded_files:
            try:
                print(f"🔍 Analyzing {file_info['filename']}...")
                
                analysis_type = self._get_analysis_type(file_info['product'])
                result = analyzer.analyze_image(
                    file_info['filepath'], 
                    file_info['product'],
                    analysis_type=analysis_type
                )
                
                if result:
                    self.analysis_results[file_info['filename']] = result
                    print(f"  ✅ Analysis complete")
                else:
                    print(f"  ⚠️  Analysis failed")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing {file_info['filename']}: {e}")
        
        if self.analysis_results:
            print(f"\n✅ Successfully analyzed {len(self.analysis_results)} images")
            return True
        else:
            print("\n❌ No successful analyses")
            return False
    
    def step3_create_visualizations(self) -> bool:
        """
        Step 3: Load all images and create visualization collage.
        """
        print("\n" + "="*60)
        print("📊 STEP 3: CREATING VISUALIZATIONS")
        print("="*60)
        
        if not self.downloaded_files:
            print("❌ No files to visualize. Run step1 first.")
            return False
        
        try:
            visualizer = SatelliteVisualizer(self.output_dir)
            
            # Create collage - for now just create a simple summary since the visualizer needs EE images
            print("⚠️  Advanced collage creation requires Earth Engine images")
            print("📝 Creating analysis summary instead...")
            collage_path = self._create_simple_collage()
            
            if collage_path:
                print(f"✅ Visualization summary created: {collage_path}")
                return True
            else:
                print("❌ Failed to create visualization summary")
                return False
                
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return False
    
    def _get_analysis_type(self, product: str) -> str:
        """Get appropriate analysis type for a product."""
        analysis_mapping = {
            'fdi': 'plastic_detection',
            'fai': 'plastic_detection', 
            'ndvi': 'vegetation_analysis',
            'ndwi': 'water_analysis',
            'mndwi': 'water_analysis',
            'rgb': 'basic_stats',
            'false_color': 'basic_stats',
            'vv': 'sar_analysis',
            'vh': 'sar_analysis'
        }
        return analysis_mapping.get(product, 'basic_stats')
    
    def _create_simple_collage(self) -> Optional[str]:
        """Create a simple analysis summary as HTML."""
        try:
            summary_path = os.path.join(self.output_dir, 'visualizations', 'analysis_summary.html')
            
            with open(summary_path, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Plastic Detection Analysis Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2e8b57; color: white; padding: 20px; text-align: center; }}
        .summary {{ margin: 20px 0; }}
        .file-list {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ Plastic Detection Analysis Summary</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>📊 Downloaded Files ({len(self.downloaded_files)})</h2>
        <div class="file-list">
""")
                
                for file_info in self.downloaded_files:
                    f.write(f"""
            <p><strong>{file_info['filename']}</strong></p>
            <ul>
                <li>Product: {file_info['product']}</li>
                <li>Satellite: {file_info['satellite']}</li>
                <li>Path: {file_info['filepath']}</li>
            </ul>
""")
                
                f.write("""
        </div>
    </div>
</body>
</html>
""")
            
            return summary_path
            
        except Exception as e:
            print(f"Error creating analysis summary: {e}")
            return None
    
    def run_complete_workflow(self, **kwargs) -> bool:
        """Run the complete 3-step workflow."""
        print("🚀 Starting Complete Plastic Detection Workflow")
        print("=" * 80)
        
        # Step 1: Download
        if not self.step1_download_all_images(**kwargs):
            print("❌ Step 1 failed, stopping workflow")
            return False
        
        # Step 2: Analysis
        if not self.step2_analyze_images():
            print("❌ Step 2 failed, stopping workflow")
            return False
        
        # Step 3: Visualization
        if not self.step3_create_visualizations():
            print("❌ Step 3 failed, stopping workflow")
            return False
        
        print("\n" + "="*80)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📁 All results saved in: {self.output_dir}")
        print(f"📊 Images downloaded: {len(self.downloaded_files)}")
        print(f"🔬 Analysis results: {len(self.analysis_results)}")
        print(f"📈 Check visualizations folder for collages and reports")
        
        return True
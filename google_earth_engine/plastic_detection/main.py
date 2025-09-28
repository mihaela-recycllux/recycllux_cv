#!/usr/bin/env python3
"""
Simple Plastic Detection Analysis Tool
GPS coordinates + area → comprehensive plastic analysis with trends

Usage: python main.py --lat 44.217 --lon 28.965 --area 25
"""

import argparse
import sys
import os
import glob
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.analyzer import PlasticAnalyzer
from core.visualizer import PlasticVisualizer
from utils.earth_engine_client import EarthEngineClient


class PlasticDetectionTool:
    """Main plastic detection analysis tool"""
    
    def __init__(self, output_dir="./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.ee_client = EarthEngineClient()
        self.analyzer = PlasticAnalyzer(self.output_dir)
        self.visualizer = PlasticVisualizer(self.output_dir)
        
    def _cleanup_temporary_files(self):
        """Clean up temporary tile files and other intermediate files"""
        import glob
        
        patterns_to_clean = [
            str(self.output_dir / "*_tile_*.tif"),  # Tile files
            str(self.output_dir / "*.zip"),         # Zip downloads
            str(self.output_dir / "temp_*"),        # Temporary files
        ]
        
        cleaned_count = 0
        for pattern in patterns_to_clean:
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                    print(f"   Removed: {Path(file_path).name}")
                except Exception as e:
                    print(f"   Warning: Could not remove {Path(file_path).name}: {e}")
        
        if cleaned_count > 0:
            print(f"✨ Cleaned up {cleaned_count} temporary files")
        else:
            print("✨ No temporary files to clean")
    
    def analyze_location(self, latitude: float, longitude: float, area_km: int = 25):
        """
        Run comprehensive plastic detection analysis for a location.
        
        Args:
            latitude: GPS latitude
            longitude: GPS longitude  
            area_km: Analysis area in kilometers (10, 25, 50, 100)
        
        Returns:
            Path to final analysis report
        """
        print(f"🛰️  Starting Plastic Detection Analysis")
        print(f"📍 Location: {latitude:.6f}°N, {longitude:.6f}°E")
        print(f"📏 Area: {area_km}x{area_km} km")
        print("=" * 60)
        
        # Step 1: Initialize Earth Engine
        if not self.ee_client.initialize():
            print("❌ Failed to initialize Earth Engine")
            return None
        
        # Step 2: Download satellite data
        print("\n🔄 Downloading satellite data...")
        data_files = self.ee_client.download_area_data(latitude, longitude, area_km)
        
        if not data_files:
            print("❌ No data downloaded")
            return None
        
        # Step 3: Analyze for plastic detection  
        print("\n🔬 Analyzing plastic presence...")
        analysis_results = self.analyzer.analyze_plastic_detection(data_files)
        
        # Step 4: Generate time series trend
        print("\n📈 Analyzing temporal trends...")
        trend_data = self.analyzer.analyze_temporal_trend(latitude, longitude, area_km)
        
        # Step 5: Create comprehensive visualization
        print("\n📊 Creating analysis visualization...")
        report_path = self.visualizer.create_comprehensive_report(
            location=(latitude, longitude),
            area_km=area_km,
            data_files=data_files,
            analysis_results=analysis_results,
            trend_data=trend_data
        )
        
        if report_path:
            print(f"\n✅ Analysis complete! Report saved to: {report_path}")
            
            # Clean up temporary files
            print("🧹 Cleaning up temporary files...")
            self._cleanup_temporary_files()
            
            return report_path
        else:
            print("\n❌ Failed to create analysis report")
            return None


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Plastic Detection Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --lat 44.217 --lon 28.965 --area 1   # Fast test (1km)
  python main.py --lat 44.217 --lon 28.965 --area 5   # Small area (5km)
  python main.py --lat 35.123 --lon -120.456 --area 10  # Medium area
  python main.py --lat 41.234 --lon 2.345 --area 25     # Large area
        """
    )
    
    parser.add_argument('--lat', type=float, required=True,
                       help='Latitude (GPS coordinate)')
    
    parser.add_argument('--lon', type=float, required=True, 
                       help='Longitude (GPS coordinate)')
    
    parser.add_argument('--area', type=int, choices=[1, 5, 10, 25, 50, 100], default=25,
                       help='Analysis area size in km (default: 25, options: 1,5,10,25,50,100)')
    
    parser.add_argument('--output', default='./results',
                       help='Output directory (default: ./results)')
    
    return parser


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate GPS coordinates"""
    if not (-90 <= lat <= 90):
        print(f"❌ Invalid latitude: {lat} (must be between -90 and 90)")
        return False
    
    if not (-180 <= lon <= 180):
        print(f"❌ Invalid longitude: {lon} (must be between -180 and 180)")
        return False
    
    return True


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate coordinates
    if not validate_coordinates(args.lat, args.lon):
        return 1
    
    # Initialize tool
    try:
        tool = PlasticDetectionTool(args.output)
        
        # Run analysis
        result_path = tool.analyze_location(args.lat, args.lon, args.area)
        
        if result_path:
            print(f"\n🎉 Success! Open this file to view results:")
            print(f"   {result_path}")
            return 0
        else:
            print(f"\n❌ Analysis failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
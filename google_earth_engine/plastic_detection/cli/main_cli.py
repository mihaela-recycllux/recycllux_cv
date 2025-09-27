#!/usr/bin/env python3
"""
Command-line interface for the plastic detection workflow.
Provides easy access to all workflow steps with comprehensive argument parsing.
"""

import argparse
import sys
from typing import List

from workflow.plastic_workflow import PlasticDetectionWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
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

  # Plastic detection focus
  python main.py --workflow download --products fdi fai ndwi --region mediterranean --satellites sentinel2
        """
    )
    
    parser.add_argument(
        '--workflow',
        choices=['complete', 'download', 'analyze', 'visualize'],
        default='complete',
        help='Workflow step to run (default: complete)'
    )
    
    parser.add_argument(
        '--region',
        choices=['great_pacific', 'mediterranean', 'caribbean', 'north_atlantic'],
        default='great_pacific',
        help='Ocean region to analyze (default: great_pacific)'
    )
    
    parser.add_argument(
        '--satellites',
        nargs='+',
        choices=['sentinel2', 'sentinel1', 'landsat', 'modis'],
        default=['sentinel2', 'sentinel1'],
        help='Satellite sources to use (default: sentinel2 sentinel1)'
    )
    
    parser.add_argument(
        '--products',
        nargs='+',
        choices=['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai', 'vv', 'vh', 'evi'],
        default=['rgb', 'false_color', 'ndvi', 'ndwi', 'mndwi', 'fdi', 'fai', 'vv', 'vh'],
        help='Data products to download and analyze'
    )
    
    parser.add_argument(
        '--start-date',
        default='2023-01-01',
        help='Start date for data collection (YYYY-MM-DD, default: 2023-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        default='2023-12-31',
        help='End date for data collection (YYYY-MM-DD, default: 2023-12-31)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./outputs',
        help='Output directory for results (default: ./outputs)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    # Validate date format (basic check)
    try:
        from datetime import datetime
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("❌ Error: Invalid date format. Use YYYY-MM-DD")
        return False
    
    # Validate satellite-product compatibility
    if 'sentinel1' in args.satellites and any(p in args.products for p in ['rgb', 'ndvi', 'fdi', 'fai']):
        print("⚠️  Warning: Sentinel-1 SAR data doesn't support optical products (RGB, NDVI, FDI, FAI)")
        print("   Compatible Sentinel-1 products: vv, vh")
    
    if 'sentinel2' in args.satellites and any(p in args.products for p in ['vv', 'vh']):
        print("⚠️  Warning: Sentinel-2 optical data doesn't support SAR products (VV, VH)")
        print("   Compatible Sentinel-2 products: rgb, false_color, ndvi, ndwi, mndwi, fdi, fai")
    
    return True


def run_workflow(args: argparse.Namespace) -> bool:
    """Execute the selected workflow with given arguments."""
    # Initialize workflow
    workflow = PlasticDetectionWorkflow(output_dir=args.output_dir)
    
    # Initialize Earth Engine
    if not workflow.initialize_earth_engine():
        print("❌ Failed to initialize Google Earth Engine")
        return False
    
    # Prepare workflow parameters
    workflow_params = {
        'region': args.region,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'satellites': args.satellites,
        'products': args.products
    }
    
    # Execute selected workflow
    try:
        if args.workflow == 'complete':
            return workflow.run_complete_workflow(**workflow_params)
        
        elif args.workflow == 'download':
            return workflow.step1_download_all_images(**workflow_params)
        
        elif args.workflow == 'analyze':
            return workflow.step2_analyze_images()
        
        elif args.workflow == 'visualize':
            return workflow.step3_create_visualizations()
        
        else:
            print(f"❌ Unknown workflow: {args.workflow}")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        return False
    
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    # Print startup banner
    print("🛰️  Plastic Detection Workflow")
    print("=" * 50)
    print(f"Workflow: {args.workflow}")
    print(f"Region: {args.region}")
    print(f"Satellites: {', '.join(args.satellites)}")
    print(f"Products: {', '.join(args.products)}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # Run workflow
    success = run_workflow(args)
    
    if success:
        print("\n✅ Workflow completed successfully!")
        return 0
    else:
        print("\n❌ Workflow failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
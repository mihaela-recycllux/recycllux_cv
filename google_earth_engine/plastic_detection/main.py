#!/usr/bin/env python3
"""
Comprehensive Plastic Debris Detection using Google Earth Engine

This script implements an object-oriented approach to plastic detection using
multiple spectral indices and detection methods. It combines optical and SAR
data from Sentinel satellites to identify floating plastic debris.

Key Features:
1. Downloads Sentinel-1 SAR and Sentinel-2 optical data from GEE
2. Calculates multiple spectral indices (FDI, NDVI, Plastic Index)
3. Applies various detection methods with confidence scoring
4. Provides comprehensive visualization and analysis
5. Modular design for easy extension and customization

Study Area: Romanian coast of the Black Sea (near Constanța port)

Author: Varun Burde
Email: varun@recycllux.com
Date: 2024
Reference: Combined methodology from Biermann et al. (2020) and Topouzelis et al. (2020)

Usage:
    python main.py
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import config
from downloader.downloader import GEEDownloader
from filters.filter_manager import FilterManager
from utils.visualization import VisualizationUtils

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Ensure log directory exists if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

def main():
    """Main function to run comprehensive plastic detection"""
    print("=" * 80)
    print("COMPREHENSIVE PLASTIC DEBRIS DETECTION - GOOGLE EARTH ENGINE")
    print("Object-Oriented Multi-sensor Fusion Approach")
    print("=" * 80)
    print("Location: Romanian coast of the Black Sea (near Constanța)")
    print("Sensors: Sentinel-1 SAR + Sentinel-2 Optical")
    print("Platform: Google Earth Engine")
    print("Methods: FDI + NDVI + Plastic Index + Ensemble")
    print("=" * 80)

    # Setup logging
    log_file = config.get_output_path('detection.log', 'logs')
    setup_logging(log_level='INFO', log_file=log_file)
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        logger.info("Initializing detection pipeline...")

        # 1. Initialize downloader
        downloader = GEEDownloader(config)

        # 2. Initialize filter manager
        filter_manager = FilterManager(config.__dict__)

        # 3. Initialize visualization utils
        viz_utils = VisualizationUtils(config)

        logger.info("✓ All components initialized successfully")

        # Step 1: Download multi-sensor data
        print(f"\n{'='*60}")
        print("STEP 1: DOWNLOADING MULTI-SENSOR DATA")
        print(f"{'='*60}")

        optical_data, sar_data, data_mask = downloader.download_multi_sensor_data()

        # Step 2: Calculate spectral indices
        print(f"\n{'='*60}")
        print("STEP 2: CALCULATING SPECTRAL INDICES")
        print(f"{'='*60}")

        indices = filter_manager.calculate_indices(optical_data, sar_data, data_mask)

        # Step 3: Create water mask (using NDVI)
        print(f"\n{'='*60}")
        print("STEP 3: CREATING WATER MASK")
        print(f"{'='*60}")

        if 'ndvi' in filter_manager.active_filters:
            ndvi_filter = filter_manager.active_filters['ndvi']
            water_mask = ndvi_filter.create_water_mask(indices['ndvi'])
        else:
            # Fallback: simple water mask based on data availability
            water_mask = np.ones_like(data_mask, dtype=float)
            water_mask[data_mask == 0] = np.nan
            logger.warning("Using fallback water mask (all valid data areas)")

        # Debug water mask
        water_pixels = np.sum(water_mask == 1)
        total_pixels = np.sum(~np.isnan(water_mask))
        water_percentage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0
        logger.info(f"Water mask: {water_pixels} water pixels ({water_percentage:.1f}%)")

        # Step 4: Run plastic detection for all filters
        print(f"\n{'='*60}")
        print("STEP 4: RUNNING PLASTIC DETECTION")
        print(f"{'='*60}")

        detections = filter_manager.detect_plastic_all_filters(indices, water_mask)

        # Step 5: Create ensemble detection
        print(f"\n{'='*60}")
        print("STEP 5: CREATING ENSEMBLE DETECTION")
        print(f"{'='*60}")

        ensemble_mask, ensemble_metadata = filter_manager.create_ensemble_detection(detections)
        detections['ensemble'] = (ensemble_mask, ensemble_metadata)

        # Step 6: Create visualizations
        print(f"\n{'='*60}")
        print("STEP 6: CREATING VISUALIZATIONS")
        print(f"{'='*60}")

        # Create RGB composite
        rgb_composite = viz_utils.create_rgb_composite(optical_data)

        # Prepare detection masks for visualization
        detection_masks = {name: mask for name, (mask, _) in detections.items()}

        # Create main detection visualization
        bbox = config.aoi_bounds
        time_period = (config.start_date, config.end_date)

        main_viz_path = config.get_output_path(
            f"detection_results_{time_period[0]}_{time_period[1]}.png", 'images'
        )
        viz_utils.create_detection_visualization(
            rgb_composite, detection_masks, bbox, time_period, main_viz_path
        )

        # Create index analysis plot
        index_viz_path = config.get_output_path(
            f"index_analysis_{time_period[0]}_{time_period[1]}.png", 'images'
        )
        viz_utils.create_index_analysis_plot(indices, index_viz_path)

        # Create detection comparison plot
        comparison_viz_path = config.get_output_path(
            f"detection_comparison_{time_period[0]}_{time_period[1]}.png", 'images'
        )
        viz_utils.create_detection_comparison_plot(detections, comparison_viz_path)

        # Step 7: Save results
        print(f"\n{'='*60}")
        print("STEP 7: SAVING RESULTS")
        print(f"{'='*60}")

        # Prepare metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'aoi': config.aoi_bounds,
                'time_period': time_period,
                'satellite_params': config.satellite_params,
                'processing_params': config.processing_params,
                'detection_thresholds': config.detection_thresholds
            },
            'data_info': {
                'optical_shape': optical_data.shape,
                'sar_shape': sar_data.shape,
                'data_mask_coverage': np.mean(data_mask) * 100,
                'water_coverage': water_percentage
            },
            'filter_info': filter_manager.get_filter_info(),
            'detection_summary': {
                name: {
                    'detections': meta.get('detections', 0),
                    'detection_rate': meta.get('detection_rate', 0),
                    'threshold': meta.get('threshold_used', 'N/A')
                }
                for name, (_, meta) in detections.items()
            }
        }

        # Save all results
        viz_utils.save_detection_results(detections, indices, metadata, config.output_dirs['data'])

        # Step 8: Generate summary report
        print(f"\n{'='*60}")
        print("STEP 8: GENERATING SUMMARY REPORT")
        print(f"{'='*60}")

        generate_summary_report(detections, indices, metadata, water_mask)

        print(f"\n{'='*80}")
        print("COMPREHENSIVE PLASTIC DETECTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

        # Final summary
        ensemble_detections = ensemble_metadata.get('detections', 0)
        ensemble_rate = ensemble_metadata.get('detection_rate', 0)

        print("\nFINAL DETECTION SUMMARY:")
        print(f"  Study Area: Romanian Black Sea Coast")
        print(f"  Time Period: {time_period[0]} to {time_period[1]}")
        print(f"  Water Coverage: {water_percentage:.1f}% of study area")
        print(f"  Ensemble Detections: {ensemble_detections:,} pixels")
        print(f"  Detection Rate: {ensemble_rate:.2f}% of water area")

        # Method comparison
        print(f"\n  Method Performance:")
        for name, (_, meta) in detections.items():
            detections_count = meta.get('detections', 0)
            detection_rate = meta.get('detection_rate', 0)
            print(f"    {name.upper()}: {detections_count:,} pixels ({detection_rate:.2f}%)")

        print(f"\n  Output Location: {config.output_dirs['data']}")
        print(f"  Log File: {log_file}")

        print("\nRECOMMENDATIONS:")
        print("• Review high-confidence detections (>0.7 confidence)")
        print("• Validate results with field observations")
        print("• Consider temporal analysis for confirmation")
        print("• Use ensemble results for operational decisions")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Check GEE authentication and credentials")
        print("2. Verify data availability for the time period")
        print("3. Ensure all required packages are installed")
        print("4. Check internet connection and GEE quota")
        print("5. Review log file for detailed error information")
        raise

def generate_summary_report(detections: Dict[str, Any], indices: Dict[str, np.ndarray],
                          metadata: Dict[str, Any], water_mask: np.ndarray):
    """Generate a comprehensive summary report"""

    report_path = config.get_output_path('detection_summary_report.txt', 'statistics')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PLASTIC DETECTION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Study Area: {metadata['config']['aoi']}\n")
        f.write(f"Time Period: {metadata['config']['time_period'][0]} to {metadata['config']['time_period'][1]}\n")
        f.write(f"Target Resolution: {metadata['config']['processing_params']['target_resolution']}m\n")
        f.write(f"Image Size: {metadata['config']['processing_params']['image_size']}\n\n")

        # Data Information
        f.write("DATA INFORMATION:\n")
        f.write("-" * 40 + "\n")
        data_info = metadata['data_info']
        f.write(f"Optical Data Shape: {data_info['optical_shape']}\n")
        f.write(f"SAR Data Shape: {data_info['sar_shape']}\n")
        f.write(f"Data Mask Coverage: {data_info['data_mask_coverage']:.1f}%\n")
        f.write(f"Water Coverage: {data_info['water_coverage']:.1f}%\n\n")

        # Detection Results
        f.write("DETECTION RESULTS:\n")
        f.write("-" * 40 + "\n")

        for name, (_, meta) in detections.items():
            f.write(f"{name.upper()} Method:\n")
            f.write(f"  Detections: {meta.get('detections', 0):,} pixels\n")
            f.write(f"  Detection Rate: {meta.get('detection_rate', 0):.2f}%\n")
            if 'threshold_used' in meta:
                f.write(f"  Threshold Used: {meta['threshold_used']:.4f}\n")
            f.write("\n")

        # Index Statistics
        f.write("INDEX STATISTICS:\n")
        f.write("-" * 40 + "\n")

        for name, index_data in indices.items():
            valid_data = index_data[~np.isnan(index_data)]
            if len(valid_data) > 0:
                f.write(f"{name.upper()} Index:\n")
                f.write(f"  Mean: {np.mean(valid_data):.4f}\n")
                f.write(f"  Std: {np.std(valid_data):.4f}\n")
                f.write(f"  Range: {np.min(valid_data):.4f} to {np.max(valid_data):.4f}\n")
                f.write(f"  Valid Pixels: {len(valid_data)}\n\n")

        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("• Focus on ensemble detection results for highest reliability\n")
        f.write("• Validate high-confidence detections with field data\n")
        f.write("• Consider multi-temporal analysis for change detection\n")
        f.write("• Use water mask to restrict analysis to relevant areas\n")
        f.write("• Review index distributions for algorithm tuning\n\n")

        f.write(f"Report Generated: {metadata['timestamp']}\n")
        f.write(f"Output Directory: {config.output_dirs['data']}\n")

    print(f"✓ Summary report saved to: {report_path}")

if __name__ == "__main__":
    main()
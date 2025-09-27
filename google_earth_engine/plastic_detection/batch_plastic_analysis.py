#!/usr/bin/env python3
"""
Batch Plastic Detection Analysis for Multiple Locations
Google Earth Engine Implementation

This script enables batch processing of multiple locations for plastic detection analysis.
Useful for large-scale environmental monitoring and research applications.

Author: Varun Burde 
Email: varun@recycllux.com
Date: 2025

"""

import ee
import csv
import json
import pandas as pd
from datetime import datetime, timedelta
from comprehensive_plastic_detection_ee import analyze_coordinates_for_plastic

def load_coordinates_from_csv(csv_file):
    """
    Load coordinates from CSV file
    
    Expected CSV format:
    name,lat,lon,description
    Location1,44.2170,28.9650,Danube Delta
    Location2,44.2500,28.9500,Coast point
    
    Args:
        csv_file: Path to CSV file with coordinates
    
    Returns:
        list: List of coordinate dictionaries
    """
    coordinates = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                coordinates.append({
                    'name': row.get('name', f"Location_{len(coordinates)+1}"),
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'description': row.get('description', '')
                })
        
        print(f"✓ Loaded {len(coordinates)} locations from {csv_file}")
        return coordinates
        
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return []

def create_monitoring_grid(center_lat, center_lon, grid_size_km, grid_points):
    """
    Create a grid of monitoring points around a center location
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        grid_size_km: Size of grid in kilometers
        grid_points: Number of points per side (e.g., 3 = 3x3 grid)
    
    Returns:
        list: List of grid coordinate dictionaries
    """
    coordinates = []
    
    # Convert km to approximate degrees
    km_to_deg = grid_size_km / 111.32
    
    # Create grid
    for i in range(grid_points):
        for j in range(grid_points):
            # Calculate offset from center
            lat_offset = (i - grid_points//2) * km_to_deg
            lon_offset = (j - grid_points//2) * km_to_deg
            
            grid_lat = center_lat + lat_offset
            grid_lon = center_lon + lon_offset
            
            coordinates.append({
                'name': f'Grid_{i}_{j}',
                'lat': grid_lat,
                'lon': grid_lon,
                'description': f'Grid point ({i},{j})'
            })
    
    print(f"✓ Created {len(coordinates)} grid points ({grid_points}x{grid_points})")
    return coordinates

def batch_plastic_analysis(coordinates_list, output_dir="batch_results", buffer_km=3):
    """
    Perform batch plastic detection analysis
    
    Args:
        coordinates_list: List of coordinate dictionaries
        output_dir: Output directory for results
        buffer_km: Buffer distance for each analysis
    
    Returns:
        dict: Batch analysis results
    """
    import os
    
    print("Starting batch plastic detection analysis...")
    print(f"Processing {len(coordinates_list)} locations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    batch_results = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'total_locations': len(coordinates_list),
            'buffer_km': buffer_km
        },
        'locations': [],
        'summary': {}
    }
    
    successful_analyses = 0
    detected_locations = []
    total_detected_area = 0
    
    for i, coords in enumerate(coordinates_list, 1):
        print(f"\n{'='*60}")
        print(f"Processing location {i}/{len(coordinates_list)}: {coords['name']}")
        print(f"Coordinates: {coords['lat']:.5f}°N, {coords['lon']:.5f}°E")
        print(f"{'='*60}")
        
        try:
            # Run analysis
            result = analyze_coordinates_for_plastic(
                lat=coords['lat'],
                lon=coords['lon'],
                buffer_km=buffer_km,
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            if result and 'error' not in result:
                successful_analyses += 1
                
                # Check for detection
                detected_area = result.get('area_statistics', {}).get('detected_area_m2', 0)
                if detected_area > 0:
                    detected_locations.append({
                        'name': coords['name'],
                        'coordinates': coords,
                        'detected_area_m2': detected_area
                    })
                    total_detected_area += detected_area
                
                # Add location info to result
                result['location_info'] = coords
                batch_results['locations'].append(result)
                
                # Save individual result
                result_filename = f"{coords['name'].replace(' ', '_')}_{coords['lat']:.3f}N_{coords['lon']:.3f}E.json"
                result_path = os.path.join(output_dir, result_filename)
                
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"✓ Analysis completed - Results saved to {result_filename}")
                
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Analysis failed'
                print(f"❌ Analysis failed: {error_msg}")
                
                batch_results['locations'].append({
                    'location_info': coords,
                    'error': error_msg
                })
        
        except Exception as e:
            print(f"❌ Exception during analysis: {e}")
            batch_results['locations'].append({
                'location_info': coords,
                'error': str(e)
            })
    
    # Calculate summary statistics
    batch_results['summary'] = {
        'successful_analyses': successful_analyses,
        'failed_analyses': len(coordinates_list) - successful_analyses,
        'success_rate': successful_analyses / len(coordinates_list) * 100,
        'locations_with_detection': len(detected_locations),
        'detection_rate': len(detected_locations) / successful_analyses * 100 if successful_analyses > 0 else 0,
        'total_detected_area_m2': total_detected_area,
        'total_detected_area_km2': total_detected_area / 1e6,
        'detected_locations': detected_locations
    }
    
    # Save batch summary
    summary_filename = f"batch_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BATCH ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}/")
    print(f"Summary file: {summary_filename}")
    
    return batch_results

def generate_batch_report(batch_results, output_dir="batch_results"):
    """
    Generate a detailed batch analysis report
    
    Args:
        batch_results: Results from batch analysis
        output_dir: Output directory
    """
    import os
    
    report_lines = []
    
    # Header
    report_lines.append("RECYLLUX PLASTIC DETECTION BATCH ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Analysis Date: {batch_results['analysis_info']['timestamp'][:10]}")
    report_lines.append("")
    
    # Summary statistics
    summary = batch_results['summary']
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 30)
    report_lines.append(f"Total locations analyzed: {batch_results['analysis_info']['total_locations']}")
    report_lines.append(f"Successful analyses: {summary['successful_analyses']}")
    report_lines.append(f"Failed analyses: {summary['failed_analyses']}")
    report_lines.append(f"Success rate: {summary['success_rate']:.1f}%")
    report_lines.append("")
    report_lines.append(f"Locations with plastic detection: {summary['locations_with_detection']}")
    report_lines.append(f"Detection rate: {summary['detection_rate']:.1f}%")
    report_lines.append(f"Total detected plastic area: {summary['total_detected_area_m2']:.0f} m²")
    report_lines.append(f"Total detected plastic area: {summary['total_detected_area_km2']:.4f} km²")
    report_lines.append("")
    
    # Detected locations details
    if summary['detected_locations']:
        report_lines.append("LOCATIONS WITH PLASTIC DETECTION")
        report_lines.append("-" * 40)
        
        for i, location in enumerate(summary['detected_locations'], 1):
            coords = location['coordinates']
            report_lines.append(f"{i}. {location['name']}")
            report_lines.append(f"   Coordinates: {coords['lat']:.5f}°N, {coords['lon']:.5f}°E")
            report_lines.append(f"   Detected area: {location['detected_area_m2']:.0f} m²")
            if coords['description']:
                report_lines.append(f"   Description: {coords['description']}")
            report_lines.append("")
    else:
        report_lines.append("No plastic detection found in any analyzed location.")
        report_lines.append("")
    
    # Location-by-location results
    report_lines.append("DETAILED RESULTS BY LOCATION")
    report_lines.append("-" * 40)
    
    for i, location_result in enumerate(batch_results['locations'], 1):
        location_info = location_result.get('location_info', {})
        report_lines.append(f"{i}. {location_info.get('name', f'Location {i}')}")
        report_lines.append(f"   Coordinates: {location_info.get('lat', 'N/A')}°N, {location_info.get('lon', 'N/A')}°E")
        
        if 'error' in location_result:
            report_lines.append(f"   Status: FAILED - {location_result['error']}")
        else:
            area_stats = location_result.get('area_statistics', {})
            data_avail = location_result.get('data_availability', {})
            
            detected_area = area_stats.get('detected_area_m2', 0)
            if detected_area > 0:
                report_lines.append(f"   Status: PLASTIC DETECTED")
                report_lines.append(f"   Detected area: {detected_area:.0f} m²")
                report_lines.append(f"   Coverage: {area_stats.get('coverage_percentage', 0):.4f}%")
            else:
                report_lines.append(f"   Status: No plastic detected")
            
            report_lines.append(f"   Data: S2({data_avail.get('sentinel2_images', 0)}) S1({data_avail.get('sentinel1_images', 0)})")
        
        report_lines.append("")
    
    # Methodology
    report_lines.append("METHODOLOGY")
    report_lines.append("-" * 20)
    report_lines.append("• Multi-sensor fusion: Sentinel-1 SAR + Sentinel-2 optical")
    report_lines.append("• Floating Debris Index (FDI) algorithm")
    report_lines.append("• Spectral classification with multiple thresholds")
    report_lines.append("• Ensemble detection for reliability")
    report_lines.append("• Water masking for focused analysis")
    report_lines.append("• Google Earth Engine cloud processing")
    report_lines.append("")
    
    # Save report
    report_filename = f"batch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print report
    print('\n'.join(report_lines))
    print(f"\nDetailed report saved to: {report_path}")

def main():
    """Main function for batch processing"""
    
    print("Recyllux Batch Plastic Detection Analysis")
    print("=" * 50)
    
    # Example 1: Romanian Black Sea monitoring points
    romanian_coast_points = [
        {
            'name': 'Target_Location',
            'lat': 44.21706925,
            'lon': 28.96504135,
            'description': 'User provided coordinates - Danube Delta region'
        },
        {
            'name': 'Constanta_Port',
            'lat': 44.1667,
            'lon': 28.6333,
            'description': 'Major port area'
        },
        {
            'name': 'Danube_Delta_North',
            'lat': 44.3500,
            'lon': 29.0500,
            'description': 'Northern Danube Delta'
        },
        {
            'name': 'Danube_Delta_South', 
            'lat': 44.1000,
            'lon': 28.9000,
            'description': 'Southern Danube Delta'
        },
        {
            'name': 'Mamaia_Beach',
            'lat': 44.2500,
            'lon': 28.6167,
            'description': 'Popular beach resort'
        }
    ]
    
    print(f"Running batch analysis for {len(romanian_coast_points)} Romanian Black Sea locations...")
    
    # Run batch analysis
    results = batch_plastic_analysis(
        coordinates_list=romanian_coast_points,
        output_dir="google_earth_engine/plastic_detection/batch_results",
        buffer_km=3
    )
    
    # Generate report
    generate_batch_report(results, "google_earth_engine/plastic_detection/batch_results")
    
    # Check specifically for the user's coordinates
    print(f"\n{'='*70}")
    print("SPECIFIC ANALYSIS FOR PROVIDED COORDINATES")
    print(f"{'='*70}")
    
    target_result = None
    for location_result in results['locations']:
        location_info = location_result.get('location_info', {})
        if location_info.get('name') == 'Target_Location':
            target_result = location_result
            break
    
    if target_result and 'error' not in target_result:
        area_stats = target_result.get('area_statistics', {})
        detected_area = area_stats.get('detected_area_m2', 0)
        
        print(f"Location: {romanian_coast_points[0]['lat']:.6f}°N, {romanian_coast_points[0]['lon']:.6f}°E")
        
        if detected_area > 0:
            print("🚨 PLASTIC ACCUMULATION DETECTED!")
            print(f"Estimated area: {detected_area:.0f} m²")
            print(f"Coverage: {area_stats.get('coverage_percentage', 0):.4f}% of analyzed area")
            print("✓ Algorithm successfully identified potential plastic debris")
            print("📸 Visualization images have been generated")
            
            # For Social Tides report
            print("\nFOR SOCIAL TIDES REPORT:")
            print("• Our AI algorithm detected plastic accumulation at the specified coordinates")
            print("• Detection method: Multi-sensor satellite data fusion")
            print("• Data sources: Sentinel-1 SAR + Sentinel-2 optical imagery")
            print("• Algorithm combines multiple spectral indices for reliable detection")
            print("• Results include confidence mapping and area quantification")
        else:
            print("✓ No significant plastic accumulation detected")
            print("This could indicate:")
            print("  • Clean water conditions at the specified location")
            print("  • Plastic concentration below detection threshold") 
            print("  • Temporal variation in debris presence")
    else:
        print("❌ Analysis failed for the target coordinates")
    
    return results

if __name__ == "__main__":
    main()
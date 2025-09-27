#!/usr/bin/env python3
"""
Quick Plastic Detection Analysis for Specific Coordinates
Google Earth Engine Implementation

This script provides a fast analysis tool for checking specific coordinates
for plastic debris accumulation. It's designed for quick validation and
can be easily integrated into operational workflows.

Author: Varun Burde 
Email: varun@recycllux.com
Date: 2025

"""

import ee
import sys
import json
from datetime import datetime, timedelta
from comprehensive_plastic_detection_ee import (
    initialize_earth_engine, create_roi_from_coordinates,
    download_multi_sensor_data_ee, calculate_comprehensive_indices_ee,
    create_water_mask_ee, detect_plastic_fdi_method_ee,
    detect_plastic_spectral_classification_ee, create_ensemble_detection_ee,
    calculate_area_statistics_ee
)

def quick_plastic_check(lat, lon, buffer_km=2, days_back=30):
    """
    Quick plastic detection check for specific coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        buffer_km: Buffer distance in kilometers (default: 2km)
        days_back: How many days back to analyze (default: 30)
    
    Returns:
        dict: Quick analysis results
    """
    print(f"Quick plastic check for coordinates: {lat:.6f}°N, {lon:.6f}°E")
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        return None
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    try:
        # Create ROI
        roi = create_roi_from_coordinates(lat, lon, buffer_km)
        
        # Download data
        data = download_multi_sensor_data_ee(roi, start_date, end_date)
        if data is None:
            return {'error': 'No satellite data available'}
        
        # Calculate indices
        indices = calculate_comprehensive_indices_ee(data)
        
        # Create water mask
        water_mask, _ = create_water_mask_ee(data['optical'])
        
        # Apply ensemble detection
        fdi_detection = detect_plastic_fdi_method_ee(indices, water_mask)
        spectral_detection, spectral_confidence = detect_plastic_spectral_classification_ee(indices, water_mask)
        ensemble_detection, _ = create_ensemble_detection_ee(
            fdi_detection, spectral_detection, spectral_confidence, water_mask
        )
        
        # Calculate area statistics
        area_stats = calculate_area_statistics_ee(ensemble_detection, roi)
        
        # Prepare results
        results = {
            'coordinates': {'lat': lat, 'lon': lon},
            'analysis_date': datetime.now().isoformat(),
            'data_period': {'start': start_date, 'end': end_date},
            'data_availability': {
                'sentinel2_images': data['s2_count'],
                'sentinel1_images': data['s1_count']
            },
            'detection_results': {
                'plastic_detected': area_stats['detected_area_m2'] > 0,
                'detected_area_m2': area_stats['detected_area_m2'],
                'detected_area_km2': area_stats['detected_area_km2'],
                'coverage_percentage': area_stats['coverage_percentage']
            },
            'confidence_level': 'medium' if area_stats['detected_area_m2'] > 100 else 'low'
        }
        
        return results
        
    except Exception as e:
        return {'error': str(e)}

def analyze_multiple_coordinates(coordinates_list):
    """
    Analyze multiple coordinate pairs for plastic detection
    
    Args:
        coordinates_list: List of [lat, lon] pairs
    
    Returns:
        list: Results for each coordinate pair
    """
    results = []
    
    for i, (lat, lon) in enumerate(coordinates_list):
        print(f"\nAnalyzing location {i+1}/{len(coordinates_list)}: {lat:.5f}°N, {lon:.5f}°E")
        
        result = quick_plastic_check(lat, lon)
        if result:
            results.append(result)
        else:
            results.append({
                'coordinates': {'lat': lat, 'lon': lon},
                'error': 'Analysis failed'
            })
    
    return results

def print_summary_report(results):
    """
    Print a summary report of the analysis results
    
    Args:
        results: Analysis results (single dict or list)
    """
    if isinstance(results, dict):
        results = [results]
    
    print("\n" + "=" * 60)
    print("PLASTIC DETECTION SUMMARY REPORT")
    print("=" * 60)
    
    total_locations = len(results)
    detected_locations = 0
    total_detected_area = 0
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"Location {i}: ERROR - {result['error']}")
            continue
        
        coords = result['coordinates']
        detection = result['detection_results']
        
        print(f"\nLocation {i}: {coords['lat']:.5f}°N, {coords['lon']:.5f}°E")
        
        if detection['plastic_detected']:
            detected_locations += 1
            total_detected_area += detection['detected_area_m2']
            print(f"  🚨 PLASTIC DETECTED: {detection['detected_area_m2']:.0f} m²")
            print(f"  Coverage: {detection['coverage_percentage']:.4f}%")
        else:
            print(f"  ✓ No significant plastic detected")
        
        # Data availability
        data_avail = result['data_availability']
        print(f"  Data: S2({data_avail['sentinel2_images']}) S1({data_avail['sentinel1_images']})")
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Locations analyzed: {total_locations}")
    print(f"  Locations with detection: {detected_locations}")
    print(f"  Total detected area: {total_detected_area:.0f} m²")
    print(f"  Detection rate: {detected_locations/total_locations*100:.1f}%")
    print("=" * 60)

def main():
    """Main function with command line support"""
    
    # Check for command line arguments
    if len(sys.argv) >= 3:
        try:
            lat = float(sys.argv[1])
            lon = float(sys.argv[2])
            buffer_km = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
            
            print(f"Command line analysis for {lat}°N, {lon}°E")
            
            result = quick_plastic_check(lat, lon, buffer_km)
            if result:
                print_summary_report(result)
                
                # Save to file
                filename = f"quick_analysis_{lat:.3f}N_{lon:.3f}E_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {filename}")
            
            return
            
        except ValueError:
            print("Error: Invalid coordinates provided")
            print("Usage: python quick_plastic_analysis.py <lat> <lon> [buffer_km]")
            return
    
    # Interactive mode
    print("Quick Plastic Detection Analysis")
    print("=" * 40)
    
    # Example coordinates (Romanian Black Sea coast)
    example_coordinates = [
        [44.21706925, 28.96504135],  # User provided coordinates
        [44.2500, 28.9500],          # Nearby location 1
        [44.1800, 29.0200],          # Nearby location 2
        [44.3000, 29.1000],          # Nearby location 3
    ]
    
    print("Analyzing example coordinates in Romanian Black Sea region...")
    
    results = analyze_multiple_coordinates(example_coordinates)
    print_summary_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"multi_location_analysis_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {filename}")
    
    # Check specifically for user's coordinates
    user_coords = example_coordinates[0]
    print(f"\n" + "=" * 60)
    print(f"SPECIFIC ANALYSIS FOR PROVIDED COORDINATES")
    print(f"Location: {user_coords[0]:.6f}°N, {user_coords[1]:.6f}°E")
    print("=" * 60)
    
    user_result = results[0] if results else None
    if user_result and not user_result.get('error'):
        detection = user_result['detection_results']
        if detection['plastic_detected']:
            print("🚨 PLASTIC ACCUMULATION FOUND!")
            print(f"Detected area: {detection['detected_area_m2']:.0f} m²")
            print("This location shows potential plastic debris accumulation.")
            print("Recommendation: Validate with high-resolution imagery or field survey.")
        else:
            print("✓ No significant plastic accumulation detected.")
            print("This could indicate:")
            print("  • Clean water conditions")
            print("  • Detection below algorithm sensitivity")  
            print("  • Temporary absence of debris")

if __name__ == "__main__":
    main()
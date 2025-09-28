#!/usr/bin/env python3
"""
Plastic Analyzer - Handles all plastic detection analysis and trends
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import calendar

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class PlasticAnalyzer:
    """Analyzes satellite data for plastic detection and trends"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_plastic_detection(self, data_files):
        """
        Analyze plastic presence from satellite data
        
        Returns:
            dict: Analysis results with probability and statistics
        """
        results = {
            'probability': 0,
            'confidence': 'Low',
            'statistics': {},
            'detection_pixels': 0,
            'total_pixels': 0
        }
        
        if not RASTERIO_AVAILABLE:
            print("⚠️  rasterio not available for analysis")
            return results
        
        # Analyze FDI (primary plastic indicator)
        if 'fdi' in data_files and data_files['fdi']:
            fdi_stats = self._analyze_fdi_data(data_files['fdi'])
            results['statistics']['fdi'] = fdi_stats
            
            # Calculate plastic probability from FDI
            fdi_probability = self._calculate_fdi_probability(fdi_stats)
            results['probability'] += fdi_probability * 0.6  # 60% weight
        
        # Analyze FAI (algae vs plastic discrimination)
        if 'fai' in data_files and data_files['fai']:
            fai_stats = self._analyze_fai_data(data_files['fai'])
            results['statistics']['fai'] = fai_stats
            
            # FAI helps distinguish plastic from algae
            fai_contribution = self._calculate_fai_contribution(fai_stats)
            results['probability'] += fai_contribution * 0.2  # 20% weight
        
        # Analyze SAR data (texture and backscatter)
        if 'vv' in data_files and 'vh' in data_files:
            sar_stats = self._analyze_sar_data(data_files['vv'], data_files['vh'])
            results['statistics']['sar'] = sar_stats
            
            # SAR contributes to detection confidence
            sar_contribution = self._calculate_sar_contribution(sar_stats)
            results['probability'] += sar_contribution * 0.2  # 20% weight
        
        # Normalize probability to 0-100%
        results['probability'] = min(max(results['probability'], 0), 100)
        
        # Determine confidence level
        if results['probability'] >= 70:
            results['confidence'] = 'High'
        elif results['probability'] >= 40:
            results['confidence'] = 'Medium'
        else:
            results['confidence'] = 'Low'
        
        return results
    
    def analyze_temporal_trend(self, latitude, longitude, area_km):
        """
        Analyze plastic trends over the past 12 months
        
        Returns:
            dict: Monthly trend data with probabilities
        """
        # Generate synthetic but realistic trend data based on seasonal patterns
        # In a real implementation, this would query historical Earth Engine data
        
        months = []
        probabilities = []
        current_date = datetime.now()
        
        # Generate 12 months of data
        for i in range(12):
            month_date = current_date - timedelta(days=30*i)
            month_name = calendar.month_abbr[month_date.month]
            months.insert(0, month_name)
            
            # Simulate seasonal variation (higher in summer, lower in winter)
            base_prob = 35  # Base probability
            seasonal_factor = 15 * np.sin((month_date.month - 3) * np.pi / 6)  # Peak in summer
            noise = np.random.uniform(-5, 5)  # Random variation
            
            probability = max(0, min(100, base_prob + seasonal_factor + noise))
            probabilities.insert(0, probability)
        
        return {
            'months': months,
            'probabilities': probabilities,
            'trend': 'stable',  # Could be 'increasing', 'decreasing', 'stable'
            'seasonal_pattern': True
        }
    
    def _analyze_fdi_data(self, fdi_path):
        """Analyze FDI (Floating Debris Index) data"""
        try:
            with rasterio.open(fdi_path) as src:
                data = src.read(1)
                
                # Remove invalid/masked pixels
                valid_data = data[~np.isnan(data) & (data != src.nodata)]
                
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'percentile_90': float(np.percentile(valid_data, 90)),
                    'percentile_95': float(np.percentile(valid_data, 95)),
                    'valid_pixels': len(valid_data)
                }
                
                return stats
                
        except Exception as e:
            print(f"⚠️  Error analyzing FDI data: {e}")
            return {}
    
    def _analyze_fai_data(self, fai_path):
        """Analyze FAI (Floating Algae Index) data"""
        try:
            with rasterio.open(fai_path) as src:
                data = src.read(1)
                valid_data = data[~np.isnan(data) & (data != src.nodata)]
                
                stats = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'valid_pixels': len(valid_data)
                }
                
                return stats
                
        except Exception as e:
            print(f"⚠️  Error analyzing FAI data: {e}")
            return {}
    
    def _analyze_sar_data(self, vv_path, vh_path):
        """Analyze SAR backscatter data"""
        try:
            stats = {}
            
            # Analyze VV polarization
            if vv_path:
                with rasterio.open(vv_path) as src:
                    vv_data = src.read(1)
                    valid_vv = vv_data[~np.isnan(vv_data) & (vv_data != src.nodata)]
                    stats['vv_mean'] = float(np.mean(valid_vv))
                    stats['vv_std'] = float(np.std(valid_vv))
            
            # Analyze VH polarization  
            if vh_path:
                with rasterio.open(vh_path) as src:
                    vh_data = src.read(1)
                    valid_vh = vh_data[~np.isnan(vh_data) & (vh_data != src.nodata)]
                    stats['vh_mean'] = float(np.mean(valid_vh))
                    stats['vh_std'] = float(np.std(valid_vh))
            
            # Calculate VH/VV ratio if both available
            if vv_path and vh_path:
                with rasterio.open(vv_path) as vv_src, rasterio.open(vh_path) as vh_src:
                    vv_data = vv_src.read(1)
                    vh_data = vh_src.read(1)
                    
                    # Convert from dB to linear for ratio calculation
                    vv_linear = 10**(vv_data/10)
                    vh_linear = 10**(vh_data/10)
                    
                    ratio = np.where(vv_linear > 0, vh_linear / vv_linear, 0)
                    valid_ratio = ratio[~np.isnan(ratio) & (ratio > 0)]
                    
                    if len(valid_ratio) > 0:
                        stats['vh_vv_ratio_mean'] = float(np.mean(valid_ratio))
                        stats['vh_vv_ratio_std'] = float(np.std(valid_ratio))
            
            return stats
            
        except Exception as e:
            print(f"⚠️  Error analyzing SAR data: {e}")
            return {}
    
    def _calculate_fdi_probability(self, fdi_stats):
        """Calculate plastic probability from FDI statistics"""
        if not fdi_stats or 'mean' not in fdi_stats:
            return 0
        
        mean_fdi = fdi_stats['mean']
        
        # FDI interpretation for plastic detection
        # Higher FDI values (closer to 0 or positive) indicate potential plastic
        if mean_fdi > -0.02:
            return 80  # High probability
        elif mean_fdi > -0.05:
            return 60  # Medium-high probability
        elif mean_fdi > -0.1:
            return 40  # Medium probability
        elif mean_fdi > -0.15:
            return 25  # Low-medium probability
        else:
            return 10   # Low probability
    
    def _calculate_fai_contribution(self, fai_stats):
        """Calculate FAI contribution to plastic vs algae discrimination"""
        if not fai_stats or 'mean' not in fai_stats:
            return 0
        
        mean_fai = fai_stats['mean']
        
        # FAI helps distinguish plastic from algae
        # Lower FAI with high FDI suggests plastic over algae
        if mean_fai < -0.01:
            return 10   # Supports plastic interpretation
        elif mean_fai < 0.01:
            return 5    # Neutral
        else:
            return -5   # Suggests algae over plastic
    
    def _calculate_sar_contribution(self, sar_stats):
        """Calculate SAR contribution to plastic detection"""
        if not sar_stats:
            return 0
        
        contribution = 0
        
        # VH/VV ratio analysis (plastic tends to have different polarimetric signature)
        if 'vh_vv_ratio_mean' in sar_stats:
            ratio = sar_stats['vh_vv_ratio_mean']
            if 0.1 <= ratio <= 0.3:  # Typical range for plastic debris
                contribution += 10
            else:
                contribution += 5
        
        return contribution
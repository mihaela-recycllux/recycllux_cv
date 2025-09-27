"""
Plastic detection analysis module.
Performs statistical and spectral analysis on downloaded satellite images.
"""

import os
import numpy as np
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import rasterio
    from rasterio import mask, features
    RASTERIO_AVAILABLE = True
except ImportError:
    print("Warning: rasterio not available. Some analysis features may be limited.")
    RASTERIO_AVAILABLE = False


class PlasticAnalyzer:
    """Analyzes satellite images for plastic detection indicators."""
    
    def __init__(self, output_dir: str):
        """Initialize the analyzer."""
        self.output_dir = output_dir
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def analyze_image(self, image_path: str, product: str, analysis_type: str = 'basic_stats') -> Optional[Dict[str, Any]]:
        """
        Analyze a satellite image based on its product type.
        
        Args:
            image_path: Path to the GeoTIFF file
            product: Product type (fdi, fai, ndvi, etc.)
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary containing analysis results
        """
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return None
        
        try:
            if analysis_type == 'plastic_detection':
                return self._analyze_plastic_indices(image_path, product)
            elif analysis_type == 'vegetation_analysis':
                return self._analyze_vegetation(image_path, product)
            elif analysis_type == 'water_analysis':
                return self._analyze_water(image_path, product)
            elif analysis_type == 'sar_analysis':
                return self._analyze_sar(image_path, product)
            else:
                return self._analyze_basic_stats(image_path, product)
                
        except Exception as e:
            print(f"❌ Error analyzing {image_path}: {e}")
            return None
    
    def _analyze_plastic_indices(self, image_path: str, product: str) -> Dict[str, Any]:
        """Analyze plastic detection indices (FDI, FAI)."""
        stats = self._get_image_statistics(image_path)
        if not stats:
            return None
        
        # Plastic-specific analysis
        analysis = {
            'product': product,
            'analysis_type': 'plastic_detection',
            'timestamp': datetime.now().isoformat(),
            'file_path': image_path,
            'statistics': stats
        }
        
        # Add plastic detection thresholds and interpretation
        if product.lower() == 'fdi':
            analysis['interpretation'] = self._interpret_fdi_values(stats)
            analysis['plastic_indicators'] = self._detect_plastic_hotspots_fdi(stats)
        elif product.lower() == 'fai':
            analysis['interpretation'] = self._interpret_fai_values(stats)
            analysis['plastic_indicators'] = self._detect_plastic_hotspots_fai(stats)
        
        # Save analysis results
        self._save_analysis_results(analysis, f"{product}_analysis.json")
        
        return analysis
    
    def _analyze_vegetation(self, image_path: str, product: str) -> Dict[str, Any]:
        """Analyze vegetation indices (NDVI)."""
        stats = self._get_image_statistics(image_path)
        if not stats:
            return None
        
        analysis = {
            'product': product,
            'analysis_type': 'vegetation_analysis',
            'timestamp': datetime.now().isoformat(),
            'file_path': image_path,
            'statistics': stats,
            'vegetation_health': self._assess_vegetation_health(stats)
        }
        
        self._save_analysis_results(analysis, f"{product}_analysis.json")
        return analysis
    
    def _analyze_water(self, image_path: str, product: str) -> Dict[str, Any]:
        """Analyze water indices (NDWI, MNDWI)."""
        stats = self._get_image_statistics(image_path)
        if not stats:
            return None
        
        analysis = {
            'product': product,
            'analysis_type': 'water_analysis',
            'timestamp': datetime.now().isoformat(),
            'file_path': image_path,
            'statistics': stats,
            'water_coverage': self._estimate_water_coverage(stats)
        }
        
        self._save_analysis_results(analysis, f"{product}_analysis.json")
        return analysis
    
    def _analyze_sar(self, image_path: str, product: str) -> Dict[str, Any]:
        """Analyze SAR data (VV, VH polarizations)."""
        stats = self._get_image_statistics(image_path)
        if not stats:
            return None
        
        analysis = {
            'product': product,
            'analysis_type': 'sar_analysis',
            'timestamp': datetime.now().isoformat(),
            'file_path': image_path,
            'statistics': stats,
            'surface_roughness': self._analyze_surface_roughness(stats)
        }
        
        self._save_analysis_results(analysis, f"{product}_analysis.json")
        return analysis
    
    def _analyze_basic_stats(self, image_path: str, product: str) -> Dict[str, Any]:
        """Perform basic statistical analysis."""
        stats = self._get_image_statistics(image_path)
        if not stats:
            return None
        
        analysis = {
            'product': product,
            'analysis_type': 'basic_stats',
            'timestamp': datetime.now().isoformat(),
            'file_path': image_path,
            'statistics': stats
        }
        
        self._save_analysis_results(analysis, f"{product}_analysis.json")
        return analysis
    
    def _get_image_statistics(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract basic statistics from a GeoTIFF image."""
        if not RASTERIO_AVAILABLE:
            return self._get_basic_file_stats(image_path)
        
        try:
            with rasterio.open(image_path) as src:
                # Read the data
                data = src.read(1, masked=True)  # Read first band with masking
                
                # Remove invalid values
                valid_data = data[~data.mask] if hasattr(data, 'mask') else data
                valid_data = valid_data[np.isfinite(valid_data)]
                
                if len(valid_data) == 0:
                    return None
                
                stats = {
                    'count': len(valid_data),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'median': float(np.median(valid_data)),
                    'std': float(np.std(valid_data)),
                    'percentiles': {
                        'p5': float(np.percentile(valid_data, 5)),
                        'p25': float(np.percentile(valid_data, 25)),
                        'p75': float(np.percentile(valid_data, 75)),
                        'p95': float(np.percentile(valid_data, 95))
                    },
                    'image_info': {
                        'width': src.width,
                        'height': src.height,
                        'crs': str(src.crs),
                        'transform': list(src.transform),
                        'bounds': list(src.bounds)
                    }
                }
                
                return stats
                
        except Exception as e:
            print(f"❌ Error reading image statistics: {e}")
            return None
    
    def _get_basic_file_stats(self, image_path: str) -> Dict[str, Any]:
        """Get basic file information when rasterio is not available."""
        try:
            stat_info = os.stat(image_path)
            return {
                'file_size_mb': stat_info.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            }
        except Exception:
            return {'error': 'Could not read file statistics'}
    
    def _interpret_fdi_values(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Interpret FDI (Floating Debris Index) values."""
        mean_fdi = stats.get('mean', 0)
        max_fdi = stats.get('max', 0)
        
        interpretation = {
            'overall_assessment': '',
            'plastic_likelihood': '',
            'recommendations': ''
        }
        
        if mean_fdi > 0.02:
            interpretation['overall_assessment'] = 'High potential for floating debris/plastics'
            interpretation['plastic_likelihood'] = 'High'
        elif mean_fdi > 0.01:
            interpretation['overall_assessment'] = 'Moderate potential for floating debris'
            interpretation['plastic_likelihood'] = 'Moderate'
        else:
            interpretation['overall_assessment'] = 'Low floating debris signal'
            interpretation['plastic_likelihood'] = 'Low'
        
        if max_fdi > 0.05:
            interpretation['recommendations'] = 'Investigate areas with highest FDI values for plastic accumulation'
        else:
            interpretation['recommendations'] = 'Monitor for changes over time'
            
        return interpretation
    
    def _interpret_fai_values(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Interpret FAI (Floating Algae Index) values."""
        mean_fai = stats.get('mean', 0)
        
        interpretation = {
            'overall_assessment': '',
            'algae_vs_plastic': '',
            'recommendations': ''
        }
        
        if mean_fai > 0.01:
            interpretation['overall_assessment'] = 'Significant floating material detected'
            interpretation['algae_vs_plastic'] = 'Compare with FDI to distinguish algae from plastics'
        else:
            interpretation['overall_assessment'] = 'Minimal floating material'
            interpretation['algae_vs_plastic'] = 'Low signal for both algae and plastics'
        
        interpretation['recommendations'] = 'Use in combination with FDI for accurate plastic detection'
        return interpretation
    
    def _detect_plastic_hotspots_fdi(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential plastic hotspots using FDI analysis."""
        p95 = stats.get('percentiles', {}).get('p95', 0)
        max_val = stats.get('max', 0)
        
        return {
            'hotspot_threshold': p95,
            'max_intensity': max_val,
            'hotspot_potential': 'High' if max_val > 0.05 else 'Moderate' if max_val > 0.02 else 'Low'
        }
    
    def _detect_plastic_hotspots_fai(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential plastic hotspots using FAI analysis."""
        p95 = stats.get('percentiles', {}).get('p95', 0)
        
        return {
            'floating_material_threshold': p95,
            'material_type': 'Needs FDI comparison for plastic vs. algae discrimination'
        }
    
    def _assess_vegetation_health(self, stats: Dict[str, Any]) -> Dict[str, str]:
        """Assess vegetation health from NDVI statistics."""
        mean_ndvi = stats.get('mean', 0)
        
        if mean_ndvi > 0.6:
            health = 'Very healthy vegetation'
        elif mean_ndvi > 0.3:
            health = 'Moderate vegetation'
        elif mean_ndvi > 0.1:
            health = 'Sparse vegetation'
        else:
            health = 'No significant vegetation or water/urban areas'
        
        return {'vegetation_status': health, 'ndvi_mean': mean_ndvi}
    
    def _estimate_water_coverage(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate water coverage from water indices."""
        mean_val = stats.get('mean', 0)
        
        coverage_estimate = {
            'water_likelihood': 'High' if mean_val > 0.3 else 'Moderate' if mean_val > 0.1 else 'Low',
            'mean_index_value': mean_val
        }
        
        return coverage_estimate
    
    def _analyze_surface_roughness(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze surface roughness from SAR data."""
        std_val = stats.get('std', 0)
        mean_val = stats.get('mean', 0)
        
        return {
            'surface_type': 'Rough surface' if std_val > 2 else 'Smooth surface',
            'backscatter_intensity': 'High' if mean_val > 0 else 'Low',
            'standard_deviation': std_val
        }
    
    def _save_analysis_results(self, analysis: Dict[str, Any], filename: str) -> None:
        """Save analysis results to JSON file."""
        try:
            output_path = os.path.join(self.analysis_dir, filename)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"  💾 Analysis saved: {output_path}")
        except Exception as e:
            print(f"  ⚠️  Could not save analysis: {e}")
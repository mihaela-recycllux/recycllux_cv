"""
Filter Manager for Plastic Detection

This module manages multiple detection filters and coordinates their execution.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Type
import logging
from .base_filter import BaseFilter
from .fdi_filter import FDIFilter
from .ndvi_filter import NDVIFilter
from .plastic_index_filter import PlasticIndexFilter

class FilterManager:
    """Manager class for coordinating multiple detection filters"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the filter manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Available filter classes
        self.filter_classes = {
            'fdi': FDIFilter,
            'ndvi': NDVIFilter,
            'plastic_index': PlasticIndexFilter
        }

        # Active filters
        self.active_filters = {}

        # Initialize default filters
        self._initialize_default_filters()

    def _initialize_default_filters(self):
        """Initialize default set of filters"""
        default_filters = ['fdi', 'ndvi', 'plastic_index']

        for filter_name in default_filters:
            if filter_name in self.filter_classes:
                filter_config = self.config.get('detection_thresholds', {}).get(filter_name, {})
                self.active_filters[filter_name] = self.filter_classes[filter_name](filter_config)

        self.logger.info(f"Initialized {len(self.active_filters)} filters: {list(self.active_filters.keys())}")

    def add_filter(self, name: str, filter_class: Type[BaseFilter], config: Optional[Dict] = None):
        """
        Add a custom filter

        Args:
            name: Filter name
            filter_class: Filter class
            config: Filter configuration
        """
        self.active_filters[name] = filter_class(config)
        self.logger.info(f"Added custom filter: {name}")

    def remove_filter(self, name: str):
        """Remove a filter"""
        if name in self.active_filters:
            del self.active_filters[name]
            self.logger.info(f"Removed filter: {name}")

    def calculate_indices(self, optical_data: np.ndarray, sar_data: Optional[np.ndarray] = None,
                         data_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate indices for all active filters

        Args:
            optical_data: Optical satellite data
            sar_data: SAR data (optional)
            data_mask: Valid data mask

        Returns:
            Dictionary of calculated indices
        """
        self.logger.info("Calculating indices for all active filters...")

        indices = {}

        for name, filter_obj in self.active_filters.items():
            try:
                index_data = filter_obj.calculate_index(optical_data, sar_data, data_mask)
                indices[name] = index_data
                self.logger.info(f"✓ Calculated {name} index")
            except Exception as e:
                self.logger.error(f"Failed to calculate {name} index: {e}")
                indices[name] = np.full_like(optical_data[:, :, 0], np.nan)

        self.logger.info(f"✓ Calculated {len(indices)} indices")
        return indices

    def detect_plastic_all_filters(self, indices: Dict[str, np.ndarray],
                                 water_mask: Optional[np.ndarray] = None) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Run plastic detection for all active filters

        Args:
            indices: Dictionary of calculated indices
            water_mask: Water mask for restricting detection

        Returns:
            Dictionary of detection results (mask, metadata) for each filter
        """
        self.logger.info("Running plastic detection for all filters...")

        detections = {}

        for name, filter_obj in self.active_filters.items():
            try:
                if name in indices:
                    mask, metadata = filter_obj.detect_plastic(indices[name], water_mask)
                    detections[name] = (mask, metadata)
                    self.logger.info(f"✓ {name} detection: {metadata.get('detections', 0)} pixels")
                else:
                    self.logger.warning(f"No index data available for {name} filter")
            except Exception as e:
                self.logger.error(f"Failed to run {name} detection: {e}")
                # Create empty result
                empty_mask = np.full_like(indices.get(name, np.zeros((10, 10))), np.nan)
                detections[name] = (empty_mask, {'error': str(e)})

        self.logger.info(f"✓ Completed detection for {len(detections)} filters")
        return detections

    def create_ensemble_detection(self, detections: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                                weights: Optional[Dict[str, float]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create ensemble detection by combining results from multiple filters

        Args:
            detections: Dictionary of detection results
            weights: Optional weights for each filter

        Returns:
            Tuple of (ensemble_mask, ensemble_metadata)
        """
        self.logger.info("Creating ensemble detection...")

        if not detections:
            self.logger.warning("No detection results available for ensemble")
            return np.array([]), {}

        # Default equal weights if not provided
        if weights is None:
            weights = {name: 1.0 / len(detections) for name in detections.keys()}

        # Get first mask to determine shape
        first_mask = list(detections.values())[0][0]
        ensemble_score = np.zeros_like(first_mask, dtype=float)

        # Combine detections with weights
        total_weight = 0
        valid_filters = 0

        for name, (mask, metadata) in detections.items():
            if name in weights and not np.all(np.isnan(mask)):
                weight = weights[name]
                # Convert mask to float and handle NaN values
                mask_float = np.nan_to_num(mask, 0)
                ensemble_score += mask_float * weight
                total_weight += weight
                valid_filters += 1

        # Normalize by total weight
        if total_weight > 0:
            ensemble_score /= total_weight

        # Create binary ensemble mask
        ensemble_mask = (ensemble_score > 0.5).astype(float)

        # Set areas with no valid data to NaN
        no_data_mask = np.all([np.isnan(mask) for mask, _ in detections.values()], axis=0)
        ensemble_mask[no_data_mask] = np.nan

        # Calculate ensemble statistics
        valid_detections = np.sum(ensemble_mask == 1)
        total_valid_pixels = np.sum(~np.isnan(ensemble_mask))
        detection_rate = (valid_detections / total_valid_pixels * 100) if total_valid_pixels > 0 else 0

        ensemble_metadata = {
            'filter_name': 'ensemble',
            'method': 'weighted_average',
            'weights': weights,
            'valid_filters': valid_filters,
            'detections': int(valid_detections),
            'detection_rate': detection_rate,
            'ensemble_score_range': {
                'min': float(np.nanmin(ensemble_score)),
                'max': float(np.nanmax(ensemble_score))
            }
        }

        self.logger.info(f"✓ Ensemble detection: {int(valid_detections)} pixels ({detection_rate:.2f}%)")

        return ensemble_mask, ensemble_metadata

    def get_filter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active filters"""
        info = {}
        for name, filter_obj in self.active_filters.items():
            info[name] = {
                'name': filter_obj.name,
                'config': filter_obj.config
            }
        return info
"""
Plastic Index Filter for Plastic Detection

This filter implements a custom plastic index that combines multiple spectral bands
to enhance plastic detection capabilities.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from .base_filter import BaseFilter

class PlasticIndexFilter(BaseFilter):
    """Custom Plastic Index filter combining multiple spectral bands"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("Plastic_Index", config)

        # Plastic index parameters
        self.threshold = self.config.get('threshold', 0.01)
        self.adaptive = self.config.get('adaptive', True)

    def calculate_index(self, optical_data: np.ndarray, sar_data: Optional[np.ndarray] = None,
                       data_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate custom Plastic Index

        Plastic Index = (Blue + Red) / (2 * Green)
        This index exploits the spectral properties of plastic materials.

        Args:
            optical_data: Optical bands [Blue, Green, Red, NIR, SWIR]
            sar_data: Not used for plastic index
            data_mask: Valid data mask

        Returns:
            Plastic index array
        """
        self.logger.info("Calculating Plastic Index...")

        # Extract bands
        if optical_data.shape[2] < 3:
            raise ValueError("Plastic Index requires at least 3 optical bands (B, G, R)")

        blue = optical_data[:, :, 0]   # Blue band
        green = optical_data[:, :, 1]  # Green band
        red = optical_data[:, :, 2]    # Red band

        # Calculate Plastic Index
        # PI = (Blue + Red) / (2 * Green)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        plastic_index = (blue + red) / (2 * green + epsilon)

        # Apply data mask
        plastic_index = self.apply_data_mask(plastic_index, data_mask)

        self.logger.info(f"✓ Plastic Index calculated - range: {np.nanmin(plastic_index):.4f} to {np.nanmax(plastic_index):.4f}")

        return plastic_index

    def detect_plastic(self, index_data: np.ndarray, water_mask: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect plastic using Plastic Index thresholding

        Args:
            index_data: Plastic Index data
            water_mask: Water mask to restrict detection
            **kwargs: Additional parameters

        Returns:
            Tuple of (detection_mask, metadata)
        """
        self.logger.info("Applying Plastic Index threshold detection...")

        pi_data = index_data.copy()

        # Determine threshold
        if self.adaptive and water_mask is not None:
            # Use adaptive threshold based on water area statistics
            water_pixels = (water_mask == 1) & (~np.isnan(pi_data))
            if np.sum(water_pixels) > 0:
                water_pi = pi_data[water_pixels]
                # Use 95th percentile as threshold
                adaptive_threshold = np.percentile(water_pi, 95)
                threshold = max(adaptive_threshold, 0.001)
                self.logger.info(f"Using adaptive Plastic Index threshold: {threshold:.4f} (95th percentile)")
            else:
                threshold = self.threshold
                self.logger.warning("No valid water pixels found, using fixed threshold")
        else:
            threshold = self.threshold

        # Create detection mask
        detection_mask = np.zeros_like(pi_data, dtype=float)

        # Apply threshold (only in water areas if water_mask provided)
        if water_mask is not None:
            valid_water = (water_mask == 1) & (~np.isnan(pi_data))
            detection_mask[valid_water & (pi_data > threshold)] = 1
            # Set land areas to NaN
            detection_mask[water_mask != 1] = np.nan
        else:
            detection_mask[pi_data > threshold] = 1

        # Count detections
        valid_detections = np.sum(detection_mask == 1)
        total_valid_pixels = np.sum(~np.isnan(detection_mask)) if water_mask is not None else np.sum(~np.isnan(pi_data))

        detection_rate = (valid_detections / total_valid_pixels * 100) if total_valid_pixels > 0 else 0

        self.logger.info(f"✓ Plastic Index detection: {int(valid_detections)} pixels detected ({detection_rate:.2f}%)")

        # Metadata
        metadata = {
            'filter_name': self.name,
            'threshold_used': threshold,
            'adaptive_threshold': self.adaptive,
            'detections': int(valid_detections),
            'detection_rate': detection_rate,
            'plastic_index_stats': self.get_statistics(pi_data, water_mask)
        }

        return detection_mask, metadata
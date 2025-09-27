"""
NDVI (Normalized Difference Vegetation Index) Filter for Plastic Detection

This filter implements NDVI-based methods for detecting plastic debris and vegetation
in satellite imagery.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from .base_filter import BaseFilter

class NDVIFilter(BaseFilter):
    """Normalized Difference Vegetation Index filter"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("NDVI", config)

        # NDVI-specific parameters
        self.water_threshold = self.config.get('water_threshold', 0.0)
        self.vegetation_threshold = self.config.get('vegetation_threshold', 0.3)

    def calculate_index(self, optical_data: np.ndarray, sar_data: Optional[np.ndarray] = None,
                       data_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI)

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            optical_data: Optical bands [Blue, Green, Red, NIR, SWIR]
            sar_data: Not used for NDVI
            data_mask: Valid data mask

        Returns:
            NDVI index array
        """
        self.logger.info("Calculating Normalized Difference Vegetation Index (NDVI)...")

        # Extract bands
        if optical_data.shape[2] < 4:
            raise ValueError("NDVI requires at least 4 optical bands (B, G, R, NIR)")

        red = optical_data[:, :, 2]    # Red band
        nir = optical_data[:, :, 3]    # NIR band

        # Calculate NDVI
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ndvi = (nir - red) / (nir + red + epsilon)

        # Apply data mask
        ndvi = self.apply_data_mask(ndvi, data_mask)

        self.logger.info(f"✓ NDVI calculated - range: {np.nanmin(ndvi):.4f} to {np.nanmax(ndvi):.4f}")

        return ndvi

    def detect_plastic(self, index_data: np.ndarray, water_mask: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect plastic using NDVI-based methods

        For plastic detection, we look for areas that are not vegetation
        (low NDVI) in water areas.

        Args:
            index_data: NDVI data
            water_mask: Water mask to restrict detection
            **kwargs: Additional parameters

        Returns:
            Tuple of (detection_mask, metadata)
        """
        self.logger.info("Applying NDVI-based detection...")

        ndvi_data = index_data.copy()

        # Create detection mask
        detection_mask = np.zeros_like(ndvi_data, dtype=float)

        # Plastic detection logic:
        # 1. In water areas (if water_mask provided)
        # 2. NDVI below vegetation threshold (not vegetation)
        # 3. But could be plastic or other non-vegetated water features

        if water_mask is not None:
            # Focus on water areas
            water_pixels = (water_mask == 1) & (~np.isnan(ndvi_data))

            # Detect potential plastic areas (low NDVI in water)
            # This is a complementary method - low NDVI in water could indicate
            # plastic or other floating debris
            potential_plastic = water_pixels & (ndvi_data < self.vegetation_threshold)

            detection_mask[potential_plastic] = 1
            detection_mask[water_mask != 1] = np.nan
        else:
            # Without water mask, look for non-vegetation areas
            non_vegetation = (~np.isnan(ndvi_data)) & (ndvi_data < self.vegetation_threshold)
            detection_mask[non_vegetation] = 1

        # Count detections
        valid_detections = np.sum(detection_mask == 1)
        total_valid_pixels = np.sum(~np.isnan(detection_mask))

        detection_rate = (valid_detections / total_valid_pixels * 100) if total_valid_pixels > 0 else 0

        self.logger.info(f"✓ NDVI detection: {int(valid_detections)} potential plastic pixels detected ({detection_rate:.2f}%)")

        # Metadata
        metadata = {
            'filter_name': self.name,
            'vegetation_threshold': self.vegetation_threshold,
            'detections': int(valid_detections),
            'detection_rate': detection_rate,
            'ndvi_stats': self.get_statistics(ndvi_data, water_mask)
        }

        return detection_mask, metadata

    def create_water_mask(self, index_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Create water mask using NDVI

        Water typically has NDVI < 0, land/vegetation has NDVI > 0

        Args:
            index_data: NDVI data

        Returns:
            Water mask (1=water, 0=land)
        """
        self.logger.info("Creating water mask using NDVI...")

        ndvi_data = index_data.copy()

        # Water mask: NDVI <= water_threshold
        water_mask = np.zeros_like(ndvi_data, dtype=float)
        water_mask[ndvi_data <= self.water_threshold] = 1

        # Set invalid pixels to NaN
        water_mask[np.isnan(ndvi_data)] = np.nan

        water_pixels = np.sum(water_mask == 1)
        total_pixels = np.sum(~np.isnan(water_mask))
        water_percentage = (water_pixels / total_pixels * 100) if total_pixels > 0 else 0

        self.logger.info(f"✓ Water mask created: {water_pixels} water pixels ({water_percentage:.1f}%)")

        return water_mask
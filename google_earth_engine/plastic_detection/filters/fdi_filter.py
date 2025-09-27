"""
FDI (Floating Debris Index) Filter for Plastic Detection

This filter implements the Floating Debris Index method for detecting plastic debris
in water bodies using optical satellite imagery.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from .base_filter import BaseFilter

class FDIFilter(BaseFilter):
    """Floating Debris Index filter for plastic detection"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("FDI", config)

        # FDI-specific parameters
        self.threshold = self.config.get('threshold', 0.01)
        self.adaptive = self.config.get('adaptive', True)
        self.percentile = self.config.get('percentile', 95)

    def calculate_index(self, optical_data: np.ndarray, sar_data: Optional[np.ndarray] = None,
                       data_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate Floating Debris Index (FDI)

        FDI = NIR - baseline, where baseline is interpolated between Red and SWIR

        Args:
            optical_data: Optical bands [Blue, Green, Red, NIR, SWIR]
            sar_data: Not used for FDI
            data_mask: Valid data mask

        Returns:
            FDI index array
        """
        self.logger.info("Calculating Floating Debris Index (FDI)...")

        # Extract bands - assuming order: Blue, Green, Red, NIR, SWIR
        if optical_data.shape[2] < 5:
            raise ValueError("FDI requires at least 5 optical bands (B, G, R, NIR, SWIR)")

        red = optical_data[:, :, 2]    # Red band
        nir = optical_data[:, :, 3]    # NIR band
        swir = optical_data[:, :, 4]   # SWIR band

        # Calculate FDI
        # FDI = NIR - baseline, where baseline = Red + (SWIR - Red) * wavelength_factor
        # wavelength_factor = (λ_NIR - λ_Red) / (λ_SWIR - λ_Red)
        lambda_red = 665.0    # Red wavelength (nm)
        lambda_nir = 842.0    # NIR wavelength (nm)
        lambda_swir = 1610.0  # SWIR wavelength (nm)

        wavelength_factor = (lambda_nir - lambda_red) / (lambda_swir - lambda_red)
        baseline = red + (swir - red) * wavelength_factor

        fdi = nir - baseline

        # Apply data mask
        fdi = self.apply_data_mask(fdi, data_mask)

        self.logger.info(f"✓ FDI calculated - range: {np.nanmin(fdi):.4f} to {np.nanmax(fdi):.4f}")

        return fdi

    def detect_plastic(self, index_data: np.ndarray, water_mask: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect plastic using FDI thresholding

        Args:
            index_data: FDI data
            water_mask: Water mask to restrict detection to water areas
            **kwargs: Additional parameters

        Returns:
            Tuple of (detection_mask, metadata)
        """
        self.logger.info("Applying FDI threshold detection...")

        fdi_data = index_data.copy()

        # Determine threshold
        if self.adaptive and water_mask is not None:
            # Use adaptive threshold based on water area statistics
            water_pixels = (water_mask == 1) & (~np.isnan(fdi_data))
            if np.sum(water_pixels) > 0:
                water_fdi = fdi_data[water_pixels]
                adaptive_threshold = np.percentile(water_fdi, self.percentile)
                # Ensure minimum threshold
                threshold = max(adaptive_threshold, 0.001)
                self.logger.info(f"Using adaptive FDI threshold: {threshold:.4f} ({self.percentile}th percentile)")
            else:
                threshold = self.threshold
                self.logger.warning("No valid water pixels found, using fixed threshold")
        else:
            threshold = self.threshold

        # Create detection mask
        detection_mask = np.zeros_like(fdi_data, dtype=float)

        # Apply threshold (only in water areas if water_mask provided)
        if water_mask is not None:
            valid_water = (water_mask == 1) & (~np.isnan(fdi_data))
            detection_mask[valid_water & (fdi_data > threshold)] = 1
            # Set land areas to NaN
            detection_mask[water_mask != 1] = np.nan
        else:
            detection_mask[fdi_data > threshold] = 1

        # Count detections
        valid_detections = np.sum(detection_mask == 1)
        total_valid_pixels = np.sum(~np.isnan(detection_mask)) if water_mask is not None else np.sum(~np.isnan(fdi_data))

        detection_rate = (valid_detections / total_valid_pixels * 100) if total_valid_pixels > 0 else 0

        self.logger.info(f"✓ FDI detection: {int(valid_detections)} pixels detected ({detection_rate:.2f}%)")

        # Metadata
        metadata = {
            'filter_name': self.name,
            'threshold_used': threshold,
            'adaptive_threshold': self.adaptive,
            'detections': int(valid_detections),
            'detection_rate': detection_rate,
            'fdi_stats': self.get_statistics(fdi_data, water_mask)
        }

        return detection_mask, metadata
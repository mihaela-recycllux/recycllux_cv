"""
Base Filter Module for Plastic Detection

This module defines the base filter class and common functionality for all detection filters.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any
import logging

class BaseFilter(ABC):
    """Abstract base class for all plastic detection filters"""

    def __init__(self, name: str, config: Optional[Dict] = None):
        """
        Initialize the filter

        Args:
            name: Filter name
            config: Filter-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def calculate_index(self, optical_data: np.ndarray, sar_data: Optional[np.ndarray] = None,
                       data_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the index for this filter

        Args:
            optical_data: Optical satellite bands
            sar_data: SAR data (optional)
            data_mask: Valid data mask

        Returns:
            Calculated index array
        """
        pass

    @abstractmethod
    def detect_plastic(self, index_data: np.ndarray, water_mask: Optional[np.ndarray] = None,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect plastic using this filter

        Args:
            index_data: Calculated index data
            water_mask: Water mask for restricting detection
            **kwargs: Additional parameters

        Returns:
            Tuple of (detection_mask, metadata)
        """
        pass

    def apply_data_mask(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply data mask to array"""
        if mask is not None:
            masked_data = data.copy()
            masked_data[mask == 0] = np.nan
            return masked_data
        return data

    def get_statistics(self, data: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic statistics for data"""
        if mask is not None:
            valid_data = data[mask == 1]
        else:
            valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            return {'count': 0, 'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}

        return {
            'count': len(valid_data),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data))
        }
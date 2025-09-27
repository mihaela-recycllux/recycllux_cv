"""
Visualization Utilities for Plastic Detection

This module provides visualization functions for displaying detection results,
satellite imagery, and analysis outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional, Any
import os
from config.config import config

class VisualizationUtils:
    """Utilities for visualizing plastic detection results"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def create_rgb_composite(self, optical_data: np.ndarray, stretch_factor: float = 3.5) -> np.ndarray:
        """
        Create RGB composite from optical data

        Args:
            optical_data: Optical bands [Blue, Green, Red, NIR, SWIR]
            stretch_factor: RGB stretch factor

        Returns:
            RGB composite array (0-1 range)
        """
        if optical_data.shape[2] < 3:
            raise ValueError("RGB composite requires at least 3 bands")

        # Extract RGB bands
        blue = optical_data[:, :, 0]
        green = optical_data[:, :, 1]
        red = optical_data[:, :, 2]

        # Stack RGB
        rgb = np.stack([red, green, blue], axis=2)

        # Apply stretch and clip to 0-1 range
        rgb_stretched = np.clip(rgb * stretch_factor, 0, 1)

        return rgb_stretched

    def create_detection_visualization(self, rgb_composite: np.ndarray,
                                     detection_masks: Dict[str, np.ndarray],
                                     bbox: Dict[str, float],
                                     time_period: Tuple[str, str],
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive detection visualization

        Args:
            rgb_composite: RGB composite image
            detection_masks: Dictionary of detection masks
            bbox: Bounding box coordinates
            time_period: Time period tuple
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        viz_config = self.config.viz_params
        fig, axes = plt.subplots(2, 3, figsize=viz_config['figure_size'])
        fig.suptitle(f'Plastic Detection Results\n{time_period[0]} to {time_period[1]}',
                    fontsize=14, y=0.98)

        # Define colormap for detections
        colors = ['navy', 'red']
        cmap_binary = ListedColormap(colors)

        # 1. RGB Composite
        axes[0, 0].imshow(rgb_composite,
                         extent=[bbox['min_lon'], bbox['max_lon'],
                                bbox['min_lat'], bbox['max_lat']])
        axes[0, 0].set_title('RGB Composite', fontsize=11, pad=15)
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        axes[0, 0].grid(True, alpha=0.3)

        # 2-4. Detection Results
        detection_names = list(detection_masks.keys())[:3]  # Show up to 3 detections

        for i, name in enumerate(detection_names):
            row, col = divmod(i + 1, 3)
            ax = axes[row, col]

            mask = detection_masks[name]
            ax.imshow(mask, cmap=cmap_binary, vmin=0, vmax=1,
                     extent=[bbox['min_lon'], bbox['max_lon'],
                            bbox['min_lat'], bbox['max_lat']])
            ax.set_title(f'{name.upper()} Detection', fontsize=11, pad=15)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)

        # 5. RGB with Detection Overlay
        if detection_masks:
            ax = axes[1, 2]
            ax.imshow(rgb_composite,
                     extent=[bbox['min_lon'], bbox['max_lon'],
                            bbox['min_lat'], bbox['max_lat']])

            # Overlay first detection mask
            first_mask = list(detection_masks.values())[0]
            overlay = np.ma.masked_where(first_mask != 1, first_mask)
            ax.imshow(overlay, cmap='Reds', alpha=0.7, vmin=0, vmax=1,
                     extent=[bbox['min_lon'], bbox['max_lon'],
                            bbox['min_lat'], bbox['max_lat']])

            ax.set_title('RGB + Detection Overlay', fontsize=11, pad=15)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=viz_config['dpi'], bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")

        return fig

    def create_index_analysis_plot(self, indices: Dict[str, np.ndarray],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create index distribution analysis plot

        Args:
            indices: Dictionary of calculated indices
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        viz_config = self.config.viz_params
        fig, axes = plt.subplots(2, 2, figsize=viz_config['figure_size'])
        fig.suptitle('Index Distribution Analysis', fontsize=14, y=0.98)

        # Plot histograms for key indices
        index_names = list(indices.keys())[:4]  # Show up to 4 indices

        for i, name in enumerate(index_names):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            index_data = indices[name]
            valid_data = index_data[~np.isnan(index_data)].flatten()

            if len(valid_data) > 1000:  # Sample for performance
                valid_data = np.random.choice(valid_data, 1000, replace=False)

            ax.hist(valid_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{name.upper()} Distribution', fontsize=11, pad=15)
            ax.set_xlabel('Index Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            # Add statistics
            if len(valid_data) > 0:
                mean_val = np.mean(valid_data)
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_val:.3f}')
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=viz_config['dpi'], bbox_inches='tight')
            print(f"✓ Index analysis saved to: {save_path}")

        return fig

    def create_detection_comparison_plot(self, detections: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detection method comparison plot

        Args:
            detections: Dictionary of detection results
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        viz_config = self.config.viz_params

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Detection Method Comparison', fontsize=14, y=0.98)

        # Bar chart of detection counts
        methods = []
        counts = []

        for name, (mask, metadata) in detections.items():
            methods.append(name.upper())
            counts.append(metadata.get('detections', 0))

        bars = ax1.bar(methods, counts, color=['skyblue', 'lightgreen', 'orange', 'red'])
        ax1.set_title('Detection Counts by Method', fontsize=11, pad=15)
        ax1.set_ylabel('Detected Pixels')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                    str(count), ha='center', va='bottom', fontsize=9)

        # Detection rates
        rates = []
        for name, (mask, metadata) in detections.items():
            rates.append(metadata.get('detection_rate', 0))

        bars2 = ax2.bar(methods, rates, color=['lightcoral', 'gold', 'lightsteelblue', 'salmon'])
        ax2.set_title('Detection Rates (%)', fontsize=11, pad=15)
        ax2.set_ylabel('Detection Rate (%)')
        ax2.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, rate in zip(bars2, rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rates)*0.02,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=viz_config['dpi'], bbox_inches='tight')
            print(f"✓ Detection comparison saved to: {save_path}")

        return fig

    def save_detection_results(self, detections: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                             indices: Dict[str, np.ndarray],
                             metadata: Dict[str, Any],
                             output_dir: str = 'data'):
        """
        Save all detection results to files

        Args:
            detections: Detection results
            indices: Calculated indices
            metadata: Additional metadata
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save detection masks
        for name, (mask, det_metadata) in detections.items():
            mask_file = os.path.join(output_dir, f'{name}_detection_mask.npy')
            np.save(mask_file, mask)

            # Save metadata
            metadata_file = os.path.join(output_dir, f'{name}_detection_metadata.json')
            import json
            with open(metadata_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_metadata = {}
                for key, value in det_metadata.items():
                    if isinstance(value, np.ndarray):
                        json_metadata[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_metadata[key] = value.item()
                    else:
                        json_metadata[key] = value
                json.dump(json_metadata, f, indent=2)

        # Save indices
        for name, index_data in indices.items():
            index_file = os.path.join(output_dir, f'{name}_index.npy')
            np.save(index_file, index_data)

        # Save overall metadata
        overall_metadata_file = os.path.join(output_dir, 'detection_summary.json')
        with open(overall_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"✓ Results saved to: {output_dir}")
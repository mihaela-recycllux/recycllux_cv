#!/usr/bin/env python3
"""
Visualization and collage creation for satellite data.

Author: Varun Burde
Email: varun@recycllux.com
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import os
from datetime import datetime
from config.settings import Settings
from utils.ee_utils import get_safe_thumbnail_url, create_timestamp

class SatelliteVisualizer:
    """Create visualizations and collages from satellite data"""
    
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or Settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.timestamp = create_timestamp()
    
    def download_thumbnail(self, image, roi, vis_params=None, retries=3):
        """Download thumbnail image from Earth Engine"""
        thumbnail_url = get_safe_thumbnail_url(image, roi, vis_params)
        
        if not thumbnail_url:
            return None
            
        for attempt in range(retries):
            try:
                response = requests.get(thumbnail_url, timeout=30)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    return np.array(img)
                else:
                    print(f"HTTP {response.status_code} for thumbnail download")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed to download thumbnail: {e}")
                if attempt < retries - 1:
                    continue
        return None
    
    def create_collage(self, images_data, roi, title="Satellite Data Collage", 
                      save_path=None, figsize=(20, 16)):
        """
        Create a collage of satellite images
        
        Args:
            images_data: List of tuples (image, title, vis_params)
            roi: Region of interest
            title: Main title for the collage
            save_path: Path to save the collage
            figsize: Figure size
        """
        print(f"Creating collage with {len(images_data)} images...")
        
        # Calculate grid dimensions
        n_images = len(images_data)
        if n_images <= 4:
            rows, cols = 2, 2
        elif n_images <= 6:
            rows, cols = 2, 3
        elif n_images <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.2)
        
        # Add main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
        
        # Process each image
        for i, (image, img_title, vis_params) in enumerate(images_data):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])
            
            # Download thumbnail
            img_array = self.download_thumbnail(image, roi, vis_params)
            
            if img_array is not None:
                ax.imshow(img_array)
                ax.set_title(img_title, fontsize=12, fontweight='bold')
            else:
                # Create placeholder for failed downloads
                ax.text(0.5, 0.5, 'Failed to\nload image', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax.set_title(img_title, fontsize=12, fontweight='bold', color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        # Add metadata
        metadata_text = (
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Region: {self._format_roi(roi)}\n"
            f"Total Images: {n_images}"
        )
        fig.text(0.02, 0.02, metadata_text, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save collage
        if not save_path:
            save_path = os.path.join(self.output_dir, f"satellite_collage_{self.timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Collage saved to: {save_path}")
        
        plt.show()
        return save_path
    
    def create_time_series_collage(self, time_series_data, roi, 
                                 title="Time Series Analysis", save_path=None):
        """
        Create a collage showing time series data
        
        Args:
            time_series_data: List of tuples (image, date, vis_params)
            roi: Region of interest
            title: Main title
            save_path: Path to save
        """
        print(f"Creating time series collage...")
        
        n_images = len(time_series_data)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (image, date, vis_params) in enumerate(time_series_data):
            if i >= len(axes):
                break
                
            ax = axes[i]
            img_array = self.download_thumbnail(image, roi, vis_params)
            
            if img_array is not None:
                ax.imshow(img_array)
                ax.set_title(date, fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(date, fontsize=10, color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        if not save_path:
            save_path = os.path.join(self.output_dir, f"time_series_{self.timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Time series collage saved to: {save_path}")
        
        plt.show()
        return save_path
    
    def create_comparison_plot(self, before_after_data, roi, 
                              title="Before/After Comparison", save_path=None):
        """
        Create before/after comparison visualization
        
        Args:
            before_after_data: List of tuples (before_image, after_image, subtitle, vis_params)
            roi: Region of interest
            title: Main title
            save_path: Path to save
        """
        print("Creating before/after comparison...")
        
        n_comparisons = len(before_after_data)
        fig, axes = plt.subplots(n_comparisons, 2, figsize=(12, 6*n_comparisons))
        
        if n_comparisons == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (before_img, after_img, subtitle, vis_params) in enumerate(before_after_data):
            # Before image
            before_array = self.download_thumbnail(before_img, roi, vis_params)
            if before_array is not None:
                axes[i, 0].imshow(before_array)
            axes[i, 0].set_title(f"Before - {subtitle}")
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            
            # After image
            after_array = self.download_thumbnail(after_img, roi, vis_params)
            if after_array is not None:
                axes[i, 1].imshow(after_array)
            axes[i, 1].set_title(f"After - {subtitle}")
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
        
        if not save_path:
            save_path = os.path.join(self.output_dir, f"comparison_{self.timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {save_path}")
        
        plt.show()
        return save_path
    
    def create_index_dashboard(self, indices_data, roi, 
                              title="Spectral Indices Dashboard", save_path=None):
        """
        Create dashboard showing various spectral indices
        
        Args:
            indices_data: Dict of {index_name: (image, vis_params)}
            roi: Region of interest
            title: Main title
            save_path: Path to save
        """
        print("Creating indices dashboard...")
        
        n_indices = len(indices_data)
        if n_indices <= 4:
            rows, cols = 2, 2
        elif n_indices <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
        axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (index_name, (image, vis_params)) in enumerate(indices_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            img_array = self.download_thumbnail(image, roi, vis_params)
            
            if img_array is not None:
                im = ax.imshow(img_array)
                ax.set_title(index_name, fontsize=12, fontweight='bold')
                
                # Add colorbar for indices
                if vis_params and 'palette' in vis_params:
                    # Create a simple colorbar representation
                    pass  # Simplified for now
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(index_name, fontsize=12, color='red')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(n_indices, len(axes)):
            axes[i].axis('off')
        
        if not save_path:
            save_path = os.path.join(self.output_dir, f"indices_dashboard_{self.timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Indices dashboard saved to: {save_path}")
        
        plt.show()
        return save_path
    
    def create_plastic_detection_report(self, detection_data, roi,
                                       title="Plastic Detection Analysis Report", 
                                       save_path=None):
        """
        Create comprehensive plastic detection report
        
        Args:
            detection_data: Dict containing various detection results
            roi: Region of interest
            title: Main title
            save_path: Path to save
        """
        print("Creating plastic detection report...")
        
        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 3, hspace=0.4, wspace=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # Row 1: Original data
        if 'rgb' in detection_data:
            ax1 = fig.add_subplot(gs[0, 0])
            rgb_array = self.download_thumbnail(detection_data['rgb'][0], roi, detection_data['rgb'][1])
            if rgb_array is not None:
                ax1.imshow(rgb_array)
            ax1.set_title('RGB True Color', fontweight='bold')
            ax1.set_xticks([])
            ax1.set_yticks([])
        
        if 'false_color' in detection_data:
            ax2 = fig.add_subplot(gs[0, 1])
            fc_array = self.download_thumbnail(detection_data['false_color'][0], roi, detection_data['false_color'][1])
            if fc_array is not None:
                ax2.imshow(fc_array)
            ax2.set_title('False Color (NIR-R-G)', fontweight='bold')
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        if 'sar' in detection_data:
            ax3 = fig.add_subplot(gs[0, 2])
            sar_array = self.download_thumbnail(detection_data['sar'][0], roi, detection_data['sar'][1])
            if sar_array is not None:
                ax3.imshow(sar_array)
            ax3.set_title('SAR VV', fontweight='bold')
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # Row 2: Water detection
        if 'ndwi' in detection_data:
            ax4 = fig.add_subplot(gs[1, 0])
            ndwi_array = self.download_thumbnail(detection_data['ndwi'][0], roi, detection_data['ndwi'][1])
            if ndwi_array is not None:
                ax4.imshow(ndwi_array)
            ax4.set_title('NDWI (Water Detection)', fontweight='bold')
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        if 'mndwi' in detection_data:
            ax5 = fig.add_subplot(gs[1, 1])
            mndwi_array = self.download_thumbnail(detection_data['mndwi'][0], roi, detection_data['mndwi'][1])
            if mndwi_array is not None:
                ax5.imshow(mndwi_array)
            ax5.set_title('MNDWI (Modified Water Index)', fontweight='bold')
            ax5.set_xticks([])
            ax5.set_yticks([])
        
        # Row 3: Plastic detection indices
        if 'fdi' in detection_data:
            ax6 = fig.add_subplot(gs[2, 0])
            fdi_array = self.download_thumbnail(detection_data['fdi'][0], roi, detection_data['fdi'][1])
            if fdi_array is not None:
                ax6.imshow(fdi_array)
            ax6.set_title('FDI (Floating Debris Index)', fontweight='bold')
            ax6.set_xticks([])
            ax6.set_yticks([])
        
        if 'fai' in detection_data:
            ax7 = fig.add_subplot(gs[2, 1])
            fai_array = self.download_thumbnail(detection_data['fai'][0], roi, detection_data['fai'][1])
            if fai_array is not None:
                ax7.imshow(fai_array)
            ax7.set_title('FAI (Floating Algae Index)', fontweight='bold')
            ax7.set_xticks([])
            ax7.set_yticks([])
        
        # Row 4: Analysis summary
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Add analysis text
        analysis_text = self._generate_analysis_text(detection_data, roi)
        ax8.text(0.02, 0.95, analysis_text, transform=ax8.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        if not save_path:
            save_path = os.path.join(self.output_dir, f"plastic_detection_report_{self.timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ Plastic detection report saved to: {save_path}")
        
        plt.show()
        return save_path
    
    def _format_roi(self, roi):
        """Format ROI coordinates for display"""
        try:
            bounds = roi.bounds().getInfo()
            coords = bounds['coordinates'][0]
            min_lon = min(coord[0] for coord in coords)
            max_lon = max(coord[0] for coord in coords)
            min_lat = min(coord[1] for coord in coords)
            max_lat = max(coord[1] for coord in coords)
            return f"({min_lon:.3f}, {min_lat:.3f}) to ({max_lon:.3f}, {max_lat:.3f})"
        except:
            return "Custom region"
    
    def _generate_analysis_text(self, detection_data, roi):
        """Generate analysis text for the report"""
        text = "PLASTIC DETECTION ANALYSIS SUMMARY\n\n"
        text += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Region: {self._format_roi(roi)}\n\n"
        
        text += "METHODOLOGY:\n"
        text += "• RGB and False Color: Visual assessment of surface conditions\n"
        text += "• SAR Data: All-weather detection capability, surface roughness analysis\n"
        text += "• NDWI/MNDWI: Water body identification and masking\n"
        text += "• FDI: Floating Debris Index optimized for plastic detection\n"
        text += "• FAI: Floating Algae Index to distinguish organic vs. synthetic debris\n\n"
        
        text += "INTERPRETATION GUIDE:\n"
        text += "• Bright areas in FDI may indicate floating plastic debris\n"
        text += "• Compare with FAI to distinguish from natural organic matter\n"
        text += "• SAR can detect plastic patches through roughness changes\n"
        text += "• Water indices help focus analysis on aquatic areas\n\n"
        
        text += "LIMITATIONS:\n"
        text += "• Cloud cover may affect optical data quality\n"
        text += "• Small debris patches may not be detectable at satellite resolution\n"
        text += "• Atmospheric conditions can influence spectral signatures\n"
        text += "• Ground truth validation recommended for confirmation\n"
        
        return text
#!/usr/bin/env python3
"""
Pleiades Satellite Imagery Visualization Exporter
================================================

Exports high-quality visualization images for all modalities to separate folders.
Creates collages and individual tile visualizations that can be viewed with any image viewer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from pathlib import Path
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class PleiadesVisualizationExporter:
    def __init__(self, outputs_dir="/Users/varunburde/projects/Recyllux/outputs",
                 export_dir="/Users/varunburde/projects/Recyllux/visualizations"):
        """
        Initialize the visualization exporter

        Args:
            outputs_dir (str): Path to the outputs directory containing tiles
            export_dir (str): Path to export visualizations
        """
        self.outputs_dir = Path(outputs_dir)
        self.export_dir = Path(export_dir)
        self.tile_dirs = self._find_tile_dirs()

        # Create export directory
        self.export_dir.mkdir(exist_ok=True)

        print("PLEIADES VISUALIZATION EXPORTER")
        print("="*45)
        print(f"Output directory: {self.outputs_dir}")
        print(f"Export directory: {self.export_dir}")
        print(f"Tile directories found: {len(self.tile_dirs)}")

        # Create modality folders
        self.modality_dirs = {}
        modalities = ['panchromatic', 'rgb', 'false_color', 'nir', 'ndvi']
        for modality in modalities:
            modality_dir = self.export_dir / modality
            modality_dir.mkdir(exist_ok=True)
            self.modality_dirs[modality] = modality_dir

    def _find_tile_dirs(self):
        """Find all tile directories"""
        tile_dirs = []
        if self.outputs_dir.exists():
            for item in self.outputs_dir.iterdir():
                if item.is_dir() and 'tiles' in item.name:
                    tile_count = len(list(item.glob('*.tif')))
                    tile_dirs.append((item, tile_count))
        return tile_dirs

    def load_tile(self, tile_path):
        """Load a single tile"""
        try:
            with rasterio.open(tile_path) as src:
                tile_data = src.read()
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs),
                    'transform': src.transform,
                    'bounds': src.bounds
                }
                return tile_data, metadata
        except Exception as e:
            print(f"Error loading tile {tile_path}: {e}")
            return None, None

    def create_panchromatic_visualization(self, tile_data, metadata, output_path, title=""):
        """Create panchromatic visualization"""
        plt.figure(figsize=(12, 10), dpi=150)

        # Display panchromatic data
        plt.imshow(tile_data[0], cmap='gray')

        if title:
            plt.title(title, fontsize=14, pad=20)
        else:
            plt.title(f"Panchromatic Tile\n{metadata['width']}×{metadata['height']} pixels", fontsize=14, pad=20)

        plt.axis('off')

        # Add colorbar
        cbar = plt.colorbar(shrink=0.8, aspect=30)
        cbar.set_label('Digital Number (DN)', fontsize=12)

        # Add metadata text
        metadata_text = f"Size: {metadata['width']}×{metadata['height']} pixels\nData type: {tile_data.dtype}"
        plt.figtext(0.02, 0.02, metadata_text, fontsize=10, verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

    def create_multispectral_visualization(self, tile_data, metadata, modality, output_path, title=""):
        """Create multispectral visualization for different modalities"""
        plt.figure(figsize=(12, 10), dpi=150)

        if modality == 'rgb':
            # Natural color RGB composite (R, G, B)
            rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)  # R, G, B
            display_title = "Natural Color RGB"
            cmap = None

        elif modality == 'false_color':
            # False color composite (NIR, R, G)
            rgb = np.stack([tile_data[3], tile_data[2], tile_data[1]], axis=-1)  # NIR, R, G
            display_title = "False Color (NIR-R-G)"
            cmap = None

        elif modality == 'nir':
            # Near-infrared only - improved visualization
            nir_data = tile_data[3].astype(float)

            # Apply contrast stretching for better visualization
            p2, p98 = np.percentile(nir_data, (2, 98))
            nir_stretched = np.clip((nir_data - p2) / (p98 - p2), 0, 1)

            # Create a better NIR colormap (dark blue to bright yellow-green)
            colors = ['#000033', '#003366', '#006600', '#339933', '#66CC66', '#CCFF66', '#FFFF99']
            nir_cmap = LinearSegmentedColormap.from_list('nir', colors)
            rgb = nir_stretched
            cmap = nir_cmap
            display_title = "Near-Infrared (Enhanced)"

        elif modality == 'ndvi':
            # NDVI calculation: (NIR - RED) / (NIR + RED)
            nir = tile_data[3].astype(float)
            red = tile_data[2].astype(float)

            # Avoid division by zero
            denominator = nir + red
            ndvi = np.zeros_like(nir)
            mask = denominator != 0
            ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]

            # Create NDVI visualization
            ndvi_normalized = (ndvi + 1) / 2  # Normalize to 0-1
            rgb = plt.cm.RdYlGn(ndvi_normalized)[:, :, :3]  # Remove alpha channel
            display_title = "NDVI (Vegetation Index)"
            cmap = None

        # Display the image
        if cmap is not None:
            plt.imshow(rgb, cmap=cmap)
        else:
            plt.imshow(rgb)

        if title:
            plt.title(title, fontsize=14, pad=20)
        else:
            plt.title(f"{display_title}\n{metadata['width']}×{metadata['height']} pixels", fontsize=14, pad=20)

        plt.axis('off')

        # Add colorbar for single-band displays
        if modality in ['nir', 'ndvi']:
            cbar = plt.colorbar(shrink=0.8, aspect=30)
            if modality == 'nir':
                cbar.set_label('NIR Reflectance (stretched)', fontsize=12)
            elif modality == 'ndvi':
                cbar.set_label('NDVI (-1 to +1)', fontsize=12)

        # Add metadata text
        band_info = []
        for i in range(metadata['bands']):
            band_min, band_max = tile_data[i].min(), tile_data[i].max()
            band_info.append(f"Band {i+1}: {band_min}-{band_max}")

        metadata_text = f"Size: {metadata['width']}×{metadata['height']} pixels\n" + "\n".join(band_info[:2])  # Show first 2 bands
        plt.figtext(0.02, 0.02, metadata_text, fontsize=10, verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

    def create_modality_collage(self, modality, tile_dir, max_tiles=16):
        """Create a collage of tiles for a specific modality"""
        tile_files = sorted(list(tile_dir.glob('*.tif')))[:max_tiles]

        if not tile_files:
            return

        # Determine grid size
        n_tiles = len(tile_files)
        cols = int(np.ceil(np.sqrt(n_tiles)))
        rows = int(np.ceil(n_tiles / cols))

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=150)
        if rows == 1 or cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.ravel()

        # Load and display tiles
        for i, tile_file in enumerate(tile_files):
            if i >= len(axes):
                break

            try:
                tile_data, metadata = self.load_tile(str(tile_file))

                if tile_data is None:
                    continue

                if 'panchromatic' in tile_dir.name:
                    axes[i].imshow(tile_data[0], cmap='gray')
                else:  # multispectral
                    if modality == 'rgb':
                        rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
                    elif modality == 'false_color':
                        rgb = np.stack([tile_data[3], tile_data[2], tile_data[1]], axis=-1)
                    elif modality == 'nir':
                        nir_data = tile_data[3].astype(float)
                        p2, p98 = np.percentile(nir_data, (2, 98))
                        nir_stretched = np.clip((nir_data - p2) / (p98 - p2), 0, 1)
                        colors = ['#000033', '#003366', '#006600', '#339933', '#66CC66', '#CCFF66', '#FFFF99']
                        nir_cmap = LinearSegmentedColormap.from_list('nir', colors)
                        rgb = nir_stretched
                        axes[i].imshow(rgb, cmap=nir_cmap)
                        continue
                    elif modality == 'ndvi':
                        nir = tile_data[3].astype(float)
                        red = tile_data[2].astype(float)
                        denominator = nir + red
                        ndvi = np.zeros_like(nir)
                        mask = denominator != 0
                        ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]
                        ndvi_normalized = (ndvi + 1) / 2
                        rgb = plt.cm.RdYlGn(ndvi_normalized)[:, :, :3]
                    else:
                        rgb = np.stack([tile_data[0], tile_data[0], tile_data[0]], axis=-1)

                    # Normalize
                    rgb_normalized = rgb.astype(float)
                    for j in range(3):
                        band = rgb_normalized[:, :, j]
                        if band.max() > band.min():
                            rgb_normalized[:, :, j] = (band - band.min()) / (band.max() - band.min())

                    axes[i].imshow(rgb_normalized)

                axes[i].set_title(f"{tile_file.stem}", fontsize=8)
                axes[i].axis('off')

            except Exception as e:
                print(f"Error processing tile {tile_file}: {e}")
                axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(tile_files), len(axes)):
            axes[i].axis('off')

        # Set title
        dataset_name = tile_dir.name.replace('_tiles', '').replace('_multispectral', '').replace('_panchromatic', '')
        modality_title = modality.upper().replace('_', ' ')
        plt.suptitle(f'{modality_title} Collage - {dataset_name}\n{len(tile_files)} tiles', fontsize=16, y=0.98)

        plt.tight_layout()

        # Save collage
        collage_path = self.modality_dirs[modality] / f"collage_{dataset_name}_{modality}.png"
        plt.savefig(collage_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()

        print(f"Saved collage: {collage_path}")

    def export_all_visualizations(self):
        """Export all visualizations"""
        print("\nStarting visualization export...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for tile_dir, tile_count in self.tile_dirs:
            print(f"\nProcessing: {tile_dir.name} ({tile_count} tiles)")

            # Get sample tile for modality determination
            tile_files = sorted(list(tile_dir.glob('*.tif')))
            if not tile_files:
                continue

            sample_tile, _ = self.load_tile(str(tile_files[0]))
            if sample_tile is None:
                continue

            is_multispectral = sample_tile.shape[0] > 1

            if is_multispectral:
                # Export multispectral modalities
                modalities = ['rgb', 'false_color', 'nir', 'ndvi']
            else:
                # Export panchromatic
                modalities = ['panchromatic']

            # Export individual tiles (first few)
            max_individual = min(5, len(tile_files))  # Export first 5 tiles

            for i, tile_file in enumerate(tile_files[:max_individual]):
                tile_data, metadata = self.load_tile(str(tile_file))
                if tile_data is None:
                    continue

                tile_name = tile_file.stem

                for modality in modalities:
                    if modality == 'panchromatic':
                        output_path = self.modality_dirs[modality] / f"{tile_name}_{timestamp}.png"
                        self.create_panchromatic_visualization(
                            tile_data, metadata, output_path,
                            title=f"Panchromatic - {tile_name}"
                        )
                    else:
                        output_path = self.modality_dirs[modality] / f"{tile_name}_{modality}_{timestamp}.png"
                        self.create_multispectral_visualization(
                            tile_data, metadata, modality, output_path,
                            title=f"{modality.upper()} - {tile_name}"
                        )

                    print(f"  Exported: {output_path.name}")

            # Create collage for this dataset and modality
            for modality in modalities:
                self.create_modality_collage(modality, tile_dir)

        # Create overview HTML file
        self.create_overview_html()

        print("\n" + "="*60)
        print("EXPORT COMPLETE!")
        print("="*60)
        print(f"All visualizations saved to: {self.export_dir}")
        print("\nModality folders created:")
        for modality, path in self.modality_dirs.items():
            tile_count = len(list(path.glob('*.png')))
            print(f"  {modality.upper()}: {tile_count} images")

        print(f"\nOpen {self.export_dir}/overview.html in your browser to view all visualizations")

    def create_overview_html(self):
        """Create an HTML overview page"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pleiades Satellite Visualizations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .modality-section {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
        }}
        .modality-title {{
            font-size: 24px;
            color: #2980b9;
            margin-bottom: 15px;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .image-item {{
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            background-color: white;
        }}
        .image-item img {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }}
        .image-caption {{
            padding: 10px;
            font-size: 14px;
            color: #555;
        }}
        .stats {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .stats h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pleiades Satellite Imagery Visualizations</h1>
        <p style="text-align: center; color: #666;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="stats">
            <h3>Export Summary</h3>
            <ul>
"""

        # Add statistics
        for modality, path in self.modality_dirs.items():
            image_count = len(list(path.glob('*.png')))
            html_content += f"                <li><strong>{modality.upper()}:</strong> {image_count} images</li>\n"

        html_content += f"""
            </ul>
        </div>
"""

        # Add each modality section
        for modality, path in self.modality_dirs.items():
            images = list(path.glob('*.png'))
            if not images:
                continue

            modality_title = modality.upper().replace('_', ' ')

            html_content += f"""
        <div class="modality-section">
            <div class="modality-title">{modality_title}</div>
            <div class="image-grid">
"""

            # Sort images: collages first, then individual tiles
            collages = [img for img in images if 'collage' in img.name]
            individuals = [img for img in images if 'collage' not in img.name]

            for image in collages + individuals[:10]:  # Show collages + first 10 individuals
                rel_path = image.relative_to(self.export_dir)
                html_content += f"""
                <div class="image-item">
                    <img src="{rel_path}" alt="{image.stem}">
                    <div class="image-caption">{image.stem}</div>
                </div>
"""

            html_content += """
            </div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        # Save HTML file
        html_path = self.export_dir / "overview.html"
        with open(html_path, 'w') as f:
            f.write(html_content)

        print(f"Created overview HTML: {html_path}")


def main():
    """Main function"""
    exporter = PleiadesVisualizationExporter()

    if not exporter.tile_dirs:
        print("No tile directories found. Please run the tile processing script first:")
        print("python Test_scripts/load_ploygon_data.py")
        return

    exporter.export_all_visualizations()


if __name__ == "__main__":
    main()
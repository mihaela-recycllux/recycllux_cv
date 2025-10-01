#!/usr/bin/env python3
"""
Pleiades Satellite Imagery Visualization Demo
=============================================

Demonstrates key visualization capabilities for processed Pleiades tiles.
Shows examples of different visualization modes and comparisons.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.plot import show
from pathlib import Path
import glob
import warnings
warnings.filterwarnings('ignore')

class PleiadesDemoVisualizer:
    def __init__(self, outputs_dir="/Users/varunburde/projects/Recyllux/outputs"):
        """
        Initialize the demo visualizer

        Args:
            outputs_dir (str): Path to the outputs directory containing tiles
        """
        self.outputs_dir = Path(outputs_dir)
        self.tile_dirs = self._find_tile_dirs()

        print("PLEIADES VISUALIZATION DEMO")
        print("="*40)
        print(f"Output directory: {self.outputs_dir}")
        print(f"Tile directories found: {len(self.tile_dirs)}")

    def _find_tile_dirs(self):
        """Find all tile directories"""
        tile_dirs = []
        if self.outputs_dir.exists():
            for item in self.outputs_dir.iterdir():
                if item.is_dir() and 'tiles' in item.name:
                    tile_count = len(list(item.glob('*.tif')))
                    tile_dirs.append((item, tile_count))
        return tile_dirs

    def load_sample_tile(self, tile_dir):
        """Load a sample tile from a directory"""
        tile_files = sorted(list(tile_dir.glob('*.tif')))
        if tile_files:
            with rasterio.open(str(tile_files[0])) as src:
                return src.read(), {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'crs': str(src.crs),
                    'transform': src.transform
                }
        return None, None

    def demo_multispectral_modes(self):
        """Demonstrate different multispectral visualization modes"""
        print("\nDEMO: Multispectral Visualization Modes")
        print("-" * 40)

        # Find multispectral tile directory
        ms_dir = None
        for tile_dir, count in self.tile_dirs:
            if 'multispectral' in tile_dir.name and 'tiles' in tile_dir.name:
                ms_dir = tile_dir
                break

        if not ms_dir:
            print("No multispectral tiles found")
            return

        print(f"Using tiles from: {ms_dir.name}")

        # Load sample tile
        tile_data, metadata = self.load_sample_tile(ms_dir)
        if tile_data is None:
            print("Could not load tile")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        modes = [
            ('rgb', 'Natural Color RGB'),
            ('false_color', 'False Color (NIR-R-G)'),
            ('nir', 'Near-Infrared'),
            ('ndvi', 'NDVI (Vegetation Index)')
        ]

        for i, (mode, title) in enumerate(modes):
            ax = axes[i//2, i%2]

            if mode == 'rgb':
                rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
            elif mode == 'false_color':
                rgb = np.stack([tile_data[3], tile_data[2], tile_data[1]], axis=-1)
            elif mode == 'nir':
                rgb = np.stack([tile_data[3], tile_data[3], tile_data[3]], axis=-1)
            elif mode == 'ndvi':
                nir = tile_data[3].astype(float)
                red = tile_data[2].astype(float)
                denominator = nir + red
                ndvi = np.zeros_like(nir)
                mask = denominator != 0
                ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]
                ndvi_normalized = (ndvi + 1) / 2
                rgb = plt.cm.RdYlGn(ndvi_normalized)[:, :, :3]

            # Normalize for display (except NDVI)
            if mode != 'ndvi':
                rgb_normalized = rgb.astype(float)
                for j in range(3):
                    band = rgb_normalized[:, :, j]
                    if band.max() > band.min():
                        rgb_normalized[:, :, j] = (band - band.min()) / (band.max() - band.min())
                ax.imshow(rgb_normalized)
            else:
                ax.imshow(rgb)

            ax.set_title(f'{title}\n{metadata["width"]}×{metadata["height"]} pixels', fontsize=12)
            ax.axis('off')

        plt.suptitle('Multispectral Visualization Modes Comparison', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.show()

    def demo_panchromatic_vs_multispectral(self):
        """Compare panchromatic and multispectral data"""
        print("\nDEMO: Panchromatic vs Multispectral Comparison")
        print("-" * 50)

        # Find both types of tiles
        pan_dir = None
        ms_dir = None

        for tile_dir, count in self.tile_dirs:
            if 'panchromatic' in tile_dir.name and 'tiles' in tile_dir.name:
                pan_dir = tile_dir
            elif 'multispectral' in tile_dir.name and 'tiles' in tile_dir.name:
                ms_dir = tile_dir

        if not pan_dir or not ms_dir:
            print("Need both panchromatic and multispectral tiles for comparison")
            return

        print(f"Panchromatic: {pan_dir.name}")
        print(f"Multispectral: {ms_dir.name}")

        # Load sample tiles
        pan_data, pan_meta = self.load_sample_tile(pan_dir)
        ms_data, ms_meta = self.load_sample_tile(ms_dir)

        if pan_data is None or ms_data is None:
            print("Could not load tiles")
            return

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panchromatic
        axes[0].imshow(pan_data[0], cmap='gray')
        axes[0].set_title('Panchromatic\n(High Resolution)', fontsize=12)
        axes[0].axis('off')

        # Multispectral RGB
        rgb = np.stack([ms_data[2], ms_data[1], ms_data[0]], axis=-1)
        rgb_normalized = rgb.astype(float)
        for i in range(3):
            band = rgb_normalized[:, :, i]
            if band.max() > band.min():
                rgb_normalized[:, :, i] = (band - band.min()) / (band.max() - band.min())

        axes[1].imshow(rgb_normalized)
        axes[1].set_title('Multispectral RGB\n(Natural Color)', fontsize=12)
        axes[1].axis('off')

        # Multispectral False Color
        false_rgb = np.stack([ms_data[3], ms_data[2], ms_data[1]], axis=-1)
        false_rgb_normalized = false_rgb.astype(float)
        for i in range(3):
            band = false_rgb_normalized[:, :, i]
            if band.max() > band.min():
                false_rgb_normalized[:, :, i] = (band - band.min()) / (band.max() - band.min())

        axes[2].imshow(false_rgb_normalized)
        axes[2].set_title('Multispectral False Color\n(NIR-R-G)', fontsize=12)
        axes[2].axis('off')

        plt.suptitle('Image Type Comparison - Panchromatic vs Multispectral', fontsize=14)
        plt.tight_layout()
        plt.show()

        print(f"\nComparison Summary:")
        print(f"Panchromatic: {pan_meta['width']}×{pan_meta['height']} pixels, 1 band")
        print(f"Multispectral: {ms_meta['width']}×{ms_meta['height']} pixels, {ms_meta['bands']} bands")

    def demo_tile_mosaic(self):
        """Create a mosaic from multiple tiles"""
        print("\nDEMO: Tile Mosaic Creation")
        print("-" * 30)

        # Find a tile directory with multiple tiles
        mosaic_dir = None
        max_tiles = 0

        for tile_dir, count in self.tile_dirs:
            if count >= 9 and count > max_tiles:  # At least 3x3 mosaic
                mosaic_dir = tile_dir
                max_tiles = count

        if not mosaic_dir:
            print("No directory with sufficient tiles for mosaic")
            return

        print(f"Creating mosaic from: {mosaic_dir.name} ({max_tiles} tiles)")

        # Load first 9 tiles for 3x3 mosaic
        tile_files = sorted(list(mosaic_dir.glob('*.tif')))[:9]

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for i, tile_file in enumerate(tile_files):
            try:
                with rasterio.open(str(tile_file)) as src:
                    tile_data = src.read()

                    row, col = i // 3, i % 3

                    if 'panchromatic' in mosaic_dir.name:
                        axes[row, col].imshow(tile_data[0], cmap='gray')
                    else:  # multispectral
                        rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
                        rgb_normalized = rgb.astype(float)
                        for j in range(3):
                            band = rgb_normalized[:, :, j]
                            if band.max() > band.min():
                                rgb_normalized[:, :, j] = (band - band.min()) / (band.max() - band.min())
                        axes[row, col].imshow(rgb_normalized)

                    axes[row, col].set_title(f"{tile_file.stem}", fontsize=8)
                    axes[row, col].axis('off')

            except Exception as e:
                print(f"Error loading tile {tile_file}: {e}")

        image_type = 'Panchromatic' if 'panchromatic' in mosaic_dir.name else 'Multispectral RGB'
        plt.suptitle(f'Tile Mosaic - {image_type}\n{mosaic_dir.name}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def demo_statistics(self):
        """Show statistics and information about the tiles"""
        print("\nDEMO: Dataset Statistics")
        print("-" * 25)

        total_tiles = 0
        total_size_mb = 0

        print("Tile Directory Summary:")
        print("-" * 40)

        for tile_dir, count in self.tile_dirs:
            total_tiles += count

            # Get size of first tile as representative
            tile_files = list(tile_dir.glob('*.tif'))
            if tile_files:
                size_mb = os.path.getsize(str(tile_files[0])) / (1024 * 1024)
                total_size_mb += size_mb * count

                image_type = 'Multispectral' if 'multispectral' in tile_dir.name else 'Panchromatic'
                print("30")

        print("-" * 40)
        print(f"Total tiles: {total_tiles}")
        print(".1f")
        print(".1f")

        # Show data ranges for a sample tile
        if self.tile_dirs:
            sample_dir = self.tile_dirs[0][0]
            sample_files = list(sample_dir.glob('*.tif'))

            if sample_files:
                print(f"\nSample tile statistics from {sample_dir.name}:")
                with rasterio.open(str(sample_files[0])) as src:
                    sample_data = src.read()
                    print(f"  Dimensions: {src.width} × {src.height} pixels")
                    print(f"  Data type: {src.dtypes[0]}")
                    print(f"  Bands: {src.count}")

                    if src.count == 1:
                        print(f"  Data range: {sample_data[0].min()} - {sample_data[0].max()}")
                    else:
                        band_names = ['Blue', 'Green', 'Red', 'NIR']
                        for i in range(src.count):
                            band_name = band_names[i] if i < len(band_names) else f'Band {i+1}'
                            print("8")

    def run_all_demos(self):
        """Run all demonstration functions"""
        if not self.tile_dirs:
            print("No tile directories found. Please run the tile processing script first:")
            print("python Test_scripts/load_ploygon_data.py")
            return

        print("\nStarting Pleiades Visualization Demo...")
        print("This will create several visualization examples.")

        try:
            self.demo_statistics()
            self.demo_multispectral_modes()
            self.demo_panchromatic_vs_multispectral()
            self.demo_tile_mosaic()

            print("\n" + "="*50)
            print("DEMO COMPLETE!")
            print("="*50)
            print("All visualizations have been generated.")
            print("\nFor interactive exploration, run:")
            print("python Test_scripts/visualize_pleiades.py")

        except Exception as e:
            print(f"Error during demo: {e}")


def main():
    """Main function"""
    demo = PleiadesDemoVisualizer()
    demo.run_all_demos()


if __name__ == "__main__":
    main()
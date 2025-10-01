#!/usr/bin/env python3
"""
Pleiades Satellite Imagery Visualization
=======================================

Comprehensive visualization tool for processed Pleiades satellite imagery tiles.
Supports both panchromatic and multispectral data with various display options.
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

class PleiadesVisualizer:
    def __init__(self, outputs_dir="/Users/varunburde/projects/Recyllux/outputs"):
        """
        Initialize the Pleiades visualizer

        Args:
            outputs_dir (str): Path to the outputs directory containing tiles
        """
        self.outputs_dir = Path(outputs_dir)
        self.datasets = self._find_datasets()

        print("PLEIADES SATELLITE IMAGERY VISUALIZER")
        print("="*50)
        print(f"Output directory: {self.outputs_dir}")
        print(f"Datasets found: {len(self.datasets)}")
        for dataset in self.datasets:
            print(f"  - {dataset}")

    def _find_datasets(self):
        """Find all available datasets in the outputs directory"""
        datasets = []
        if self.outputs_dir.exists():
            for item in self.outputs_dir.iterdir():
                if item.is_dir() and 'polygon' in item.name.lower():
                    datasets.append(item.name)
        return sorted(datasets)

    def list_available_tiles(self, dataset_name=None):
        """List all available tile directories"""
        print("\nAVAILABLE TILE DIRECTORIES:")
        print("="*40)

        tile_dirs = []
        for item in self.outputs_dir.iterdir():
            if item.is_dir() and 'tiles' in item.name:
                tile_count = len(list(item.glob('*.tif')))
                print(f"  {item.name}: {tile_count} tiles")

                if dataset_name and dataset_name in item.name:
                    tile_dirs.append(item)

        return tile_dirs

    def load_tile(self, tile_path, show_info=True):
        """
        Load a single tile and return its data and metadata

        Args:
            tile_path (str): Path to the tile file
            show_info (bool): Whether to display tile information

        Returns:
            tuple: (tile_data, metadata_dict)
        """
        try:
            with rasterio.open(tile_path) as src:
                tile_data = src.read()
                metadata = {
                    'width': src.width,
                    'height': src.height,
                    'bands': src.count,
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs),
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'file_size_mb': os.path.getsize(tile_path) / (1024 * 1024)
                }

                if show_info:
                    print(f"Tile: {os.path.basename(tile_path)}")
                    print(f"  Dimensions: {src.width} x {src.height} pixels")
                    print(f"  Bands: {src.count}")
                    print(f"  Data type: {src.dtypes[0]}")
                    print(f"  Size: {metadata['file_size_mb']:.2f} MB")
                    if src.count == 1:
                        print(f"  Data range: {tile_data.min()} - {tile_data.max()}")
                    else:
                        for i in range(src.count):
                            print(f"  Band {i+1} range: {tile_data[i].min()} - {tile_data[i].max()}")

                return tile_data, metadata

        except Exception as e:
            print(f"Error loading tile {tile_path}: {e}")
            return None, None

    def visualize_panchromatic_tile(self, tile_data, metadata, title=None, figsize=(10, 8)):
        """
        Visualize a panchromatic tile

        Args:
            tile_data (numpy.ndarray): Tile data array
            metadata (dict): Tile metadata
            title (str): Plot title
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)

        # Display panchromatic data
        plt.imshow(tile_data[0], cmap='gray')

        if title:
            plt.title(title, fontsize=12)
        else:
            plt.title(f"Panchromatic Tile\n{metadata['width']}×{metadata['height']} pixels", fontsize=12)

        plt.axis('off')

        # Add colorbar
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label('Digital Number (DN)')

        plt.tight_layout()
        plt.show()

    def visualize_multispectral_tile(self, tile_data, metadata, mode='rgb', title=None, figsize=(12, 8)):
        """
        Visualize a multispectral tile in different modes

        Args:
            tile_data (numpy.ndarray): Tile data array (4 bands: B, G, R, NIR)
            metadata (dict): Tile metadata
            mode (str): Visualization mode ('rgb', 'false_color', 'nir', 'ndvi')
            title (str): Plot title
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)

        if mode == 'rgb':
            # Natural color RGB composite (R, G, B)
            rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)  # R, G, B
            display_title = "Natural Color RGB"

        elif mode == 'false_color':
            # False color composite (NIR, R, G)
            rgb = np.stack([tile_data[3], tile_data[2], tile_data[1]], axis=-1)  # NIR, R, G
            display_title = "False Color (NIR-R-G)"

        elif mode == 'nir':
            # Near-infrared only
            rgb = np.stack([tile_data[3], tile_data[3], tile_data[3]], axis=-1)  # NIR grayscale
            display_title = "Near-Infrared (NIR)"

        elif mode == 'ndvi':
            # NDVI calculation: (NIR - RED) / (NIR + RED)
            nir = tile_data[3].astype(float)
            red = tile_data[2].astype(float)

            # Avoid division by zero
            denominator = nir + red
            ndvi = np.zeros_like(nir)
            mask = denominator != 0
            ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]

            # Create NDVI colormap (red to green)
            ndvi_normalized = (ndvi + 1) / 2  # Normalize to 0-1
            rgb_normalized = plt.cm.RdYlGn(ndvi_normalized)[:, :, :3]  # Remove alpha channel
            display_title = "NDVI (Vegetation Index)"

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Normalize for display (except NDVI which is already normalized)
        if mode != 'ndvi':
            rgb_normalized = rgb.astype(float)
            for i in range(3):
                band = rgb_normalized[:, :, i]
                if band.max() > band.min():
                    rgb_normalized[:, :, i] = (band - band.min()) / (band.max() - band.min())

        plt.imshow(rgb_normalized)

        if title:
            plt.title(title, fontsize=12)
        else:
            plt.title(f"Multispectral Tile - {display_title}\n{metadata['width']}×{metadata['height']} pixels", fontsize=12)

        plt.axis('off')

        # Add colorbar for single-band displays
        if mode in ['nir', 'ndvi']:
            cbar = plt.colorbar(shrink=0.8)
            if mode == 'nir':
                cbar.set_label('NIR Reflectance (DN)')
            elif mode == 'ndvi':
                cbar.set_label('NDVI (-1 to +1)')

        plt.tight_layout()
        plt.show()

    def create_tile_mosaic(self, tile_dir, rows=3, cols=3, image_type='panchromatic',
                          mode='rgb', figsize=(15, 15)):
        """
        Create a mosaic from multiple tiles

        Args:
            tile_dir (str): Path to tile directory
            rows (int): Number of rows in mosaic
            cols (int): Number of columns in mosaic
            image_type (str): 'panchromatic' or 'multispectral'
            mode (str): Visualization mode for multispectral
            figsize (tuple): Figure size
        """
        tile_files = sorted(glob.glob(os.path.join(tile_dir, 'tile_*.tif')))

        if not tile_files:
            print(f"No tiles found in {tile_dir}")
            return

        # Select tiles for mosaic (first rows*cols tiles)
        mosaic_tiles = tile_files[:rows*cols]

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 or cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.ravel()

        for i, tile_path in enumerate(mosaic_tiles):
            if i >= len(axes):
                break

            try:
                tile_data, metadata = self.load_tile(tile_path, show_info=False)

                if tile_data is None:
                    continue

                if image_type == 'panchromatic':
                    axes[i].imshow(tile_data[0], cmap='gray')
                    title = f"Tile {os.path.basename(tile_path)[:-4]}"
                else:  # multispectral
                    if mode == 'rgb':
                        rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
                    elif mode == 'false_color':
                        rgb = np.stack([tile_data[3], tile_data[2], tile_data[1]], axis=-1)
                    else:
                        rgb = np.stack([tile_data[0], tile_data[0], tile_data[0]], axis=-1)

                    # Normalize
                    rgb_normalized = rgb.astype(float)
                    for j in range(3):
                        band = rgb_normalized[:, :, j]
                        if band.max() > band.min():
                            rgb_normalized[:, :, j] = (band - band.min()) / (band.max() - band.min())

                    axes[i].imshow(rgb_normalized)
                    title = f"{os.path.basename(tile_path)[:-4]}"

                axes[i].set_title(title, fontsize=8)
                axes[i].axis('off')

            except Exception as e:
                print(f"Error processing tile {tile_path}: {e}")
                axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(mosaic_tiles), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Tile Mosaic - {os.path.basename(tile_dir)}\n{image_type.upper()} - {mode.upper()}", fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_dataset_overview(self, dataset_name):
        """
        Show overview of a specific dataset

        Args:
            dataset_name (str): Name of the dataset
        """
        print(f"\nDATASET OVERVIEW: {dataset_name}")
        print("="*50)

        # Find tile directories for this dataset
        pan_tiles_dir = None
        ms_tiles_dir = None

        for item in self.outputs_dir.iterdir():
            if item.is_dir() and dataset_name in item.name and 'tiles' in item.name:
                if 'panchromatic' in item.name:
                    pan_tiles_dir = item
                elif 'multispectral' in item.name:
                    ms_tiles_dir = item

        if pan_tiles_dir:
            pan_tiles = list(pan_tiles_dir.glob('*.tif'))
            print(f"Panchromatic tiles: {len(pan_tiles)}")
            if pan_tiles:
                # Load first tile to show info
                tile_data, metadata = self.load_tile(str(pan_tiles[0]), show_info=False)
                print(f"  Tile size: {metadata['width']}×{metadata['height']} pixels")
                print(f"  File size: {metadata['file_size_mb']:.2f} MB per tile")

        if ms_tiles_dir:
            ms_tiles = list(ms_tiles_dir.glob('*.tif'))
            print(f"Multispectral tiles: {len(ms_tiles)}")
            if ms_tiles:
                # Load first tile to show info
                tile_data, metadata = self.load_tile(str(ms_tiles[0]), show_info=False)
                print(f"  Tile size: {metadata['width']}×{metadata['height']} pixels")
                print(f"  Bands: {metadata['bands']}")
                print(f"  File size: {metadata['file_size_mb']:.2f} MB per tile")

    def interactive_visualization_menu(self):
        """Interactive menu for visualization options"""
        while True:
            print("\n" + "="*60)
            print("PLEIADES VISUALIZATION MENU")
            print("="*60)
            print("1. List available datasets and tiles")
            print("2. Visualize single panchromatic tile")
            print("3. Visualize single multispectral tile")
            print("4. Create tile mosaic")
            print("5. Show dataset overview")
            print("6. Compare panchromatic vs multispectral")
            print("7. Exit")

            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == '1':
                self.list_available_tiles()

            elif choice == '2':
                self._visualize_single_tile('panchromatic')

            elif choice == '3':
                self._visualize_single_tile('multispectral')

            elif choice == '4':
                self._create_mosaic_interactive()

            elif choice == '5':
                if self.datasets:
                    print("\nAvailable datasets:")
                    for i, dataset in enumerate(self.datasets, 1):
                        print(f"{i}. {dataset}")
                    try:
                        idx = int(input("Select dataset number: ")) - 1
                        if 0 <= idx < len(self.datasets):
                            self.show_dataset_overview(self.datasets[idx])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Invalid input")
                else:
                    print("No datasets found")

            elif choice == '6':
                self._compare_image_types()

            elif choice == '7':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

    def _visualize_single_tile(self, image_type):
        """Helper method for single tile visualization"""
        tile_dirs = [item for item in self.outputs_dir.iterdir()
                    if item.is_dir() and image_type in item.name and 'tiles' in item.name]

        if not tile_dirs:
            print(f"No {image_type} tile directories found")
            return

        print(f"\nAvailable {image_type} tile directories:")
        for i, tile_dir in enumerate(tile_dirs, 1):
            tile_count = len(list(tile_dir.glob('*.tif')))
            print(f"{i}. {tile_dir.name} ({tile_count} tiles)")

        try:
            idx = int(input("Select directory number: ")) - 1
            if 0 <= idx < len(tile_dirs):
                selected_dir = tile_dirs[idx]
                tile_files = sorted(list(selected_dir.glob('*.tif')))

                if not tile_files:
                    print("No tiles found in selected directory")
                    return

                print(f"\nFirst few tiles in {selected_dir.name}:")
                for i, tile_file in enumerate(tile_files[:5], 1):
                    print(f"{i}. {tile_file.name}")

                try:
                    tile_idx = int(input("Select tile number to visualize: ")) - 1
                    if 0 <= tile_idx < len(tile_files):
                        tile_path = str(tile_files[tile_idx])
                        tile_data, metadata = self.load_tile(tile_path, show_info=True)

                        if tile_data is not None:
                            if image_type == 'panchromatic':
                                self.visualize_panchromatic_tile(tile_data, metadata)
                            else:  # multispectral
                                print("\nVisualization modes:")
                                print("1. RGB (Natural Color)")
                                print("2. False Color (NIR-R-G)")
                                print("3. Near-Infrared (NIR)")
                                print("4. NDVI (Vegetation Index)")

                                mode_choice = input("Select mode (1-4): ").strip()
                                mode_map = {'1': 'rgb', '2': 'false_color', '3': 'nir', '4': 'ndvi'}
                                mode = mode_map.get(mode_choice, 'rgb')

                                self.visualize_multispectral_tile(tile_data, metadata, mode=mode)
                    else:
                        print("Invalid tile selection")
                except ValueError:
                    print("Invalid input")
            else:
                print("Invalid directory selection")
        except ValueError:
            print("Invalid input")

    def _create_mosaic_interactive(self):
        """Helper method for interactive mosaic creation"""
        tile_dirs = [item for item in self.outputs_dir.iterdir()
                    if item.is_dir() and 'tiles' in item.name]

        if not tile_dirs:
            print("No tile directories found")
            return

        print("\nAvailable tile directories:")
        for i, tile_dir in enumerate(tile_dirs, 1):
            tile_count = len(list(tile_dir.glob('*.tif')))
            image_type = 'multispectral' if 'multispectral' in tile_dir.name else 'panchromatic'
            print(f"{i}. {tile_dir.name} ({tile_count} tiles, {image_type})")

        try:
            idx = int(input("Select directory number: ")) - 1
            if 0 <= idx < len(tile_dirs):
                selected_dir = tile_dirs[idx]
                image_type = 'multispectral' if 'multispectral' in selected_dir.name else 'panchromatic'

                rows = int(input("Number of rows in mosaic (default 3): ") or 3)
                cols = int(input("Number of columns in mosaic (default 3): ") or 3)

                if image_type == 'multispectral':
                    print("\nVisualization modes:")
                    print("1. RGB (Natural Color)")
                    print("2. False Color (NIR-R-G)")
                    mode_choice = input("Select mode (1-2, default 1): ").strip()
                    mode = 'false_color' if mode_choice == '2' else 'rgb'
                else:
                    mode = 'rgb'

                self.create_tile_mosaic(str(selected_dir), rows=rows, cols=cols,
                                      image_type=image_type, mode=mode)
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")

    def _compare_image_types(self):
        """Compare panchromatic and multispectral data from the same dataset"""
        # Find datasets that have both panchromatic and multispectral
        datasets_with_both = []
        for dataset in self.datasets:
            # Extract base dataset name (remove _panchromatic_pyramid, etc.)
            base_name = dataset.replace('_panchromatic_pyramid', '').replace('_multispectral_pyramid', '') \
                              .replace('_panchromatic_tiles', '').replace('_multispectral_tiles', '')

            has_pan = any('panchromatic' in item.name and base_name in item.name
                         for item in self.outputs_dir.iterdir() if item.is_dir())
            has_ms = any('multispectral' in item.name and base_name in item.name
                        for item in self.outputs_dir.iterdir() if item.is_dir())
            if has_pan and has_ms and base_name not in datasets_with_both:
                datasets_with_both.append(base_name)

        if not datasets_with_both:
            print("No datasets found with both panchromatic and multispectral data")
            return

        print("\nDatasets with both image types:")
        for i, dataset in enumerate(datasets_with_both, 1):
            print(f"{i}. {dataset}")

        try:
            idx = int(input("Select dataset number: ")) - 1
            if 0 <= idx < len(datasets_with_both):
                dataset = datasets_with_both[idx]

                # Find corresponding tile directories
                pan_dir = None
                ms_dir = None
                for item in self.outputs_dir.iterdir():
                    if item.is_dir() and dataset in item.name:
                        if 'panchromatic' in item.name and 'tiles' in item.name:
                            pan_dir = item
                        elif 'multispectral' in item.name and 'tiles' in item.name:
                            ms_dir = item

                if pan_dir and ms_dir:
                    # Load first tile from each
                    pan_tiles = list(pan_dir.glob('*.tif'))
                    ms_tiles = list(ms_dir.glob('*.tif'))

                    if pan_tiles and ms_tiles:
                        # Create comparison visualization
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                        # Panchromatic
                        pan_data, pan_meta = self.load_tile(str(pan_tiles[0]), show_info=False)
                        axes[0].imshow(pan_data[0], cmap='gray')
                        axes[0].set_title('Panchromatic\n(High Resolution)', fontsize=12)
                        axes[0].axis('off')

                        # Multispectral RGB
                        ms_data, ms_meta = self.load_tile(str(ms_tiles[0]), show_info=False)
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

                        plt.suptitle(f'Image Type Comparison - {dataset}', fontsize=14)
                        plt.tight_layout()
                        plt.show()

                        print(f"\nComparison for {dataset}:")
                        print(f"Panchromatic: {pan_meta['width']}×{pan_meta['height']} pixels, 1 band")
                        print(f"Multispectral: {ms_meta['width']}×{ms_meta['height']} pixels, {ms_meta['bands']} bands")
                    else:
                        print("No tiles found in selected directories")
                else:
                    print("Could not find both panchromatic and multispectral directories")
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")


def main():
    """Main function"""
    visualizer = PleiadesVisualizer()

    if not visualizer.datasets:
        print("No processed datasets found in outputs directory.")
        print("Please run the tile processing script first:")
        print("python Test_scripts/load_ploygon_data.py")
        return

    # Show available options
    visualizer.list_available_tiles()

    # Start interactive visualization
    visualizer.interactive_visualization_menu()


if __name__ == "__main__":
    main()
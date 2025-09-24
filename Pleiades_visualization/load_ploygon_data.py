#!/usr/bin/env python3
"""
Pleiades High-Resolution Image Loader
=====================================

This script loads and processes Pleiades satellite imagery data in DIMAP format.
It handles both panchromatic (high-resolution) and multispectral data.

Data Structure:
- DIMAP format with .JP2 image files
- Panchromatic: ~0.5m resolution, single band
- Multispectral: ~2m resolution, 4 bands (B, G, R, NIR)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.plot import show
import warnings
import math
warnings.filterwarnings('ignore')

class PleiadesLoader:
    def __init__(self, base_path):
        """
        Initialize Pleiades data loader

        Args:
            base_path (str): Path to the Pleiades dataset directory
        """
        self.base_path = Path(base_path)
        self.dataset_name = self.base_path.name

        # Find the main data directory (usually a number like 6979455101)
        data_dirs = [d for d in self.base_path.iterdir() if d.is_dir() and d.name.isdigit()]
        if not data_dirs:
            raise ValueError(f"No data directory found in {base_path}")
        self.data_dir = data_dirs[0]

        print(f"Loading Pleiades dataset: {self.dataset_name}")
        print(f"Data directory: {self.data_dir.name}")

    def find_image_files(self):
        """Find all image files in the dataset"""
        img_files = {}

        # Look for IMG_PHR1B directories
        for img_dir in self.data_dir.glob("IMG_PHR1B_*"):
            if img_dir.is_dir():
                # Find JP2 files
                jp2_files = list(img_dir.glob("*.JP2"))
                if jp2_files:
                    img_type = "panchromatic" if "P_001" in img_dir.name else "multispectral"
                    img_files[img_type] = {
                        'directory': img_dir,
                        'jp2_file': jp2_files[0],
                        'metadata': list(img_dir.glob("DIM_*.XML"))[0] if list(img_dir.glob("DIM_*.XML")) else None
                    }

        return img_files

    def load_image(self, image_type='panchromatic', show_info=True):
        """
        Load a specific image type

        Args:
            image_type (str): 'panchromatic' or 'multispectral'
            show_info (bool): Whether to display image information

        Returns:
            tuple: (image_data, metadata_dict)
        """
        img_files = self.find_image_files()

        if image_type not in img_files:
            available = list(img_files.keys())
            raise ValueError(f"Image type '{image_type}' not found. Available: {available}")

        img_info = img_files[image_type]
        jp2_path = img_info['jp2_file']

        print(f"\nLoading {image_type} image: {jp2_path.name}")

        # Open with rasterio
        with rasterio.open(jp2_path) as src:
            # Read all bands
            image_data = src.read()
            metadata = {
                'width': src.width,
                'height': src.height,
                'bands': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'transform': src.transform,
                'bounds': src.bounds,
                'resolution': (abs(src.transform[0]), abs(src.transform[4])),
                'file_size_mb': os.path.getsize(jp2_path) / (1024 * 1024)
            }

            if show_info:
                print(f"Image dimensions: {src.width} x {src.height} pixels")
                print(f"Number of bands: {src.count}")
                print(f"Data type: {src.dtypes[0]}")
                print(f"Spatial resolution: {metadata['resolution'][0]:.2f}m x {metadata['resolution'][1]:.2f}m")
                print(f"File size: {metadata['file_size_mb']:.1f} MB")
                print(f"Coordinate system: {src.crs}")
                print(f"Bounds: {src.bounds}")

            return image_data, metadata

    def split_and_save_tiles(self, image_type='panchromatic', tile_size=(2048, 2048),
                           output_dir=None, format='tiff', compression='lzw'):
        """
        Split large image into smaller tiles and save them

        Args:
            image_type (str): 'panchromatic' or 'multispectral'
            tile_size (tuple): Size of each tile (width, height)
            output_dir (str): Output directory for tiles (default: dataset_name_tiles)
            format (str): Output format ('tiff', 'jpeg', 'png')
            compression (str): Compression method for TIFF ('lzw', 'deflate', 'none')
        """
        from rasterio.windows import Window
        import math

        # Load image data
        img_data, metadata = self.load_image(image_type, show_info=False)

        # Create output directory
        if output_dir is None:
            output_dir = f"{self.dataset_name}_{image_type}_tiles"
        # Save to workspace instead of dataset directory
        workspace_path = Path("/Users/varunburde/projects/Recyllux")
        output_path = workspace_path / "outputs" / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        width, height = metadata['width'], metadata['height']
        tile_width, tile_height = tile_size

        # Calculate number of tiles
        n_tiles_x = math.ceil(width / tile_width)
        n_tiles_y = math.ceil(height / tile_height)

        print(f"\nSplitting {image_type} image into {n_tiles_x}x{n_tiles_y} tiles")
        print(f"Tile size: {tile_width}x{tile_height} pixels")
        print(f"Output directory: {output_path}")

        # Get original file path for reference
        img_files = self.find_image_files()
        original_path = img_files[image_type]['jp2_file']

        tiles_info = []

        with rasterio.open(original_path) as src:
            for i in range(n_tiles_x):
                for j in range(n_tiles_y):
                    # Calculate tile boundaries
                    x_start = i * tile_width
                    y_start = j * tile_height
                    x_end = min(x_start + tile_width, width)
                    y_end = min(y_start + tile_height, height)

                    # Create window for this tile
                    window = Window(x_start, y_start, x_end - x_start, y_end - y_start)

                    # Read tile data
                    tile_data = src.read(window=window)

                    # Create output filename
                    tile_filename = f"tile_{i+1:02d}_{j+1:02d}"

                    if format.lower() == 'tiff':
                        tile_filename += '.tif'
                        # Save as GeoTIFF with compression
                        profile = src.profile.copy()
                        profile.update({
                            'width': tile_data.shape[2],
                            'height': tile_data.shape[1],
                            'transform': rasterio.windows.transform(window, src.transform),
                            'compress': compression
                        })

                        with rasterio.open(output_path / tile_filename, 'w', **profile) as dst:
                            dst.write(tile_data)

                    elif format.lower() == 'jpeg':
                        tile_filename += '.jpg'
                        # For JPEG, we need to handle the data differently
                        if tile_data.shape[0] == 1:  # Panchromatic
                            plt.imsave(output_path / tile_filename, tile_data[0],
                                     cmap='gray', format='jpg')
                        else:  # Multispectral - save as RGB
                            rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
                            rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                            rgb_normalized = (rgb_normalized * 255).astype(np.uint8)
                            plt.imsave(output_path / tile_filename, rgb_normalized, format='jpg')

                    elif format.lower() == 'png':
                        tile_filename += '.png'
                        if tile_data.shape[0] == 1:  # Panchromatic
                            plt.imsave(output_path / tile_filename, tile_data[0],
                                     cmap='gray', format='png')
                        else:  # Multispectral - save as RGB
                            rgb = np.stack([tile_data[2], tile_data[1], tile_data[0]], axis=-1)
                            rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())
                            plt.imsave(output_path / tile_filename, rgb_normalized, format='png')

                    tiles_info.append({
                        'filename': tile_filename,
                        'position': (i, j),
                        'bounds': (x_start, y_start, x_end, y_end),
                        'size': (tile_data.shape[2], tile_data.shape[1])
                    })

                    print(f"Saved tile {i+1:02d}_{j+1:02d}")

        print(f"\nCompleted! Saved {len(tiles_info)} tiles to {output_path}")
        return tiles_info, str(output_path)

    def load_image_efficiently(self, image_type='panchromatic', max_size_mb=500):
        """
        Load image with memory-efficient approach for very large files

        Args:
            image_type (str): 'panchromatic' or 'multispectral'
            max_size_mb (int): Maximum memory to use in MB

        Returns:
            tuple: (image_data, metadata_dict)
        """
        img_files = self.find_image_files()
        jp2_path = img_files[image_type]['jp2_file']

        print(f"\nEfficiently loading {image_type} image: {jp2_path.name}")

        with rasterio.open(jp2_path) as src:
            # Calculate memory usage
            bytes_per_pixel = np.dtype(src.dtypes[0]).itemsize
            total_pixels = src.width * src.height * src.count
            estimated_mb = (total_pixels * bytes_per_pixel) / (1024 * 1024)

            print(f"Estimated memory usage: {estimated_mb:.1f} MB")

            if estimated_mb > max_size_mb:
                print(f"Warning: Image requires {estimated_mb:.1f} MB, exceeding limit of {max_size_mb} MB")
                print("Consider using split_and_save_tiles() instead")

                # Option to load at reduced resolution
                response = input("Load at reduced resolution? (y/n): ")
                if response.lower() == 'y':
                    # Calculate reduction factor
                    reduction_factor = math.ceil(estimated_mb / max_size_mb)
                    new_width = src.width // reduction_factor
                    new_height = src.height // reduction_factor

                    print(f"Reducing resolution by factor of {reduction_factor}")
                    print(f"New dimensions: {new_width} x {new_height}")

                    # Resample the image
                    image_data = src.read(
                        out_shape=(src.count, new_height, new_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                else:
                    print("Aborting load...")
                    return None, None
            else:
                # Load normally
                image_data = src.read()

            metadata = {
                'width': image_data.shape[2] if len(image_data.shape) > 2 else image_data.shape[1],
                'height': image_data.shape[1] if len(image_data.shape) > 2 else image_data.shape[0],
                'bands': src.count,
                'dtype': str(src.dtypes[0]),
                'crs': str(src.crs),
                'transform': src.transform,
                'bounds': src.bounds,
                'resolution': (abs(src.transform[0]), abs(src.transform[4])),
                'file_size_mb': os.path.getsize(jp2_path) / (1024 * 1024)
            }

            return image_data, metadata

    def create_image_pyramid(self, image_type='panchromatic', levels=4, output_dir=None):
        """
        Create an image pyramid with multiple resolution levels

        Args:
            image_type (str): 'panchromatic' or 'multispectral'
            levels (int): Number of pyramid levels
            output_dir (str): Output directory for pyramid files
        """
        import math

        # Load original image
        img_data, metadata = self.load_image(image_type, show_info=False)

        # Create output directory
        if output_dir is None:
            output_dir = f"{self.dataset_name}_{image_type}_pyramid"
        # Save to workspace instead of dataset directory
        workspace_path = Path("/Users/varunburde/projects/Recyllux")
        output_path = workspace_path / "outputs" / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating {levels}-level image pyramid")

        # Get original file path
        img_files = self.find_image_files()
        original_path = img_files[image_type]['jp2_file']

        pyramid_files = []

        with rasterio.open(original_path) as src:
            for level in range(levels):
                if level == 0:
                    # Original resolution
                    level_data = img_data
                    level_width, level_height = metadata['width'], metadata['height']
                else:
                    # Reduced resolution
                    reduction_factor = 2 ** level
                    level_width = metadata['width'] // reduction_factor
                    level_height = metadata['height'] // reduction_factor

                    print(f"Creating level {level}: {level_width} x {level_height} pixels")

                    level_data = src.read(
                        out_shape=(src.count, level_height, level_width),
                        resampling=rasterio.enums.Resampling.bilinear
                    )

                # Save level
                level_filename = f"level_{level}.tif"
                profile = src.profile.copy()
                profile.update({
                    'width': level_width,
                    'height': level_height,
                    'transform': src.transform * src.transform.scale(level_width / src.width, level_height / src.height),
                    'compress': 'lzw'
                })

                with rasterio.open(output_path / level_filename, 'w', **profile) as dst:
                    dst.write(level_data)

                pyramid_files.append({
                    'level': level,
                    'filename': level_filename,
                    'resolution_factor': 2 ** level,
                    'dimensions': (level_width, level_height)
                })

                print(f"Saved {level_filename}")

        print(f"\nPyramid created with {levels} levels in {output_path}")
        return pyramid_files, str(output_path)

    def compare_datasets(self, other_loader):
        """
        Compare two Pleiades datasets

        Args:
            other_loader (PleiadesLoader): Another PleiadesLoader instance
        """
        print("\n" + "="*60)
        print("DATASET COMPARISON")
        print("="*60)

        # Load panchromatic images from both datasets
        try:
            img1_data, img1_meta = self.load_image('panchromatic', show_info=False)
            img2_data, img2_meta = other_loader.load_image('panchromatic', show_info=False)

            print(f"Dataset 1: {self.dataset_name}")
            print(f"  Resolution: {img1_meta['resolution'][0]:.2f}m")
            print(f"  Dimensions: {img1_meta['width']} x {img1_meta['height']}")
            print(f"  File size: {img1_meta['file_size_mb']:.1f} MB")

            print(f"\nDataset 2: {other_loader.dataset_name}")
            print(f"  Resolution: {img2_meta['resolution'][0]:.2f}m")
            print(f"  Dimensions: {img2_meta['width']} x {img2_meta['height']}")
            print(f"  File size: {img2_meta['file_size_mb']:.1f} MB")

            # Check if they have the same dimensions
            if img1_meta['width'] == img2_meta['width'] and img1_meta['height'] == img2_meta['height']:
                print("\n✓ Datasets have matching dimensions - can be used together")
            else:
                print("\n⚠ Datasets have different dimensions")

        except Exception as e:
            print(f"Error comparing datasets: {e}")

def main():
    """Main function to demonstrate Pleiades data loading and processing"""

    # Define paths to your Pleiades datasets
    dataset1_path = "/Users/varunburde/Downloads/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298"
    dataset2_path = "/Users/varunburde/Downloads/Polygon1_SO24012539-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298"

    print("PLEIADES HIGH-RESOLUTION IMAGE PROCESSOR")
    print("="*55)
    print("Options:")
    print("1. Compare datasets")
    print("2. Load and visualize images")
    print("3. Split large images into tiles")
    print("4. Create image pyramid")
    print("5. Efficient loading with memory limits")

    try:
        # Initialize loaders
        loader1 = PleiadesLoader(dataset1_path)
        loader2 = PleiadesLoader(dataset2_path)

        # Option 3: Split images into tiles (recommended for large images)
        print("\n" + "="*60)
        print("SPLITTING LARGE IMAGES INTO MANAGEABLE TILES")
        print("="*60)

        # Split first dataset panchromatic image
        print(f"\nProcessing {loader1.dataset_name}:")
        tiles1, output_dir1 = loader1.split_and_save_tiles(
            image_type='panchromatic',
            tile_size=(2048, 2048),  # 2048x2048 pixel tiles
            format='tiff',
            compression='lzw'
        )

        # Split second dataset panchromatic image
        print(f"\nProcessing {loader2.dataset_name}:")
        tiles2, output_dir2 = loader2.split_and_save_tiles(
            image_type='panchromatic',
            tile_size=(2048, 2048),
            format='tiff',
            compression='lzw'
        )

        print("\nTile splitting completed!")
        print(f"Dataset 1 tiles: {output_dir1}")
        print(f"Dataset 2 tiles: {output_dir2}")
        print(f"Each tile: 2048x2048 pixels")
        print(f"Format: GeoTIFF with LZW compression")

        # Option 4: Create image pyramids for fast visualization
        print("\n" + "="*60)
        print("CREATING IMAGE PYRAMIDS FOR FAST VISUALIZATION")
        print("="*60)

        print(f"\nCreating pyramid for {loader1.dataset_name}:")
        pyramid1, pyramid_dir1 = loader1.create_image_pyramid(
            image_type='panchromatic',
            levels=4
        )

        print(f"\nCreating pyramid for {loader2.dataset_name}:")
        pyramid2, pyramid_dir2 = loader2.create_image_pyramid(
            image_type='panchromatic',
            levels=4
        )

        print("\nPyramid creation completed!")
        print("Pyramid levels:")
        for level_info in pyramid1:
            print(f"  Level {level_info['level']}: {level_info['dimensions'][0]}x{level_info['dimensions'][1]} "
                  f"({level_info['resolution_factor']}x reduction)")

        # Option 5: Demonstrate efficient loading
        print("\n" + "="*60)
        print("EFFICIENT LOADING WITH MEMORY MANAGEMENT")
        print("="*60)

        print("\nFor very large images, use load_image_efficiently():")
        print("- Automatically checks memory requirements")
        print("- Offers to reduce resolution if needed")
        print("- Prevents system memory exhaustion")

        # Show file sizes
        img_files1 = loader1.find_image_files()
        img_files2 = loader2.find_image_files()

        pan_path1 = img_files1['panchromatic']['jp2_file']
        pan_path2 = img_files2['panchromatic']['jp2_file']

        size1_mb = os.path.getsize(pan_path1) / (1024 * 1024)
        size2_mb = os.path.getsize(pan_path2) / (1024 * 1024)

        print("\nFile sizes:")
        print(f"  Dataset 1: {size1_mb:.1f} MB")
        print(f"  Dataset 2: {size2_mb:.1f} MB")

        print("\nUSAGE EXAMPLES:")
        print("="*30)
        print("# Load a specific tile:")
        print("from rasterio import open")
        print("with open('path/to/tile_01_01.tif') as src:")
        print("    tile_data = src.read()")
        print()
        print("# Load pyramid level for quick preview:")
        print("with open('path/to/level_2.tif') as src:")
        print("    preview_data = src.read()")
        print()
        print("# Process tiles in batches:")
        print("import glob")
        print("tile_files = glob.glob('path/to/tiles/*.tif')")
        print("for tile_path in tile_files:")
        print("    # Process each tile...")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("pip install rasterio matplotlib numpy")

        print("\nFor faster processing of large images, consider:")
        print("- Using SSD storage for temporary files")
        print("- Increasing system memory if possible")
        print("- Processing tiles in parallel (advanced)")

if __name__ == "__main__":
    # Process both Pleiades datasets with both panchromatic and multispectral images
    print("PLEIADES HIGH-RESOLUTION IMAGE PROCESSOR - FULL DATASET")
    print("="*65)
    print("Processing both datasets with panchromatic and multispectral images")
    print("="*65)

    # Define paths to both Pleiades datasets
    datasets = [
        "/Users/varunburde/Downloads/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298",
        "/Users/varunburde/Downloads/Polygon1_SO24012539-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298"
    ]

    image_types = ['panchromatic', 'multispectral']

    try:
        for dataset_path in datasets:
            print(f"\n{'='*60}")
            print(f"PROCESSING DATASET: {dataset_path.split('/')[-1]}")
            print(f"{'='*60}")

            # Initialize loader for this dataset
            loader = PleiadesLoader(dataset_path)

            # Check what image types are available
            available_types = []
            img_files = loader.find_image_files()
            for img_type in image_types:
                if img_type in img_files:
                    available_types.append(img_type)
                    print(f"✓ Found {img_type} data")
                else:
                    print(f"✗ {img_type} data not found")

            if not available_types:
                print(f"No image data found in {dataset_path}")
                continue

            # Process each available image type
            for image_type in available_types:
                print(f"\n{'─'*50}")
                print(f"PROCESSING {image_type.upper()} IMAGES")
                print(f"{'─'*50}")

                try:
                    # Split images into tiles
                    print(f"Splitting {image_type} images into tiles...")
                    tiles, output_dir = loader.split_and_save_tiles(
                        image_type=image_type,
                        tile_size=(2048, 2048),
                        format='tiff',
                        compression='lzw'
                    )

                    print(f"✓ Created {len(tiles)} {image_type} tiles in:")
                    print(f"  {output_dir}")

                    # Create image pyramid for fast visualization
                    print(f"\nCreating {image_type} image pyramid...")
                    pyramid, pyramid_dir = loader.create_image_pyramid(
                        image_type=image_type,
                        levels=4
                    )

                    print(f"✓ Created {len(pyramid)} pyramid levels in:")
                    print(f"  {pyramid_dir}")

                except Exception as e:
                    print(f"✗ Error processing {image_type} images: {e}")
                    continue

        # Summary
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")

        # Count total tiles created
        outputs_dir = Path("/Users/varunburde/projects/Recyllux/outputs")
        if outputs_dir.exists():
            total_tiles = 0
            total_size = 0

            for subdir in outputs_dir.iterdir():
                if subdir.is_dir() and 'tiles' in subdir.name:
                    tif_files = list(subdir.glob('*.tif'))
                    total_tiles += len(tif_files)

                    for tif_file in tif_files:
                        total_size += tif_file.stat().st_size

            print("\nSUMMARY:")
            print(f"  Total tiles created: {total_tiles}")
            print(f"  Total size: {total_size / (1024*1024):.1f} MB")
            print(f"  Output directory: {outputs_dir}")

            print("\nTILE DIRECTORIES:")
            for subdir in sorted(outputs_dir.iterdir()):
                if subdir.is_dir() and 'tiles' in subdir.name:
                    tif_count = len(list(subdir.glob('*.tif')))
                    print(f"  {subdir.name}: {tif_count} tiles")

        print("\nUSAGE:")
        print("  Run: python Test_scripts/tile_processing_example.py")
        print("  To work with the processed tiles")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have the required dependencies installed:")
        print("pip install rasterio matplotlib numpy")

        print("\nFor faster processing of large images, consider:")
        print("- Using SSD storage for temporary files")
        print("- Increasing system memory if possible")
        print("- Processing tiles in parallel (advanced)")

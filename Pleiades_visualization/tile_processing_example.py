#!/usr/bin/env python3
"""
Example: How to Load and Process Pleiades Image Tiles
====================================================

This script demonstrates how to work with the tiles created by the main processor.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.plot import show
import glob

def load_tile_example():
    """Example of loading a single tile"""

    # Path to your tiles directory
    tiles_dir = "/Users/varunburde/projects/Recyllux/outputs/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298_panchromatic_tiles"

    # Load a specific tile
    tile_path = os.path.join(tiles_dir, "tile_01_01.tif")

    print(f"Loading tile: {tile_path}")

    with rasterio.open(tile_path) as src:
        # Read the tile data
        tile_data = src.read(1)  # Read first (and only) band

        print(f"Tile dimensions: {src.width} x {src.height}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Data range: {tile_data.min()} - {tile_data.max()}")

        # Display the tile
        plt.figure(figsize=(8, 8))
        plt.imshow(tile_data, cmap='gray')
        plt.title("Pleiades Tile 01_01")
        plt.axis('off')
        plt.show()

def process_tiles_batch():
    """Example of processing multiple tiles in batch"""

    tiles_dir = "/Users/varunburde/projects/Recyllux/outputs/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298_panchromatic_tiles"

    # Find all tile files
    tile_files = glob.glob(os.path.join(tiles_dir, "tile_*.tif"))
    tile_files.sort()  # Sort for consistent ordering

    print(f"Found {len(tile_files)} tiles")

    # Process first 9 tiles as an example
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i, tile_path in enumerate(tile_files[:9]):
        with rasterio.open(tile_path) as src:
            tile_data = src.read(1)

            axes[i].imshow(tile_data, cmap='gray')
            axes[i].set_title(f"Tile {os.path.basename(tile_path)[:-4]}")
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_statistics():
    """Example of calculating statistics across tiles"""

    tiles_dir = "/Users/varunburde/projects/Recyllux/outputs/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298_panchromatic_tiles"

    tile_files = glob.glob(os.path.join(tiles_dir, "tile_*.tif"))

    stats = {
        'min_values': [],
        'max_values': [],
        'mean_values': [],
        'std_values': []
    }

    print("Calculating statistics across tiles...")

    for tile_path in tile_files[:10]:  # Process first 10 tiles as example
        with rasterio.open(tile_path) as src:
            tile_data = src.read(1).astype(float)

            stats['min_values'].append(tile_data.min())
            stats['max_values'].append(tile_data.max())
            stats['mean_values'].append(tile_data.mean())
            stats['std_values'].append(tile_data.std())

    print("\nStatistics across tiles:")
    print(f"Min values range: {min(stats['min_values']):.1f} - {max(stats['min_values']):.1f}")
    print(f"Max values range: {min(stats['max_values']):.1f} - {max(stats['max_values']):.1f}")
    print(f"Mean values range: {min(stats['mean_values']):.1f} - {max(stats['mean_values']):.1f}")
    print(f"Std values range: {min(stats['std_values']):.1f} - {max(stats['std_values']):.1f}")

def create_mosaic_preview():
    """Example of creating a mosaic from multiple tiles"""

    tiles_dir = "/Users/varunburde/projects/Recyllux/outputs/Polygon1_SO24012538-2-01_DS_PHR1B_202404231033373_FR1_PX_E005N43_0118_03298_panchromatic_tiles"

    # Create a simple 2x2 mosaic from corner tiles
    corner_tiles = [
        "tile_01_01.tif",  # Top-left
        "tile_01_12.tif",  # Top-right
        "tile_12_01.tif",  # Bottom-left
        "tile_12_12.tif"   # Bottom-right
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for i, tile_name in enumerate(corner_tiles):
        tile_path = os.path.join(tiles_dir, tile_name)

        with rasterio.open(tile_path) as src:
            tile_data = src.read(1)

        row, col = i // 2, i % 2
        axes[row, col].imshow(tile_data, cmap='gray')
        axes[row, col].set_title(f"Corner Tile {tile_name[:-4]}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("PLEIADES TILE PROCESSING EXAMPLES")
    print("="*40)

    print("\n1. Loading a single tile:")
    load_tile_example()

    print("\n2. Processing tiles in batch:")
    process_tiles_batch()

    print("\n3. Calculating statistics:")
    calculate_statistics()

    print("\n4. Creating mosaic preview:")
    create_mosaic_preview()

    print("\n" + "="*40)
    print("TIPS FOR WORKING WITH TILES:")
    print("- Each tile is 2048x2048 pixels (~2MB compressed)")
    print("- Tiles maintain georeferencing information")
    print("- Process tiles individually to manage memory")
    print("- Use parallel processing for batch operations")
    print("- Combine tiles for full-resolution analysis when needed")
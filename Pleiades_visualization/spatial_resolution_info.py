#!/usr/bin/env python3
"""
Pleiades Spatial Resolution Calculator
====================================

Calculate and display spatial resolution information for Pleiades tiles.
"""

import os
from pathlib import Path
import rasterio

def calculate_spatial_info():
    """Calculate spatial resolution and coverage information"""

    outputs_dir = Path("/Users/varunburde/projects/Recyllux/outputs")

    print("PLEIADES SPATIAL RESOLUTION CALCULATOR")
    print("="*50)

    # Find tile directories
    tile_dirs = []
    if outputs_dir.exists():
        for item in outputs_dir.iterdir():
            if item.is_dir() and 'tiles' in item.name:
                tile_count = len(list(item.glob('*.tif')))
                tile_dirs.append((item, tile_count))

    if not tile_dirs:
        print("No tile directories found!")
        return

    print(f"Found {len(tile_dirs)} tile directories\n")

    for tile_dir, count in tile_dirs:
        print(f"Dataset: {tile_dir.name}")
        print(f"Tiles: {count}")

        # Get first tile for analysis
        tiles = list(tile_dir.glob('*.tif'))
        if not tiles:
            continue

        with rasterio.open(str(tiles[0])) as src:
            pixel_size_x, pixel_size_y = src.res
            width_pixels = src.width
            height_pixels = src.height

            # Calculate ground coverage
            ground_width_m = width_pixels * abs(pixel_size_x)
            ground_height_m = height_pixels * abs(pixel_size_y)
            ground_area_km2 = (ground_width_m * ground_height_m) / 1_000_000

            # Determine image type
            is_panchromatic = 'panchromatic' in tile_dir.name

            print("  Pixel dimensions: {} × {} pixels".format(width_pixels, height_pixels))
            print("  Pixel size: {:.1f} × {:.1f} meters".format(abs(pixel_size_x), abs(pixel_size_y)))
            print("  Ground coverage: {:.0f} × {:.0f} meters".format(ground_width_m, ground_height_m))
            print("  Area: {:.3f} km² per tile".format(ground_area_km2))

            if is_panchromatic:
                print("  Type: Panchromatic (high-resolution)")
                print("  Scale: 1 pixel = 0.5m on ground")
                print("  Example: A 4m car spans ~8 pixels")
            else:
                print("  Type: Multispectral (4-band)")
                print("  Scale: 1 pixel = 2.0m on ground")
                print("  Example: A 4m car spans ~2 pixels")

        print("-" * 50)

    # Summary
    print("\nSPATIAL RESOLUTION SUMMARY")
    print("="*30)
    print("Panchromatic:  0.5m per pixel (50cm GSD)")
    print("Multispectral: 2.0m per pixel (200cm GSD)")
    print()
    print("Ground Distance Examples:")
    print("• 1 meter on ground = 2 pixels (panchromatic) / 0.5 pixels (multispectral)")
    print("• 10 meters on ground = 20 pixels (panchromatic) / 5 pixels (multispectral)")
    print("• 100 meters on ground = 200 pixels (panchromatic) / 50 pixels (multispectral)")

if __name__ == "__main__":
    calculate_spatial_info()
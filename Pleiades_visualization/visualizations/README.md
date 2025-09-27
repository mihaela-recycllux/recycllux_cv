# Pleiades Satellite Imagery Visualizations

This folder contains high-quality visualizations of your processed Pleiades satellite imagery tiles, organized by modality for easy browsing.

## 📁 Folder Structure

```
visualizations/
├── overview.html          # Web-based gallery of all visualizations
├── panchromatic/          # High-resolution panchromatic images
├── rgb/                   # Natural color RGB composites
├── false_color/           # False color (NIR-R-G) composites
├── nir/                   # Near-infrared visualizations (enhanced)
└── ndvi/                  # Vegetation index maps
```

## 🎨 Visualization Types

### Panchromatic
- **High-resolution grayscale images** from the panchromatic band
- Perfect for detailed feature identification
- Files: `tile_XX_YY_*.png` and `collage_*.png`

### RGB (Natural Color)
- **True color composites** using Red, Green, Blue bands
- Looks like natural photography
- Files: `tile_XX_YY_rgb_*.png` and `collage_*_rgb.png`

### False Color
- **Enhanced visualization** using Near-IR, Red, Green bands
- Vegetation appears bright red, water appears dark
- Great for vegetation and land use analysis
- Files: `tile_XX_YY_false_color_*.png` and `collage_*_false_color.png`

### NIR (Near-Infrared)
- **Enhanced NIR visualization** with improved contrast stretching
- Uses a blue-to-yellow-green colormap for better detail
- Shows vegetation health and water content
- Files: `tile_XX_YY_nir_*.png` and `collage_*_nir.png`

### NDVI (Vegetation Index)
- **Vegetation health indicator** from -1 to +1
- Red = low vegetation, Green/Yellow = healthy vegetation
- Quantitative measure of vegetation density
- Files: `tile_XX_YY_ndvi_*.png` and `collage_*_ndvi.png`

## 🖼️ File Types

- **Individual tiles**: High-resolution images of single 2048×2048 pixel tiles
- **Collages**: Overview images showing multiple tiles in a grid layout
- All images are saved as PNG format with 150 DPI for crisp quality

## 🚀 How to View

### Option 1: Web Gallery (Recommended)
1. Open `overview.html` in your web browser
2. Browse all visualizations in an organized gallery
3. Click any image to view full size

### Option 2: File Explorer
1. Open each modality folder (panchromatic, rgb, etc.)
2. Browse images with your system's image viewer
3. Use slideshow mode for easy comparison

### Option 3: Image Viewer Software
- Use any image viewer (Preview, IrfanView, etc.)
- All images are standard PNG format
- High resolution suitable for detailed analysis

## � Spatial Resolution & Scale

### Pixel-to-Ground Relationship

**Panchromatic Data:**
- **1 pixel = 0.5 meters** on the ground
- **1 meter on ground = 2 pixels**
- **1 kilometer on ground = 2,000 pixels**
- **Actual tile coverage**: 1.0-4.2 km² per tile (1024×1024m to 2048×2048m)

**Multispectral Data:**
- **1 pixel = 2.0 meters** on the ground
- **1 meter on ground = 0.5 pixels**
- **1 kilometer on ground = 500 pixels**
- **Actual tile coverage**: 12.7-30.9 km² per tile (3092×4096m to 3772×8192m)

### Real-World Examples

**Panchromatic (0.5m resolution):**
- A car (~4m long) spans ~8 pixels
- A house (~10m wide) spans ~20 pixels
- Street width (~8m) spans ~16 pixels
- Individual trees are clearly distinguishable
- **Your tiles**: 1024×1024m to 2048×2048m coverage

**Multispectral (2m resolution):**
- A car (~4m long) spans ~2 pixels
- A house (~10m wide) spans ~5 pixels
- Street width (~8m) spans ~4 pixels
- Good for land cover classification, vegetation analysis
- **Your tiles**: 3092×4096m to 3772×8192m coverage

### Practical Applications

- **Panchromatic**: Best for detailed feature identification, urban planning, infrastructure mapping
- **Multispectral**: Ideal for vegetation monitoring, land use classification, environmental analysis
- **Combined**: Panchromatic provides detail, multispectral provides spectral information

## 🔧 Technical Details

- **Data source**: Pleiades satellite imagery tiles
- **Spatial resolution**:
  - Panchromatic: 0.5m ground sample distance (GSD)
  - Multispectral: 2.0m ground sample distance (GSD)
- **Processing**: Contrast stretching, histogram equalization
- **Colormaps**: Optimized for each modality
- **Georeferencing**: Maintained from original tiles
- **Compression**: Lossless PNG format

## 📈 Usage Tips

1. **Compare modalities**: Open the same tile in different folders to see how different visualizations highlight different features
2. **Use collages**: Start with collage images to get an overview, then zoom into individual tiles
3. **NDVI analysis**: Use NDVI images for vegetation health assessment
4. **False color**: Best for identifying different land cover types
5. **Panchromatic**: Highest detail for feature identification

### Quick Scale Reference

| Object Size | Panchromatic Pixels | Multispectral Pixels |
|-------------|-------------------|---------------------|
| 1 meter     | 2 pixels          | 0.5 pixels         |
| Car (4m)    | 8 pixels          | 2 pixels           |
| House (10m) | 20 pixels         | 5 pixels           |
| Football field (100m) | 200 pixels | 50 pixels |
| 1 km² area  | ~4 million pixels | ~250,000 pixels    |
| Your tiles  | 1.0-4.2 km²       | 12.7-30.9 km²      |

## 📝 Generated Files Summary

- **Panchromatic**: 7 images (2 collages + 5 individual tiles)
- **RGB**: 7 images (2 collages + 5 individual tiles)
- **False Color**: 7 images (2 collages + 5 individual tiles)
- **NIR**: 7 images (2 collages + 5 individual tiles)
- **NDVI**: 7 images (2 collages + 5 individual tiles)

**Total**: 35 high-quality visualization images across all modalities

---

*Generated on: September 18, 2025*
*Satellite data: Pleiades imagery from two datasets*
*Processing: Enhanced with improved NIR visualization and contrast stretching*
*Spatial measurements: Based on actual processed tile data*
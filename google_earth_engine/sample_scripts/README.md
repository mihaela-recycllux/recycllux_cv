# Google Earth Engine Scripts

This directory contains Google Earth Engine implementations of satellite data analysis scripts, providing powerful cloud-based processing alternatives to the Sentinel Hub API scripts in the `examples_scripts` directory.

## Overview

These scripts demonstrate how to:
- Download and process satellite data using Google Earth Engine
- Perform advanced spectral analysis and index calculations
- Conduct time series analysis and change detection
- Export data in various formats for further analysis
- Leverage Earth Engine's global datasets and computational power

## Prerequisites

1. **Google Earth Engine Account**: Sign up at [earthengine.google.com](https://earthengine.google.com/)
2. **Authentication**: Run `ee.Authenticate()` on first use
3. **Python Dependencies**:
   ```bash
   pip install earthengine-api matplotlib numpy pandas pillow requests
   ```
4. **Project ID**: Update the project ID in scripts (currently set to 'gedospatial-data')

## Scripts Overview

### 01_sentinel2_basic_download_ee.py
**Basic Sentinel-2 data download and visualization**
- True color RGB composites
- False color (NIR-Red-Green) composites
- NDVI calculation and visualization
- Individual band downloads
- Covers fundamental Earth Engine concepts

**Key Features:**
- Cloud filtering (<20% cloud cover)
- Temporal compositing using median reducer
- Direct download URLs for small areas
- Visualization with matplotlib integration

### 02_sentinel1_sar_download_ee.py
**Sentinel-1 SAR data processing**
- VV and VH polarization analysis
- SAR RGB composites
- Ascending vs descending orbit comparison
- Coherence-like measures
- All-weather monitoring capabilities

**Applications:**
- Ship detection over water
- Flood mapping and monitoring
- Agricultural crop monitoring
- Surface roughness analysis

### 03_sentinel2_advanced_bands_ee.py
**Advanced spectral band analysis**
- Multiple vegetation indices (NDVI, EVI, SAVI, NDRE)
- Water detection indices (NDWI, MNDWI, AWEI)
- Built-up area indices (NDBI, UI, BU)
- Specialized band composites for different applications

**Spectral Indices:**
- **Vegetation**: NDVI, EVI, SAVI, NDRE
- **Water**: NDWI, MNDWI, AWEI  
- **Urban**: NDBI, UI, BU
- **Specialized composites**: Agriculture, geology, bathymetry

### 04_multi_satellite_comparison_ee.py
**Multi-satellite data integration**
- Sentinel-2 vs Landsat 8/9 comparison
- Sentinel-1 SAR integration
- MODIS large-scale monitoring
- Resolution and temporal comparison
- Seasonal analysis

**Satellite Characteristics:**
- **Sentinel-2**: 10-60m, 13 bands, 5-day revisit
- **Landsat 8/9**: 15-100m, 11 bands, 16-day revisit
- **Sentinel-1**: 5-20m, SAR all-weather, 6-day revisit
- **MODIS**: 250m-1km, daily global coverage

### 05_time_series_change_detection_ee.py
**Temporal analysis and change detection**
- Monthly NDVI time series
- RGB and NDVI change detection
- SAR-based flood detection
- Vegetation phenology monitoring
- Statistical change analysis

**Change Detection Methods:**
- Image differencing for visual changes
- NDVI differences for vegetation monitoring
- SAR analysis for flood detection
- Statistical threshold-based classification

### 06_data_export_formats_ee.py
**Data export and integration workflows**
- Direct downloads via URLs (small areas)
- Google Drive exports (large areas)
- Google Cloud Storage integration
- Statistical summaries and metadata
- Batch processing capabilities

**Export Options:**
- **Direct downloads**: <32MB, immediate access
- **Google Drive**: Larger datasets, background processing
- **Cloud Storage**: Enterprise workflows
- **Statistical CSV**: Quantitative summaries

## Quick Start Guide

### 1. Authentication Setup
```python
import ee

# First time setup
ee.Authenticate()

# Initialize for each session
ee.Initialize(project='your-project-id')
```

### 2. Basic Usage Pattern
```python
# Define area of interest
roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

# Build image collection
collection = (
    ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2023-01-01', '2023-12-31')
    .filterBounds(roi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
)

# Get composite image
image = collection.median()

# Generate download URL
url = image.getDownloadURL({
    'scale': 10,
    'crs': 'EPSG:4326',
    'region': roi,
    'fileFormat': 'GeoTIFF'
})
```

### 3. Running the Scripts
```bash
# Run any script directly
python 01_sentinel2_basic_download_ee.py

# Or import functions for custom analysis
from google_earth_engine.01_sentinel2_basic_download_ee import download_sentinel2_ndvi
```

## Key Advantages of Earth Engine

### 1. **Planetary-Scale Processing**
- Global datasets available instantly
- No need to download large files
- Massive computational resources

### 2. **Temporal Analysis**
- Easy multi-temporal compositing
- Built-in cloud masking
- Seasonal and annual trends

### 3. **Integration Capabilities**
- Combine multiple satellite missions
- Access to auxiliary datasets (DEM, weather, etc.)
- Advanced machine learning algorithms

### 4. **Export Flexibility**
- Multiple export destinations
- Batch processing for large areas
- Automatic tiling and projection handling

## Best Practices

### 1. **Region of Interest**
- Start with small areas for testing
- Use appropriate coordinate systems
- Consider computational limits

### 2. **Temporal Filtering**
- Apply cloud cover filters
- Use appropriate date ranges
- Consider seasonal variations

### 3. **Scale and Resolution**
- Match scale to analysis needs
- Consider processing time vs accuracy
- Use appropriate resampling methods

### 4. **Export Strategy**
- Direct downloads: <32MB areas
- Drive exports: Larger analyses
- Monitor export tasks in Code Editor

## Troubleshooting

### Common Issues:

1. **Authentication Errors**
   ```python
   # Re-authenticate
   ee.Authenticate()
   ee.Initialize(project='your-project-id')
   ```

2. **Download Size Limits**
   - Reduce area size or increase scale parameter
   - Use Drive exports for larger areas

3. **No Data Available**
   - Check date ranges and cloud cover filters
   - Verify area has satellite coverage

4. **Projection Issues**
   - Ensure consistent CRS across datasets
   - Use EPSG:4326 for global analyses

## Advanced Usage

### Custom Functions
```python
def add_indices(image):
    """Add multiple spectral indices to image"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndwi])

# Apply to collection
collection_with_indices = collection.map(add_indices)
```

### Batch Processing
```python
# Export multiple products
for year in range(2020, 2024):
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    annual_composite = (
        collection
        .filterDate(start_date, end_date)
        .median()
    )
    
    task = ee.batch.Export.image.toDrive(
        image=annual_composite,
        description=f'sentinel2_{year}',
        scale=10,
        region=roi
    )
    task.start()
```

## Resources

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine/)
- [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets/)
- [Earth Engine Code Editor](https://code.earthengine.google.com/)
- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)

## Support

For questions or issues:
1. Check the [Earth Engine FAQ](https://developers.google.com/earth-engine/faq)
2. Visit the [Earth Engine Forum](https://groups.google.com/forum/#!forum/google-earth-engine-developers)
3. Review script comments and documentation
4. Contact: varun@recycllux.com

---

**Note**: These scripts are designed for educational and research purposes. For production use, consider implementing proper error handling, logging, and resource management.

# Satellite Data Learning Scripts

This directory contains learning scripts for downloading and processing satellite data from Sentinel-1 and Sentinel-2 missions using the `sentinelhub-py` library.

## Scripts Overview

### 1. `01_sentinel2_basic_download.py`
**Basic Sentinel-2 Data Download**
- True color RGB images
- False color composites
- NDVI calculation
- Individual band downloads
- Comparison of L1C vs L2A processing levels

### 2. `02_sentinel1_sar_download.py`
**Sentinel-1 SAR Data Download**
- VV and VH polarizations
- SAR RGB composites
- Ascending vs Descending orbits comparison
- Basic coherence analysis
- SAR data visualization techniques

### 3. `03_sentinel2_advanced_bands.py`
**Advanced Sentinel-2 Band Analysis**
- Vegetation indices (NDVI, EVI, SAVI, NDRE)
- Water detection indices (NDWI, MNDWI, AWEI)
- Built-up area indices (NDBI, UI, BU)
- Advanced band composites for different applications

### 4. `04_multi_satellite_comparison.py`
**Multi-Satellite Data Comparison**
- Sentinel-2 vs Landsat comparison
- Optical vs SAR data comparison
- Resolution comparison (10m vs 20m bands)
- Temporal analysis across seasons

### 5. `05_time_series_change_detection.py`
**Time Series Analysis and Change Detection**
- Monthly NDVI time series
- RGB change detection
- NDVI change detection
- SAR-based flood detection
- Vegetation phenology monitoring

### 6. `06_data_export_formats.py`
**Data Download and Export**
- GeoTIFF export for GIS applications
- NetCDF export for climate analysis
- CSV statistical summaries
- Quicklook image generation
- Metadata creation

## Setup Instructions

### 1. Environment Setup

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv satellite_env
source satellite_env/bin/activate  # On Windows: satellite_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Sentinel Hub Account Setup

1. Create a free account at [Sentinel Hub](https://apps.sentinel-hub.com/)
2. Go to your Dashboard and create a new configuration
3. Note down your Client ID and Client Secret

### 3. Credentials Configuration

Create a `.env` file in your project root directory:

```env
SH_CLIENT_ID=your_client_id_here
SH_CLIENT_SECRET=your_client_secret_here
```


## Running the Scripts

### Basic Usage
```bash
python 01_sentinel2_basic_download.py
```

### Modify Area of Interest
Each script contains a `bbox` variable that you can modify:
```python
# Example: Lake Bled, Slovenia
bbox = BBox(bbox=[14.0, 46.0, 14.2, 46.15], crs=CRS.WGS84)

# Change to your area of interest
bbox = BBox(bbox=[lon_min, lat_min, lon_max, lat_max], crs=CRS.WGS84)
```

### Modify Time Range
Change the `time_interval` variable:
```python
time_interval = ('2023-07-01', '2023-07-31')
```

## Sentinel Hub Data Collections

### Sentinel-2 (Optical)
- **L1C**: Top of atmosphere reflectance
- **L2A**: Bottom of atmosphere reflectance (atmospherically corrected)
- **Resolution**: 10m, 20m, 60m (depending on band)
- **Revisit time**: 5 days

### Sentinel-1 (SAR)
- **IW**: Interferometric Wide Swath mode
- **EW**: Extra Wide Swath mode
- **Polarizations**: VV, VH, HH, HV
- **Resolution**: 5-20m
- **Revisit time**: 6 days

## Common Use Cases

### Agriculture
- Crop monitoring using NDVI time series
- Irrigation mapping with NDWI
- Crop type classification with multi-temporal data

### Urban Planning
- Urban expansion monitoring
- Built-up area mapping with NDBI
- Land use change detection

### Water Resources
- Water body detection with NDWI/MNDWI
- Flood mapping with SAR data
- Water quality monitoring

### Disaster Management
- Flood detection with SAR change detection
- Fire damage assessment with NDVI differences
- Emergency response mapping

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Check your Client ID and Client Secret
   - Ensure they're correctly set in `.env` file or config

2. **No Data Available**
   - Check if data exists for your area and time range
   - Try expanding the time interval
   - Some areas might have limited coverage

3. **Cloud Cover**
   - Optical data (Sentinel-2) can be affected by clouds
   - Try different time periods
   - Use SAR data (Sentinel-1) for all-weather monitoring

4. **Memory Issues**
   - Reduce image size: `size=(256, 256)` instead of `size=(512, 512)`
   - Process smaller areas
   - Use fewer bands or shorter time series

### Data Availability
- Sentinel-2: Available from 2015
- Sentinel-1: Available from 2014
- Some areas have better coverage than others
- Polar regions have more frequent coverage

## Next Steps

1. **Combine Scripts**: Mix techniques from different scripts for your specific application
2. **Automation**: Create batch processing scripts for multiple areas or time periods
3. **Validation**: Compare results with field data or other remote sensing products
4. **Integration**: Use exported data in GIS software (QGIS, ArcGIS) for further analysis
5. **Machine Learning**: Use downloaded data as input for classification or regression models

## Resources

- [Sentinel Hub Documentation](https://docs.sentinel-hub.com/)
- [sentinelhub-py Documentation](https://sentinelhub-py.readthedocs.io/)
- [ESA Copernicus Data Portal](https://browser.dataspace.copernicus.eu/)

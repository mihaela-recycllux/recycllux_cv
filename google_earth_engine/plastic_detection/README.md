# Plastic Detection System - Google Earth Engine

A comprehensive, object-oriented system for detecting floating plastic debris using multi-sensor satellite data from Google Earth Engine.

## Overview

This system implements advanced plastic detection algorithms using Sentinel-1 SAR and Sentinel-2 optical satellite data. It combines multiple spectral indices and detection methods to provide reliable identification of floating plastic debris in marine environments.

## Features

- **Multi-sensor Fusion**: Combines Sentinel-1 SAR and Sentinel-2 optical data
- **Multiple Detection Methods**: FDI, NDVI, Plastic Index, and Ensemble approaches
- **Object-Oriented Design**: Modular, extensible architecture
- **Comprehensive Visualization**: Multiple plots and analysis outputs
- **Configurable Parameters**: Easy customization of detection thresholds and parameters
- **Google Earth Engine Integration**: Leverages GEE's processing power and data catalog

## Directory Structure

```
plastic_detection_oop/
├── config/
│   └── config.py              # Configuration parameters
├── downloader/
│   └── downloader.py          # GEE data downloading
├── filters/
│   ├── base_filter.py         # Base filter class
│   ├── fdi_filter.py          # Floating Debris Index
│   ├── ndvi_filter.py         # NDVI-based detection
│   ├── plastic_index_filter.py # Custom plastic index
│   └── filter_manager.py      # Filter coordination
├── utils/
│   └── visualization.py       # Visualization utilities
├── data/                      # Output data directory
├── main.py                    # Main execution script
├── __init__.py               # Package initialization
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone or download** the project files

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Google Earth Engine**:
   - Create a GEE account at https://earthengine.google.com/
   - Install GEE Python API: `pip install earthengine-api`
   - Authenticate: `earthengine authenticate`
   - (Optional) Setup service account for production use

4. **Configure credentials** (if using service account):
   ```python
   # In config/config.py
   service_account = 'your-service-account@project.iam.gserviceaccount.com'
   private_key_path = '/path/to/private-key.json'
   ```

## Usage

### Basic Usage

Run the main detection pipeline:

```bash
python main.py
```

### Advanced Usage

```python
from plastic_detection_oop import GEEDownloader, FilterManager, VisualizationUtils, config

# Initialize components
downloader = GEEDownloader(config)
filter_manager = FilterManager(config.__dict__)
viz_utils = VisualizationUtils(config)

# Download data
optical_data, sar_data, data_mask = downloader.download_multi_sensor_data()

# Calculate indices
indices = filter_manager.calculate_indices(optical_data, sar_data, data_mask)

# Run detections
detections = filter_manager.detect_plastic_all_filters(indices)

# Create ensemble
ensemble_mask, ensemble_meta = filter_manager.create_ensemble_detection(detections)

# Visualize results
rgb = viz_utils.create_rgb_composite(optical_data)
viz_utils.create_detection_visualization(rgb, detections, config.aoi_bounds,
                                       (config.start_date, config.end_date))
```

## Configuration

Edit `config/config.py` to customize:

- **Area of Interest**: Modify `aoi_bounds` for different study areas
- **Time Period**: Change `start_date` and `end_date`
- **Detection Thresholds**: Adjust thresholds in `detection_thresholds`
- **Satellite Parameters**: Configure band selections and processing parameters
- **Output Directories**: Customize where results are saved

## Detection Methods

### 1. Floating Debris Index (FDI)
- Based on NIR and SWIR reflectance differences
- Effective for detecting floating materials
- Reference: Biermann et al. (2020)

### 2. NDVI-based Detection
- Uses Normalized Difference Vegetation Index
- Identifies non-vegetated water features
- Creates water masks for analysis restriction

### 3. Plastic Index
- Custom index: (Blue + Red) / (2 × Green)
- Exploits spectral properties of plastic materials

### 4. Ensemble Method
- Combines multiple detection methods
- Weighted averaging with confidence scoring
- Reduces false positives through consensus

## Output Files

The system generates:

- **Detection Masks**: Binary masks for each detection method
- **Index Arrays**: Calculated spectral indices
- **Visualizations**: RGB composites, detection overlays, analysis plots
- **Metadata**: JSON files with detection statistics and parameters
- **Summary Report**: Text report with comprehensive analysis

## Study Area

Default configuration targets the Romanian coast of the Black Sea, near Constanța port - a major shipping and river confluence area with high plastic pollution potential.

## Validation and Performance

- **Adaptive Thresholds**: Automatically adjusts based on data statistics
- **Multi-method Validation**: Cross-verification between detection approaches
- **Confidence Scoring**: Provides uncertainty estimates for detections
- **Water Masking**: Restricts analysis to relevant water areas

## Extension

The modular design allows easy addition of:

- **New Filters**: Implement `BaseFilter` for custom detection methods
- **Additional Indices**: Add new spectral indices in filter classes
- **Data Sources**: Extend `GEEDownloader` for other satellite data
- **Visualization**: Add new plot types in `VisualizationUtils`

## Troubleshooting

### Common Issues

1. **GEE Authentication**: Ensure proper authentication setup
2. **Data Availability**: Check if Sentinel data exists for your time/area
3. **Memory Issues**: Reduce image size in config for large areas
4. **No Detections**: Adjust thresholds or check water mask quality

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## References

- Biermann, L., et al. (2020). Finding plastic patches in coastal waters using optical satellite data. *Scientific Reports*.
- Topouzelis, K., et al. (2020). Detection of floating plastics from satellite imagery. *Marine Pollution Bulletin*.

## License

This project is part of the Recyllux initiative for marine plastic detection and monitoring.

## Contact

Varun Burde
varun@recycllux.com
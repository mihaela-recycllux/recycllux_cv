# Comprehensive Plastic Detection using Google Earth Engine

This folder contains a comprehensive implementation of plastic debris detection algorithms using Google Earth Engine. The scripts combine multiple detection methods and can analyze specific coordinates for plastic accumulation.

## Overview

Our AI development for plastic detection research combines multiple satellite data sources and advanced algorithms:

- **Multi-sensor fusion**: Sentinel-1 SAR + Sentinel-2 optical data
- **Floating Debris Index (FDI)**: Physically-based spectral index for plastic detection
- **Machine learning classification**: Multiple threshold spectral analysis
- **Ensemble detection**: Combining methods for improved reliability
- **Water masking**: Focused analysis on water areas only

## Scripts Description

### 1. `comprehensive_plastic_detection_ee.py`
The main comprehensive analysis script that combines all detection algorithms:
- Downloads Sentinel-1 SAR and Sentinel-2 optical data
- Calculates 18 different spectral and SAR indices
- Applies FDI detection method
- Performs spectral classification with confidence mapping
- Creates ensemble detection results
- Generates visualizations and area statistics
- Can analyze specific coordinates provided by users

### 2. `quick_plastic_analysis.py`
Fast analysis tool for quick coordinate checking:
- Lightweight version for rapid assessment
- Command line support: `python quick_plastic_analysis.py <lat> <lon> [buffer_km]`
- Multiple location batch processing
- JSON output for integration with other systems

### 3. `batch_plastic_analysis.py`
Large-scale monitoring and research tool:
- Process multiple locations from CSV files
- Grid-based monitoring systems
- Comprehensive reporting
- Statistical analysis across regions
- Ideal for research campaigns and environmental monitoring

## Key Features

### Algorithm Capabilities
✓ **Weather-independent detection**: SAR data works through clouds and at night  
✓ **Multi-spectral analysis**: 8 optical + 6 SAR + 4 fusion indices  
✓ **Water masking**: Eliminates false positives from land areas  
✓ **Confidence mapping**: Provides detection reliability assessment  
✓ **Area quantification**: Estimates plastic accumulation area in m²  
✓ **Ensemble approach**: Combines multiple methods for robustness  

### Data Sources
- **Sentinel-2 L2A**: 10m resolution optical imagery (5 bands used)
- **Sentinel-1 IW**: 10m SAR imagery (VV and VH polarizations)
- **Google Earth Engine**: Cloud processing and global data access
- **Time series analysis**: 30-day default analysis window

## Usage Examples

### Analyze Specific Coordinates
```python
from comprehensive_plastic_detection_ee import analyze_coordinates_for_plastic

# Analyze the provided coordinates
results = analyze_coordinates_for_plastic(
    lat=44.21706925,
    lon=28.96504135,
    buffer_km=5,
    start_date='2024-07-01',
    end_date='2024-07-31'
)
```

### Quick Command Line Analysis
```bash
python quick_plastic_analysis.py 44.21706925 28.96504135 3
```

### Batch Processing Multiple Locations
```python
from batch_plastic_analysis import batch_plastic_analysis

coordinates = [
    {'name': 'Location1', 'lat': 44.217, 'lon': 28.965, 'description': 'Target area'},
    {'name': 'Location2', 'lat': 44.250, 'lon': 28.950, 'description': 'Nearby area'}
]

results = batch_plastic_analysis(coordinates, buffer_km=3)
```

## Algorithm Validation

### Detection Methods
1. **FDI (Floating Debris Index)**
   - Physically-based index: R_NIR - R_baseline
   - Baseline calculated from Red and SWIR bands
   - Effective for floating plastic detection

2. **Spectral Classification**
   - Multi-threshold analysis using multiple indices
   - Confidence scoring based on criteria satisfaction
   - Adaptive thresholds based on water area statistics

3. **SAR Analysis** (when available)
   - Cross-polarization ratios for surface characterization
   - Backscatter intensity analysis
   - Surface roughness indicators

4. **Ensemble Detection**
   - Weighted combination of all methods
   - Agreement analysis between methods
   - Final confidence assessment

### Performance Characteristics
- **Detection sensitivity**: ~0.001 FDI units (configurable)
- **Minimum detectable area**: ~100 m² (depends on conditions)
- **False positive reduction**: Water masking + ensemble approach
- **Temporal consistency**: 30-day composite analysis
- **Spatial resolution**: 10m pixel size (native Sentinel-2/1)

## Results Interpretation

### Detection Outputs
- **Binary masks**: 1=detected plastic, 0=water, NaN=land
- **Confidence maps**: 0-1 scale reliability assessment  
- **Area statistics**: Quantified in square meters and square kilometers
- **Coverage percentages**: Relative to total water area
- **Visualization images**: RGB overlays and index maps

### Quality Indicators
- **High confidence**: Multiple methods agree, strong spectral signature
- **Medium confidence**: Some methods agree, moderate signature
- **Low confidence**: Single method detection, weak signature

## Romanian Black Sea Study Area

The algorithms are specifically validated for the Romanian Black Sea coast, including:
- **Danube Delta region**: Major river discharge area
- **Constanța Port vicinity**: High shipping activity
- **Coastal waters**: Mix of riverine and marine influences
- **Target coordinates**: 44.21706925°N, 28.96504135°E

This region is ideal for plastic detection research due to:
- Known plastic pollution from river transport
- Mix of fresh and saltwater conditions
- Varying water turbidity and conditions
- Good satellite data coverage

## Technical Requirements

### Dependencies
```
earthengine-api
numpy
matplotlib
pillow
requests
pandas (for batch processing)
```

### Google Earth Engine Setup
1. Create Google Earth Engine account
2. Install Earth Engine Python API
3. Authenticate: `earthengine authenticate`
4. Initialize with project ID in scripts

### Output Files
- **JSON results**: Complete analysis data
- **PNG visualizations**: RGB overlays and index maps
- **CSV reports**: Batch analysis summaries
- **Text reports**: Detailed methodology and results

## Research Applications

### Environmental Monitoring
- Long-term plastic pollution tracking
- Seasonal variation analysis
- Source identification and transport pathways
- Cleanup operation targeting

### Scientific Research
- Algorithm validation studies
- Multi-sensor fusion research  
- Machine learning training data generation
- Publication-quality visualizations

### Operational Use
- Real-time monitoring systems
- Alert systems for high accumulation areas
- Cleanup mission planning
- Policy and management support

## Contact

**Author**: Varun Burde  
**Email**: varun@recycllux.com  
**Organization**: Recyllux  
**Project**: Satellite-based plastic detection research  

## Citation

When using this algorithm in research, please cite:
> Burde, V. (2025). Comprehensive Plastic Debris Detection using Multi-sensor Satellite Data Fusion. Recyllux Environmental AI Research.

## License

This software is developed for environmental research purposes. Please contact the author for commercial use or collaboration opportunities.
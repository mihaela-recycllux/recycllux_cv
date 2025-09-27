# 🛰️ Plastic Detection System - Modular Architecture

A comprehensive, modular satellite-based plastic detection system using Google Earth Engine. This system has been refactored from a single 900+ line file into clean, maintainable modules.

## 📁 Project Structure

```
plastic_detection/
├── main.py                     # 🚀 Main entry point (lightweight)
├── README.md                   # 📖 This documentation
├── requirements.txt            # 📦 Python dependencies
│
├── cli/                        # 🖥️ Command Line Interface
│   ├── __init__.py
│   └── main_cli.py            # CLI argument parsing and workflow orchestration
│
├── workflow/                   # 🔄 Core Workflow Management
│   ├── __init__.py
│   └── plastic_workflow.py    # PlasticDetectionWorkflow class
│
├── analyzer/                   # 🔬 Image Analysis
│   ├── __init__.py
│   └── plastic_analyzer.py    # PlasticAnalyzer for statistical analysis
│
├── downloader/                 # 📡 Satellite Data Downloaders
│   ├── __init__.py
│   └── satellite_downloader.py # Sentinel-2, Sentinel-1, Landsat, MODIS
│
├── config/                     # ⚙️ Configuration
│   ├── __init__.py
│   └── settings.py            # Settings and region coordinates
│
├── utils/                      # 🛠️ Utilities
│   ├── __init__.py
│   ├── ee_utils.py           # Google Earth Engine utilities
│   └── file_downloader.py    # File download and ZIP handling
│
├── visualizer/                 # 📊 Visualization
│   ├── __init__.py
│   └── visualization.py      # Satellite visualization tools
│
└── outputs/                    # 📁 Generated Files
    ├── analysis/              # JSON analysis results
    └── visualizations/        # HTML summaries and plots
```

## 🚀 Quick Start

### Basic Usage
```bash
# Complete workflow (download → analyze → visualize)
python main.py --workflow complete --region mediterranean

# Individual steps
python main.py --workflow download --region great_pacific --products fdi fai ndwi
python main.py --workflow analyze
python main.py --workflow visualize
```

### Plastic Detection Focus
```bash
# Download plastic detection indices for Mediterranean
python main.py --workflow download \
    --products fdi fai ndwi \
    --region mediterranean \
    --satellites sentinel2

# Download SAR data for texture analysis
python main.py --workflow download \
    --products vv vh \
    --satellites sentinel1
```

## 📋 Available Options

### Workflows
- `complete` - Full 3-step pipeline
- `download` - Download satellite data only
- `analyze` - Analyze existing downloaded files
- `visualize` - Create visualizations from analyzed data

### Regions
- `great_pacific` - Great Pacific Garbage Patch
- `mediterranean` - Mediterranean Sea
- `caribbean` - Caribbean Sea
- `north_atlantic` - North Atlantic Ocean

### Satellites
- `sentinel2` - Optical data (10m resolution)
- `sentinel1` - SAR data (20m resolution) 
- `landsat` - Optical data (30m resolution)
- `modis` - Lower resolution (500m)

### Products

#### Optical Products (Sentinel-2, Landsat)
- `rgb` - True color composite
- `false_color` - False color (NIR-Red-Green)
- `ndvi` - Normalized Difference Vegetation Index
- `ndwi` - Normalized Difference Water Index
- `mndwi` - Modified Normalized Difference Water Index
- `fdi` - 🎯 **Floating Debris Index** (plastic detection)
- `fai` - 🎯 **Floating Algae Index** (distinguish from plastics)

#### SAR Products (Sentinel-1)
- `vv` - VV polarization
- `vh` - VH polarization

## 🏗️ Module Details

### 🔄 `workflow/plastic_workflow.py`
**Core orchestration class managing the 3-step process:**
- `PlasticDetectionWorkflow` - Main workflow controller
- `step1_download_all_images()` - Download satellite data as GeoTIFF
- `step2_analyze_images()` - Statistical analysis of downloaded images
- `step3_create_visualizations()` - Create analysis summaries

### 🔬 `analyzer/plastic_analyzer.py`
**Image analysis and interpretation:**
- `PlasticAnalyzer` - Statistical analysis engine
- Plastic detection analysis (FDI/FAI interpretation)
- Vegetation health assessment (NDVI)
- Water coverage estimation (NDWI/MNDWI)
- SAR surface roughness analysis

### 📡 `downloader/satellite_downloader.py`
**Satellite-specific data downloaders:**
- `Sentinel2Downloader` - Optical indices and composites
- `Sentinel1Downloader` - SAR polarizations
- Individual methods: `download_fdi()`, `download_fai()`, `download_ndwi()`, etc.
- Composite methods: `download_plastic_detection_indices()`

### 🖥️ `cli/main_cli.py`
**Command-line interface:**
- Argument parsing and validation
- Satellite-product compatibility checking
- Workflow execution with error handling
- Verbose logging options

### 🛠️ `utils/`
**Supporting utilities:**
- `ee_utils.py` - Google Earth Engine initialization and geometry
- `file_downloader.py` - HTTP downloads with ZIP extraction

## 🔧 Key Improvements from Monolithic Version

### ✅ Maintainability
- **900+ lines → 5 focused modules** (~150-200 lines each)
- Single Responsibility Principle for each module
- Clear separation of concerns

### ✅ Reusability  
- Import individual components: `from analyzer import PlasticAnalyzer`
- Modular testing capabilities
- Easy to extend with new satellites or products

### ✅ Readability
- Focused module documentation
- Type hints throughout
- Clear function names and purposes

### ✅ Error Handling
- Module-specific error handling
- Graceful degradation (e.g., analysis without rasterio)
- Better error messages and debugging

## 📊 Output Files

### Downloaded Data
```
outputs/
├── fdi_20230927_143026.tif     # Floating Debris Index
├── fai_20230927_143029.tif     # Floating Algae Index  
└── ndwi_20230927_143032.tif    # Water Index
```

### Analysis Results
```
outputs/analysis/
├── fdi_analysis.json           # Plastic detection analysis
├── fai_analysis.json           # Algae vs plastic analysis
└── ndwi_analysis.json          # Water coverage analysis
```

### Visualizations
```
outputs/visualizations/
└── analysis_summary.html       # Interactive summary report
```

## 🔬 Plastic Detection Workflow

1. **Download** FDI, FAI, and NDWI indices from Sentinel-2
2. **Analyze** statistical properties and plastic indicators  
3. **Interpret** results:
   - High FDI values → potential plastic accumulation
   - FAI comparison → distinguish algae from plastics
   - NDWI masking → focus analysis on water areas

## 🎯 QGIS Integration

All outputs are high-resolution GeoTIFF files (EPSG:4326) ready for QGIS:

1. Open QGIS Desktop
2. Drag & drop `.tif` files from `outputs/` directory
3. Use Symbology to enhance plastic detection visualization
4. Focus on FDI and FAI bands for plastic analysis

## 🚀 Future Extensions

The modular architecture makes it easy to add:
- New satellite data sources (Planet, Maxar)
- Additional analysis algorithms (ML-based detection)
- Real-time processing capabilities
- Advanced visualization tools
- API integration for automated monitoring

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `earthengine-api` - Google Earth Engine
- `rasterio` - GeoTIFF processing
- `numpy` - Numerical computations
- `requests` - HTTP downloads
- `tqdm` - Progress bars

---

*This modular architecture transforms a complex 900-line script into maintainable, extensible components for professional satellite-based plastic detection research.* 🛰️🌊
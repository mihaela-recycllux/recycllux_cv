# Plastic Detection using Google Earth Engine

**Complete workflow for marine plastic detection using satellite imagery**

This system provides a comprehensive 3-step pipeline:
1. **Download** → Download satellite images and save as GeoTIFF files
2. **Analysis** → Load images, perform analysis, and save results  
3. **Visualization** → Create collages and analysis reports

## 🏗️ Directory Structure

```
plastic_detection/
├── main.py                 # Main workflow script (THIS IS THE ONLY SCRIPT YOU NEED!)
├── README.md              # This file
├── config/                # Configuration settings
│   └── settings.py       
├── downloader/           # Satellite data downloaders
│   └── satellite_downloader.py
├── utils/                # Utility functions
│   ├── ee_utils.py
│   └── file_downloader.py
├── visualizer/           # Visualization tools
│   └── visualization.py
└── outputs/              # All downloaded files and results
    ├── analysis/         # Analysis results (JSON)
    └── visualizations/   # Collages and reports
```

## 🚀 Quick Start

### Prerequisites
```bash
# 1. Authenticate with Google Earth Engine
earthengine authenticate

# 2. Install dependencies (use main requirements.txt in project root)
pip install -r ../../requirements.txt
```

### Run Complete Workflow (Recommended)
```bash
# Download + Analyze + Visualize everything
python main.py --workflow complete --region great_pacific

# Custom workflow with specific satellites and products
python main.py --workflow complete --region mediterranean --satellites sentinel2 sentinel1 --products rgb ndvi fdi fai
```

### Run Individual Steps
```bash
# Step 1: Download only
python main.py --workflow download --region great_pacific

# Step 2: Analyze downloaded images
python main.py --workflow analyze

# Step 3: Create visualizations
python main.py --workflow visualize
```

## 🌊 Workflow Details

### Step 1: Download All Images
- Downloads satellite data from multiple missions
- Saves as individual GeoTIFF files with timestamp naming
- Format: `{product}_{timestamp}.tif` (e.g., `rgb_20230927_140455.tif`)
- Progress tracking with download status

### Step 2: Analysis
- Loads each downloaded image
- Performs product-specific analysis:
  - **FDI/FAI**: Plastic detection potential analysis
  - **NDVI/NDWI**: Environmental indicators  
  - **VV/VH**: SAR texture analysis
  - **RGB/False Color**: Optical imagery analysis
- Saves analysis results as JSON files

### Step 3: Visualization
- Creates comprehensive collages of all images
- Generates analysis reports with recommendations
- Produces summary visualizations
- Saves everything in `outputs/visualizations/`

## 🛰️ Available Data

### Regions
- `great_pacific` - Great Pacific Garbage Patch
- `mediterranean` - Mediterranean Sea hotspots  
- `north_atlantic` - North Atlantic accumulation zones
- `caribbean` - Caribbean Sea plastic accumulation areas

### Satellites & Products

**Sentinel-2 (Optical - 10m resolution)**
- `rgb` - True color composite
- `false_color` - False color (NIR-Red-Green)
- `ndvi` - Vegetation index
- `ndwi` - Water index
- `mndwi` - Modified water index
- `fdi` - **Floating Debris Index** (🎯 primary for plastic detection)
- `fai` - **Floating Algae Index** (helps distinguish plastic from algae)

**Sentinel-1 (SAR - all weather)**
- `vv` - VV polarization (surface roughness)
- `vh` - VH polarization (surface texture)

**Landsat 8/9 & MODIS**
- Similar products with different spatial/temporal resolution

## 📊 Output Files

### Downloaded Images
```
outputs/
├── rgb_20230927_140455.tif          (6.0 MB)
├── false_color_20230927_140455.tif  (6.4 MB)  
├── ndvi_20230927_140455.tif         (18 MB)
├── fdi_20230927_140455.tif          (19 MB)  ← Main plastic detection
├── fai_20230927_140455.tif          (19 MB)  ← Algae discrimination
├── vv_20230927_140455.tif           (6.7 MB)
└── vh_20230927_140455.tif           (6.6 MB)
```

### Analysis Results
```
outputs/analysis/
└── analysis_results_20230927_140455.json
```

### Visualizations
```
outputs/visualizations/
├── satellite_collage_20230927_140455.png
├── analysis_summary_20230927_140455.txt
└── simple_collage_summary_20230927_140455.txt
```

## 🎯 Usage Examples

### Basic Workflows
```bash
# Complete pipeline for Great Pacific region
python main.py --workflow complete --region great_pacific

# Focus on plastic detection products only
python main.py --workflow complete --products fdi fai ndwi --satellites sentinel2

# Quick download for analysis
python main.py --workflow download --region mediterranean --products rgb fdi
```

### Advanced Options
```bash
# Custom date range
python main.py --workflow complete --start-date 2023-06-01 --end-date 2023-08-31

# Specific output directory
python main.py --workflow complete --output-dir /path/to/custom/outputs

# Multiple regions (run separately)
python main.py --workflow download --region great_pacific
python main.py --workflow download --region mediterranean --output-dir ./outputs_med
```

## 🔬 Analysis Features

### Plastic Detection Analysis
- **FDI (Floating Debris Index)**: Primary plastic detection algorithm
- **FAI (Floating Algae Index)**: Distinguishes plastic from natural materials
- **Combined Analysis**: Cross-references multiple indices for accuracy

### Environmental Context
- **NDVI**: Vegetation health (coastal contamination impact)
- **NDWI/MNDWI**: Water quality and turbidity
- **SAR Data**: Surface texture analysis (independent of weather)

### Output Analysis
- File size analysis (data quality indicator)
- Product-specific recommendations
- Plastic detection potential scoring
- Analysis timestamp and metadata

## ⚙️ Configuration

### Main Settings (`config/settings.py`)
```python
EE_PROJECT_ID = "recycllux-satellite-data"
GREAT_PACIFIC_REGION = [...]  # Coordinates
VISUALIZATION_PARAMS = {...}  # Display parameters
```

### Custom Regions
Add new regions to `PlasticDetectionConfig` class:
```python
YOUR_REGION = [
    [lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]
]
```

## 🚨 Important Notes

1. **Earth Engine Authentication**: Must run `earthengine authenticate` first
2. **Dependencies**: Use main `requirements.txt` in project root, not local ones
3. **File Sizes**: Expect 100-200 MB total for complete download
4. **Processing Time**: Complete workflow takes 5-15 minutes depending on region
5. **Internet Required**: All downloads happen in real-time from Google servers

## 🎉 Success Indicators

**Download Step**:
- ✅ Files appear in `outputs/` with proper naming
- ✅ Multiple `.tif` files with different products
- ✅ File sizes in MB range (not KB)

**Analysis Step**:  
- ✅ JSON file in `outputs/analysis/`
- ✅ Analysis notes for each product
- ✅ Plastic detection potential scores

**Visualization Step**:
- ✅ Collage images in `outputs/visualizations/`
- ✅ Text summaries with recommendations
- ✅ Analysis reports linking data to plastic detection

---

**🌊 Ready to detect marine plastic pollution? Run the complete workflow and start analyzing!**
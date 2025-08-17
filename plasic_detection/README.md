# Plastic Debris Detection in Ocean Waters

This directory contains two Python scripts for detecting plastic debris in ocean waters using satellite data from the Sentinel constellation. Both scripts focus on the Romanian coast of the Black Sea and use the SentinelHub API to download data on-the-fly.

## Prerequisites

### 1. Sentinel Hub Account
- Sign up at: https://apps.sentinel-hub.com/
- Create a new configuration to get your `CLIENT_ID` and `CLIENT_SECRET`

### 2. Environment Variables
Set your Sentinel Hub credentials as environment variables:

```bash
export SH_CLIENT_ID="your_client_id_here"
export SH_CLIENT_SECRET="your_client_secret_here"
```

Or create a `.env` file in the project root:
```
SH_CLIENT_ID=your_client_id_here
SH_CLIENT_SECRET=your_client_secret_here
```

### Python Dependencies
Install required packages:
```bash
pip install sentinelhub matplotlib numpy python-dotenv scikit-learn
```

**Note**: Script 3 requires additional ML packages (scikit-learn) for clustering and anomaly detection.

## Scripts Overview

### Script 1: `01_sentinel2_plastic_fdi_detection.py`
**Naive Plastic Detection using Floating Debris Index (FDI)**

This script implements the Floating Debris Index method for detecting plastic debris using Sentinel-2 L2A imagery.

**Key Features:**
- Downloads Red (B04), NIR (B08), and SWIR (B11) bands
- Calculates FDI using linear baseline interpolation
- Creates binary detection mask with configurable threshold
- Generates comprehensive visualization with statistics

**Scientific Method:**
The FDI algorithm works by:
1. Calculating baseline reflectance: `R_baseline = R_red + (R_swir - R_red) × (λ_nir - λ_red)/(λ_swir - λ_red)`
2. Computing FDI: `FDI = R_nir - R_baseline`
3. Positive FDI values indicate potential floating debris

**Usage:**
```bash
python 01_sentinel2_plastic_fdi_detection.py
```

**Outputs:**
- RGB true-color image
- FDI heatmap
- Binary detection mask
- Statistical analysis
- Combined visualization saved as PNG in `data/` directory

### Script 2: `02_sentinel_data_fusion.py`
**Multi-sensor Data Fusion (Sentinel-1 SAR + Sentinel-2 Optical)**

This script demonstrates advanced data fusion by combining SAR and optical data to create a comprehensive dataset for machine learning applications.

**Key Features:**
- Downloads Sentinel-2 optical bands (Blue, Green, Red, NIR, SWIR)
- Downloads Sentinel-1 SAR data (VV and VH polarizations)
- Calculates 8 derived indices including FDI, NDVI, SAR ratios
- Creates ML-ready dataset with 15 features per pixel
- Performs spatial alignment and quality control

**Advantages of Data Fusion:**
- SAR works in all weather conditions (clouds, rain, darkness)
- Optical provides spectral discrimination
- Combined features reduce false positive rates
- Enhanced detection accuracy

**Usage:**
```bash
python 02_sentinel_data_fusion.py
```

**Outputs:**
- Fused dataset (NumPy compressed format) in `data/` directory
- Feature names and descriptions
- Comprehensive multi-panel visualization
- Band correlation analysis
- ML-ready dataset for training

### Script 3: `03_comprehensive_plastic_detection.py`
**Comprehensive Plastic Detection using Multiple Methods and Ensemble Learning**

This script combines the FDI detection method from Script 1 with the data fusion approach from Script 2, and adds machine learning techniques to create a comprehensive plastic detection system.

**Key Features:**
- Downloads both Sentinel-1 SAR and Sentinel-2 optical data
- Calculates 18 comprehensive indices (optical, SAR, and multi-sensor)
- Implements 4 detection methods:
  1. **FDI Thresholding** - Traditional physical method
  2. **ML Clustering** - K-means unsupervised learning
  3. **Anomaly Detection** - Isolation Forest algorithm
  4. **Ensemble Method** - Weighted combination of all methods
- Provides confidence scores and uncertainty estimation
- Creates comprehensive 20-panel visualization

**Detection Methods:**

1. **FDI Thresholding**: Physical baseline method using spectral interpolation
2. **ML Clustering**: K-means clustering to identify natural groupings in feature space
3. **Anomaly Detection**: Isolation Forest to detect unusual patterns indicative of debris
4. **Ensemble Detection**: Weighted voting system combining all methods

**Advanced Features:**
- Multi-method agreement analysis
- Confidence mapping and quality assessment
- Feature correlation analysis
- Detection method comparison
- Comprehensive statistical reporting

**Usage:**
```bash
python 03_comprehensive_plastic_detection.py
```

**Outputs:**
- Individual detection masks for each method
- Ensemble detection mask with confidence scores
- Comprehensive 20-panel visualization
- Detection quality metrics and method comparisons
- All results saved in compressed NumPy format in `data/` directory

**Ensemble Weighting:**
- FDI Method: 40% (well-established physical basis)
- ML Clustering: 40% (data-driven pattern recognition)  
- Anomaly Detection: 20% (complementary outlier detection)

## Data Storage

All generated data, images, and results are automatically saved to:
```
/Users/varunburde/projects/Recyllux/plasic_detection/data/
```

**File Structure:**
```
data/
├── plastic_detection_fdi_2024-07-10_2024-07-20.png          # Script 1 visualization
├── data_fusion_sar_optical_2024-07-10_2024-07-20.png        # Script 2 visualization  
├── comprehensive_plastic_detection_2024-07-10_2024-07-20.png # Script 3 visualization
├── fused_dataset_2024-07-10_2024-07-20.npz                  # Script 2 ML dataset
├── feature_names_2024-07-10_2024-07-20.txt                  # Script 2 feature list
├── plastic_detection_results_2024-07-10_2024-07-20.npz      # Script 3 detection results
└── calculated_indices_2024-07-10_2024-07-20.npz             # Script 3 indices
```

## Study Area

Both scripts focus on the **Romanian coast of the Black Sea** including the port of Constanța and Danube Delta area:
- **Bounding Box:** [28.5°E, 44.0°N, 29.2°E, 44.5°N]
- **Rationale:** Major shipping activity area with confluence of Danube River - high probability of marine debris
- **Coverage:** Includes Constanța port, Mamaia resort area, and southern Danube Delta
- **Time Period:** July 2024 (clear summer conditions)

## Output Features (Scripts 2 & 3)

### Script 2 - Fused Dataset (15 features):

**Original Bands (7):**
1. Blue (B02) - 490nm
2. Green (B03) - 560nm  
3. Red (B04) - 665nm
4. NIR (B08) - 842nm
5. SWIR (B11) - 1610nm
6. VV polarization (SAR)
7. VH polarization (SAR)

**Derived Indices (8):**
8. NDVI (vegetation index)
9. FDI (floating debris index)
10. NDWI (water index)
11. Plastic Index (empirical)
12. VH/VV Ratio (cross-polarization)
13. SAR Intensity
14. Depolarization Ratio
15. Enhanced Plastic Index (multi-sensor)

### Script 3 - Comprehensive Detection (18 features):

**Optical Indices (8):**
1. FDI (Floating Debris Index) - Primary plastic detection
2. NDVI (Normalized Difference Vegetation Index)
3. NDWI (Normalized Difference Water Index)
4. EVI (Enhanced Vegetation Index)
5. SR (Simple Ratio Index)
6. ARVI (Atmospherically Resistant Vegetation Index)
7. Plastic Index (empirical combination)
8. Modified FDI (normalized by red band)

**SAR Indices (6):**
9. VH/VV Ratio (cross-polarization)
10. SAR Intensity (total backscatter)
11. Depolarization Ratio
12. RVI (Radar Vegetation Index)
13. VV in dB scale
14. VH in dB scale

**Multi-sensor Fusion Indices (4):**
15. Optical-SAR Plastic Index
16. Enhanced Plastic Index
17. Water Quality Proxy
18. Surface Roughness Indicator

## Applications

### Script 1 - FDI Detection
- Visual inspection of potential plastic debris locations
- Threshold-based detection for operational monitoring
- Baseline method for comparison with advanced techniques

### Script 2 - Data Fusion
- Train supervised ML models for plastic detection
- Unsupervised clustering for water quality analysis
- Feature engineering for advanced algorithms
- Multi-temporal change detection

### Script 3 - Comprehensive Detection
- **Operational plastic detection** with confidence scores
- **Multi-method validation** for reduced false positives
- **Ensemble learning** for improved accuracy
- **Uncertainty quantification** for decision making
- **Method comparison** and performance evaluation

### Machine Learning Applications (All Scripts)
- Train supervised classifiers (Random Forest, SVM, Neural Networks)
- Unsupervised clustering for water quality analysis
- Change detection algorithms
- Anomaly detection for pollution events
- Real-time monitoring system development

## Limitations and Considerations

### FDI Method (Script 1)
- Works best in clear, calm water conditions
- May produce false positives from:
  - Whitecaps and foam
  - Suspended sediments
  - Cloud shadows
  - Bright boats or structures
- Requires validation with high-resolution imagery

### Data Fusion (Script 2)
- SAR data availability is typically more sparse than optical
- Temporal matching between SAR and optical may not be perfect
- Weather conditions can affect optical data quality
- Computational requirements increase with additional features

## Validation and Next Steps

1. **Ground Truth Collection:** 
   - Coordinate with marine research vessels
   - Use high-resolution imagery for validation
   - Collect water samples for verification

2. **Model Development:**
   - Train machine learning models on labeled data
   - Cross-validate with independent datasets
   - Optimize detection thresholds

3. **Operational Deployment:**
   - Scale up to larger geographic areas
   - Implement automated processing pipelines
   - Integrate with marine monitoring systems

## Troubleshooting

### Common Issues

1. **Authentication Errors:**
   - Verify SH_CLIENT_ID and SH_CLIENT_SECRET are set correctly
   - Check account status at Sentinel Hub dashboard

2. **No Data Available:**
   - Try different time periods
   - Check cloud coverage in the area
   - SAR data has lower temporal resolution than optical

3. **Processing Units:**
   - Ensure sufficient processing units in your Sentinel Hub account
   - Reduce image size or time range if needed

4. **Memory Issues:**
   - Reduce image resolution for large areas
   - Process data in smaller chunks
   - Close other applications to free memory

### Performance Tips

- Start with smaller image sizes (256x256) for testing
- Use cloud-optimized time periods for better data availability
- Consider processing multiple time periods for temporal analysis
- Save intermediate results to avoid re-downloading data

## Scientific References

1. Biermann, L., et al. (2020). "Finding Plastic Patches in Coastal North Sea Waters using Satellite Data." *Scientific Reports*, 10, 5364.

2. Topouzelis, K., et al. (2020). "Detection of floating plastics from satellite and unmanned aerial systems (Plastic Litter Project 2019)." *International Journal of Applied Earth Observation and Geoinformation*, 79, 175-183.

3. Garaba, S.P., & Dierssen, H.M. (2018). "An airborne remote sensing case study of synthetic hydrocarbon detection using short wave infrared absorption features identified from marine-harvested macro- and microplastics." *Remote Sensing of Environment*, 205, 224-235.

## Contact

For questions or issues:
- Author: Varun Burde
- Email: varun@recycllux.com
- Project: Recyllux Computer Vision

## License

This code is part of the Recyllux project. Please refer to the main project license for usage terms.

# **📡 Satellite Data Processing Pipeline**

This repository contains Python scripts that together form a pipeline for acquiring, processing, and analyzing satellite imagery data about Land Surface Temperature from Sentinel-2 and Sentinel-3 satellites. 

The pipeline includes:
- **🛰 Data Acquisition** → Downloads and processes NDVI (Sentinel-2) and LST (Sentinel-3).
- **⚙️ Data Processing** → Analyzes the correlation between NDVI and temperature.
- **📊 Visualization** → Generates maps, graphs, and predictive models.

---

## **📥 Installation**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
conda create -n venv python=3.10
conda activate venv
pip install -r requirements.txt
```

---

## **Main scripts**

### Data Acquisition
✔ Downloads **Sentinel-3 LST** and **Sentinel-2 reflectance** (for NDVI).  
✔ Clips and reprojects data to the area of interest.  
✔ Saves results as **GeoTIFFs** in `outputs/`.

### Data Processing
✔ Clips NDVI and LST rasters.  
✔ Creates a grid based on the temperature product resolution.  
✔ Computes **zonal statistics** (NDVI & LST mean per cell).  
✔ Saves results as **CSV**.

### Visualization
✔ Displays NDVI & LST rasters.  
✔ Generates **scatter plots** to analyze correlation.  
✔ Fits a **linear regression model** to refine LST resolution.  
✔ Saves outputs in `outputs/`.

### Outputs
📂 **GeoTIFFs** → NDVI (10m) & LST (1000m).  
📂 **CSV** → Zonal statistics.  
📂 **Graphs** → Correlation, raster visualizations & predictive model.  

---

### **📩 Contact**
For any questions or suggestions, feel free to open an issue in the repository or mail me to [julian.garped@usal.es](mailto:julian.garped@usal.es)

---

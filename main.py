from data_acquisition import get_s3, get_s2
from processing import create_grid_from_raster, calculate_zonal_stats
from visualization import plot_rasters, scatter_ndvi_temp
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
from config import *
import os

def run_data_acquisition():
    """Ejecuta la adquisición de datos para Sentinel-3 y Sentinel-2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Asegura que el directorio de salida exista

    # Sentinel-3
    out_path_s3 = os.path.join(OUTPUT_DIR, f"output_s3_{DATE_INI_S3}_{DATE_END_S3}.tif")
    get_s3(os.path.join(GEOJSON_DIR, "area.geojson"), out_path_s3, DATE_INI_S3, DATE_END_S3)

    # Sentinel-2
    out_path_s2 = os.path.join(OUTPUT_DIR, f"output_s2_{DATE_INI_S2}_{DATE_END_S2}.tif")
    get_s2(os.path.join(GEOJSON_DIR, "networks.geojson"), out_path_s2, DATE_INI_S2, DATE_END_S2, row=2)

def run_processing_and_visualization():
    """Carga, procesa y visualiza los datos adquiridos."""
    ndvi_src = os.path.join(OUTPUT_DIR, f"output_s2_{DATE_INI_S2}_{DATE_END_S2}.tif")
    temp_src = os.path.join(OUTPUT_DIR, f"output_s3_{DATE_INI_S3}_{DATE_END_S3}.tif")

    # Cargar NDVI
    with rio.open(ndvi_src) as src:
        ndvi_array = np.clip(src.read(1), -1, 1)
        ndvi_transform = src.transform
        ndvi_crs = src.crs
        print("ndvi_crs: ", ndvi_crs)
        ndvi_bounds = src.bounds
    area_1 = gpd.GeoDataFrame(index=[0], crs=ndvi_crs, geometry=[box(*ndvi_bounds)])

    # Cargar temperatura
    with rio.open(temp_src) as src:
        temp_array = src.read(1)
        temp_crs = src.crs
        print("temp_crs: ", temp_crs)
        area_2 = area_1.to_crs(temp_crs)
        area_2_crs = [area_2.loc[0, "geometry"]]
        print(area_2.shape)

        temp_cropped, temp_cropped_transform = mask(src, area_2_crs, crop=True)

    # Generar visualizaciones
    plot_rasters(ndvi_array, ndvi_transform, temp_cropped, temp_cropped_transform, area_1, area_2)
    
def main():
    run_data_acquisition()
    run_processing_and_visualization()

if __name__ == "__main__":
    main()

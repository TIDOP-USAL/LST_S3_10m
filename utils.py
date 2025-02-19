import logging
import os
import geopandas as gpd

def setup_logging(level=logging.INFO):
    """Configura el logging para el proyecto."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level
    )

def ensure_dir(directory):
    """Crea el directorio si no existe."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def read_geojson(file_name, crs=4326):
    """
    Lee un archivo GeoJSON y lo proyecta al CRS deseado.
    
    :param file_name: Ruta del archivo GeoJSON.
    :param crs: Código EPSG deseado.
    :return: GeoDataFrame.
    """
    gdf = gpd.read_file(file_name)
    return gdf.to_crs(epsg=crs)
    
def get_area_from_geojson(file_path, row=0, target_crs=4326):
    """
    Lee un GeoJSON y retorna la geometría y el centroide de una fila determinada.
    
    :param file_path: Ruta del archivo GeoJSON.
    :param row: Índice de la fila a utilizar.
    :param target_crs: CRS al que proyectar.
    :return: (GeoDataFrame, [x, y] del centroide)
    """
    area = read_geojson(file_path, crs=target_crs)
    centroid = area.geometry.iloc[row].centroid
    coords = [centroid.x, centroid.y]
    return area, coords

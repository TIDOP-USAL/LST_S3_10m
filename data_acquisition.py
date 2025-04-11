import fsspec
import stackstac
import numpy as np
import pandas as pd
import xarray as xr
import pystac_client
import planetary_computer
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from config import *
import rasterio as rio
from rasterio.mask import mask


# Inicializar los catálogos
CATALOG_S3 = pystac_client.Client.open(S3_CATALOG_URL, modifier=planetary_computer.sign_inplace)
CATALOG_S2 = pystac_client.Client.open(S2_CATALOG_URL)

def get_items(catalog, collection, coords, date_ini, date_end):
    search = catalog.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": coords},
        datetime=f"{date_ini}/{date_end}"
    )
    return search

def get_data_s3(item):
    geo = xr.open_dataset(fsspec.open(item.assets["slstr-geodetic-in"].href).open()).load()
    dataset = xr.open_dataset(fsspec.open(item.assets["lst-in"].href).open())
    latitude = geo["latitude_in"].values
    longitude = geo["longitude_in"].values
    lst_data = dataset["LST"].values
    df = pd.DataFrame({
        "longitude": longitude.flatten(),
        "latitude": latitude.flatten(),
        "lst": lst_data.flatten(),
    })
    return df

def df_to_gdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    gdf = gdf.set_crs(epsg=EPSG_INPUT)
    return gdf.to_crs(epsg=EPSG_PROJECTED)

def create_geocube(gdf, resolution):
    geo_grid = make_geocube(
        vector_data=gdf,
        measurements=["lst"],
        resolution=resolution,
        rasterize_function=rasterize_points_griddata,
    )
    return geo_grid

def get_area(area_path, row=0):
    area = gpd.read_file(area_path).to_crs(epsg=EPSG_INPUT)
    centroid = area.geometry.iloc[row].centroid
    coords = [centroid.x, centroid.y]
    return area, coords

def get_s3(area_path, out_path, date_ini, date_end):
    area, coords = get_area(area_path, 0)
    item_s3 = next(get_items(CATALOG_S3, S3_COLLECTION, coords, date_ini, date_end).items())
    df_s3 = get_data_s3(item_s3)
    gdf_s3 = df_to_gdf(df_s3)
    gdc_s3 = create_geocube(gdf_s3, 1000)  # 1000 metros de resolución
    clipped_s3 = gdc_s3.rio.clip(area.geometry.values, area.crs)
    clipped_s3.lst.rio.to_raster(out_path, driver="GTiff")

def get_s2(area_path, out_path, date_ini, date_end, row=0):
    area = gpd.read_file(area_path).to_crs(epsg=EPSG_INPUT)
    centroid = area.geometry.iloc[row].centroid
    coords = [centroid.x, centroid.y]
    item_s2 = next(get_items(CATALOG_S2, S2_COLLECTION, coords, date_ini, date_end).items())
    stack_s2 = stackstac.stack(item_s2, assets=["red", "nir"], epsg=EPSG_INPUT)
    clipped_s2 = stack_s2.rio.clip([area.geometry.iloc[row]], area.crs)
    ndvi = (clipped_s2[0][1] - clipped_s2[0][0]) / (clipped_s2[0][1] + clipped_s2[0][0])
    ndvi_reprojected = ndvi.rio.reproject(f"epsg:{EPSG_PROJECTED}")
    ndvi_reprojected.rio.to_raster(out_path, driver="GTiff")

def get_dem(area_path, dem_path, out_path, row=0):
    area = gpd.read_file(area_path).to_crs(epsg=EPSG_INPUT)
    centroid = area.geometry.iloc[row].centroid
    coords = [centroid.x, centroid.y]
    
    with rio.open(dem_path) as src:
        # Ensure the area geometry is in the same CRS as the DEM
        area = area.to_crs(src.crs)
        dem_array, dem_transform = mask(src, [area.geometry.iloc[row]], crop=True)
        dem_array = np.clip(dem_array, 0, None)
        dem_array = dem_array[0]  # Extract the first band
        dem_meta = src.meta.copy()
    
    dem_meta.update({
        'height': dem_array.shape[0],
        'width': dem_array.shape[1],
        'transform': dem_transform,
        'crs': src.crs.to_proj4(),
    })
    
    with rio.open(out_path, 'w', **dem_meta) as dst:
        dst.write(dem_array, 1)
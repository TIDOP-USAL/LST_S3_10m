import fsspec
import stackstac
import numpy as np
import pandas as pd
import xarray as xr
import pystac_client
import planetary_computer
import geopandas as gpd
import matplotlib.pyplot as plt
from geocube.api.core import make_geocube
from rasterio.transform import from_origin, from_bounds
from geocube.rasterize import rasterize_points_griddata

catalog_s3 = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

catalog_s2 = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

def get_items(catalog, collection, coords, date_ini, date_end):
    search = catalog.search(
        collections=[collection],
        intersects={"type": "Point", "coordinates": coords},
            datetime=f"{date_ini}/{date_end}" # 2021-03-01/2021-03-31
    )
    return search

def get_data_s3(item):
    geo = xr.open_dataset(fsspec.open(item.assets["slstr-geodetic-in"].href).open()).load()
    dataset = xr.open_dataset(fsspec.open(item.assets["lst-in"].href).open())
    latitude = geo["latitude_in"].values
    longitude = geo["longitude_in"].values
    lst_data = dataset["LST"].values
    df = pd.DataFrame(
        {
            "longitude": longitude.flatten(),
            "latitude": latitude.flatten(),
            "lst": lst_data.flatten(),
        }
    )
    return df

def df_to_gdf(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=25830)
    return gdf

def create_geocube(gdf, res):
    geo_grid = make_geocube(
        vector_data=gdf,
        measurements=["lst"],
        resolution=res,  # degrees
        rasterize_function=rasterize_points_griddata,
    )
    return geo_grid

def get_area(area_path, row=0):
    area = gpd.read_file(area_path)
    area = area.to_crs(epsg=4326)
    centroid = area["geometry"].iloc[row].centroid
    coords = [centroid.x, centroid.y]
    return area, coords

def get_s3(area_path, out_path, date_ini, date_end):
    area, coords = get_area(area_path, 0)

    search_s3 = get_items(catalog_s3, "sentinel-3-slstr-lst-l2-netcdf", coords, date_ini, date_end)
    item_s3 = next(search_s3.items())

    df_s3 = get_data_s3(item_s3)
    gdf_s3 = df_to_gdf(df_s3)
    gdc_s3 = create_geocube(gdf_s3, 1000)

    clipped_s3 = gdc_s3.rio.clip(area.geometry.values, area.crs)
    clipped_s3.lst.rio.to_raster(out_path, driver="GTiff")

def get_s2(area_path, out_path, date_ini, date_end, row=0):
    area = gpd.read_file(area_path)
    area = area.to_crs(epsg=4326)
    centroid = area["geometry"].iloc[row].centroid
    coords = [centroid.x, centroid.y]

    search_s2 = get_items(catalog_s2, "sentinel-2-l2a", coords, date_ini, date_end)
    item_s2 = next(search_s2.items())
    stack_s2 = stackstac.stack(item_s2, assets=["red", "nir"])
    clipped_s2 = stack_s2.rio.clip([area.geometry.iloc[row]], area.crs)

    ndvi = (clipped_s2[0][1] - clipped_s2[0][0]) / (clipped_s2[0][1] + clipped_s2[0][0])
    ndvi.rio.to_raster(out_path, driver="GTiff")

def process():
    ################ Sentinel 3 ################
    area_path = "geojson/area.geojson"
    date_ini = "2024-05-04"
    date_end = "2024-05-06"
    out_path = f"out_small/output_s3_{date_ini}_{date_end}.tif"
    get_s3(area_path, out_path, date_ini, date_end)

    ################ Sentinel 2 ################
    area_path = "geojson/networks.geojson"
    date_ini = "2024-05-10"
    date_end = "2024-05-12"
    row = 0
    out_path = f"out_small/output_s2_{date_ini}_{date_end}_{row + 1}.tif"
    get_s2(area_path, out_path, date_ini, date_end, row)

def main():
    process()

if __name__ == "__main__":
    main()
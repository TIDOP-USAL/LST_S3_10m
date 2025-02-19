import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import box
import rasterstats

def crop_raster(raster_path, geojson_geometry):
    with rio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geojson_geometry], crop=True)
    return out_image, out_transform

def create_grid_from_raster(raster_path):
    with rio.open(raster_path) as src:
        xmin, ymin, xmax, ymax = src.bounds
        transform = src.transform
    cell_size_x, cell_size_y = transform[0], transform[4]
    grid_cells = []
    for x in np.arange(xmin, xmax, cell_size_x):
        for y in np.arange(ymin, ymax, -cell_size_y):
            grid_cells.append(box(x, y + cell_size_y, x + cell_size_x, y))
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=src.crs)
    return grid

def calculate_zonal_stats(grid, raster_array, transform, stat='mean'):
    stats = rasterstats.zonal_stats(grid.geometry, raster_array, affine=transform, stats=stat)
    return [x[stat] for x in stats]

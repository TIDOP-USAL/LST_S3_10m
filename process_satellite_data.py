import rasterio as rio
import rasterstats
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from shapely.geometry import box
from rasterio.mask import mask
from shapely.geometry import Polygon
from rasterio.warp import calculate_default_transform, reproject, Resampling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from rasterio.plot import show

# Cargar el GeoTIFF de NDVI
with rio.open("out_small/output_s2_2024-06-03_2024-06-05_4.tif") as ndvi_src:
    ndvi_transform = ndvi_src.transform
    ndvi_array = ndvi_src.read(1)
    ndvi_crs = ndvi_src.crs

    # Extraer el bounding box del NDVI raster y convertirlo en un GeoDataFrame
    ndvi_bounds = box(*ndvi_src.bounds)
    area = gpd.GeoDataFrame(index=[0], crs=ndvi_crs, geometry=[ndvi_bounds])

# Cargar el GeoTIFF de S3
with rio.open("out_small/output_s3_2024-06-03_2024-06-05.tif") as temp_src:
    temp_transform = temp_src.transform
    temp_array = temp_src.read(1)
    temp_crs = temp_src.crs
    
    # Transform area_1 to the same CRS as the temperature raster
    area_1_transformed = area.to_crs(temp_crs)
    area_1_crs2 = [area_1_transformed.loc[0, "geometry"]]

    # Recortar la imagen de temperatura al área de interés
    temp_cropped, temp_cropped_transform = mask(temp_src, area_1_crs2, crop=True)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("NDVI Raster")
show(ndvi_array, transform=ndvi_transform, ax=ax1, cmap='RdYlGn')
ax2.set_title("S3 Raster Recortado")
show(temp_cropped, transform=temp_cropped_transform, ax=ax2, cmap='coolwarm')
area.boundary.plot(ax=ax1, edgecolor='black')
area.boundary.plot(ax=ax2, edgecolor='black')
area_1_transformed.boundary.plot(ax=ax2, edgecolor='black')

plt.tight_layout()
plt.show()


# Crear grillado (grid) basado en el raster S3
xmin, ymin, xmax, ymax = temp_src.bounds
grid_cells = []
cell_size_x, cell_size_y = temp_transform[0], temp_transform[4]
for x in np.arange(xmin, xmax, cell_size_x):
    for y in np.arange(ymin, ymax, -cell_size_y):
        grid_cells.append(box(x, y + cell_size_y, x + cell_size_x, y))

# Convertir el grillado en un GeoDataFrame
grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=temp_crs)


area_1 = area.loc[0, "geometry"]

# Recortar el grillado al área de interés
grid_clipped = gpd.clip(grid, area_1)

# Realizar el clip de las matrices NDVI y S3 usando el área de interés
mask_ndvi = geometry_mask([area_1], out_shape=ndvi_array.shape, transform=ndvi_transform, invert=True)
ndvi_array_clipped = ndvi_array.copy()
ndvi_array_clipped[~mask_ndvi] = np.nan

mask_temp = geometry_mask([area_1], out_shape=temp_array.shape, transform=temp_transform, invert=True)
temp_array_clipped = temp_array.copy()
temp_array_clipped[~mask_temp] = np.nan

# Extraer la media de NDVI y temp utilizando el grillado recortado
ndvi_stats = rasterstats.zonal_stats(grid_clipped.geometry, ndvi_array_clipped, affine=ndvi_transform, stats='mean')
temp_stats = rasterstats.zonal_stats(grid_clipped.geometry, temp_array_clipped, affine=temp_transform, stats='mean')

# Agregar los resultados al GeoDataFrame del grillado
grid_clipped['NDVI_mean'] = [x['mean'] for x in ndvi_stats]
grid_clipped['temp_mean'] = [x['mean'] for x in temp_stats]

# Export the clipped grid to a csv
grid_clipped.to_csv("csv/area4.csv", index=False)

# Filter when the NDVI or S3 values are NaN
grid_clipped = grid_clipped.dropna(subset=['NDVI_mean', 'temp_mean'])



# Plot the two arrays in two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot NDVI
grid_clipped.plot(column='NDVI_mean', ax=ax1, legend=True)
ax1.set_title("NDVI")
ax1.set_axis_off()

# Plot S3
grid_clipped.plot(column='temp_mean', ax=ax2, legend=True)
ax2.set_title("Temp")
ax2.set_axis_off()

plt.show()


# Create a scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(data=grid_clipped, x='NDVI_mean', y='temp_mean')
plt.title('Inverse correlation betweeen NDVI vs Temperature')
plt.xlabel('NDVI')
plt.ylabel('Land Surface Temperature (LST)')

# Show the plot
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

ndvi_min, ndvi_max = 0.22, 0.9
temp_min, temp_max = 295, 315

# Filter the DataFrame based on the specified conditions
filtered_df = grid_clipped[(grid_clipped['NDVI_mean'] >= ndvi_min) & (grid_clipped['NDVI_mean'] <= ndvi_max) 
                           & (grid_clipped['temp_mean'] >= temp_min) & (grid_clipped['temp_mean'] <= temp_max)]

# Create a scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(data=filtered_df, x='NDVI_mean', y='temp_mean', palette='coolwarm')

# Fit a linear regression model
slope, intercept = np.polyfit(filtered_df['NDVI_mean'], filtered_df['temp_mean'], 1)

# Plot the fitted line
x_line = np.linspace(min(filtered_df['NDVI_mean']), max(filtered_df['NDVI_mean']), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r', label='Fitted line')

# Create the equation string
equation_str = f'y = {slope:.2f} * x + {intercept:.2f}'

# Display the equation on the plot
plt.text(max(filtered_df['NDVI_mean']) - 0.3, max(filtered_df['temp_mean']), equation_str, fontsize=12, color='black')

# Create the correlation coefficient string
corr_coeff = stats.pearsonr(filtered_df['NDVI_mean'], filtered_df['temp_mean'])[0]
corr_coeff_str = f'Correlation coefficient: {corr_coeff:.2f}'

# Display the correlation coefficient on the plot
plt.text(max(filtered_df['NDVI_mean']) - 0.3, max(filtered_df['temp_mean']) - 1, corr_coeff_str, fontsize=12, color='black')

# Set plot title and axis labels
plt.title('Inverse correlation betweeen NDVI vs Temperature')
plt.xlabel('NDVI')
plt.ylabel('Land Surface Temperature (LST)')

# Show the plot
plt.show()



import rasterio
import numpy as np
import matplotlib.pyplot as plt

ndvi = ndvi_array_clipped
temp = slope * ndvi + intercept

ndvi_cmap = plt.cm.RdYlGn

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot NDVI
im1 = ax1.imshow(ndvi, cmap=ndvi_cmap.reversed(), vmin=ndvi_min, vmax=ndvi_max)
ax1.set_title("NDVI", fontweight='bold', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=8)
fig.colorbar(im1, ax=ax1, shrink=0.5)

# Plot temperature
im2 = ax2.imshow(temp, cmap=ndvi_cmap.reversed(), vmin=temp_min, vmax=temp_max)
ax2.set_title("LST (K)", fontweight='bold', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=8)
fig.colorbar(im2, ax=ax2, shrink=0.5)

plt.show()



# Plot NDVI and temperature side by side
fig, axs = plt.subplots(ncols=3, figsize=(22, 10))

# Plot NDVI
ndvi_subset = ndvi[int(0.75 * ndvi.shape[0]):, int(0.75 * ndvi.shape[1]):]
im1 = axs[0].imshow(ndvi_subset, cmap=ndvi_cmap.reversed(), vmin=ndvi_min, vmax=ndvi_max)
axs[0].set_title('Sentinel 2 - NDVI', fontweight='bold', fontsize=14)
axs[0].set_xticks([])
axs[0].set_yticks([])
fig.colorbar(im1, ax=axs[0], shrink=0.3)

# Plot temperature
temp_subset = temp_cropped[0][int(0.75 * temp_cropped[0].shape[0]):, int(0.75 * temp_cropped[0].shape[1]):]
im3 = axs[1].imshow(temp_subset, cmap=ndvi_cmap.reversed(), vmin=temp_min, vmax=temp_max)
axs[1].set_title('Sentinel 3 - LST (1000m)', fontweight='bold', fontsize=14)
axs[1].set_xticks([])
axs[1].set_yticks([])
fig.colorbar(im3, ax=axs[1], shrink=0.3)

# Plot temperature
temp_subset = temp[int(0.75 * temp.shape[0]):, int(0.75 * temp.shape[1]):]
im2 = axs[2].imshow(temp_subset, cmap=ndvi_cmap.reversed(), vmin=temp_min, vmax=temp_max)
axs[2].set_title('Sentinel 3 - LST (10m)', fontweight='bold', fontsize=14)
axs[2].set_xticks([])
axs[2].set_yticks([])
fig.colorbar(im2, ax=axs[2], shrink=0.3)

# Adjust the spacing between the subplots
fig.subplots_adjust(wspace=0.2)

plt.show()
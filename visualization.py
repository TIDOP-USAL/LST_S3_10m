import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.plot import show

def plot_rasters(ndvi_array, ndvi_transform, temp_array, temp_transform, area_1, area_2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("NDVI Raster")
    show(ndvi_array, transform=ndvi_transform, ax=ax1, cmap='RdYlGn')
    ax2.set_title("S3 Raster Recortado")
    show(temp_array, transform=temp_transform, ax=ax2, cmap='coolwarm')
    area_1.boundary.plot(ax=ax1, edgecolor='black')
    area_2.boundary.plot(ax=ax2, edgecolor='black')
    plt.tight_layout()
    plt.show()

def scatter_ndvi_temp(dataframe, x_col='NDVI_mean', y_col='temp_mean'):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=dataframe, x=x_col, y=y_col)
    plt.title('Correlación entre NDVI y Temperatura')
    plt.xlabel('NDVI')
    plt.ylabel('Temperatura Superficial (LST)')
    plt.show()
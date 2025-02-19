import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Función para cargar valores de un GeoTIFF
def load_raster_values(raster_path, points):
    with rasterio.open(raster_path) as src:
        values = []
        for _, point in points.iterrows():
            row, col = src.index(point.geometry.x, point.geometry.y)  # Obtén fila y columna de la coordenada
            value = src.read(1)[row, col]  # Lee el valor del raster
            values.append(value)
    return np.array(values)

# Cargar los puntos de validación
validation_points = gpd.read_file('C:/Users/Tidop/Desktop/Interpol_Art_02/ARCGIS/shapes/puntos/puntos_validacion.shp')

# Cargar valores de temperatura de lst_sat1 y convertir de Kelvin a Celsius
lst_sat1_kelvin = load_raster_values('C:/Users/Tidop/Desktop/Interpol_Art_02/LST/validacion_LST/Resultados/drive-download-20241021T071049Z-001/ls3_s3_crop.tif', validation_points)
lst_sat1_celsius = lst_sat1_kelvin - 273.15  # Convertir a Celsius

# Obtener los valores reales de temperatura de los puntos de validación
true_temperature_values = validation_points['RASTERVALU'].values  # Cambia 'RASTERVALU' por el nombre correcto

# Asegúrate de que ambos arrays tengan el mismo tamaño
assert lst_sat1_celsius.shape == true_temperature_values.shape, "Los arrays deben tener el mismo tamaño."

# Calcular errores punto por punto
def calculate_pointwise_errors(true_values, predicted_values):
    # Cálculo correcto punto por punto
    mbe = predicted_values - true_values  # Sesgo medio por punto
    ase = np.abs(predicted_values - true_values)  # Error absoluto por punto
    mse = (predicted_values - true_values) ** 2  # Error cuadrático medio por punto
    rmse = np.sqrt(mse)  # RMSE por punto
    return rmse, mbe, ase, mse

# Calcular errores para cada punto
rmse_values, mbe_values, ase_values, mse_values = calculate_pointwise_errors(true_temperature_values, lst_sat1_celsius)

# Crear un DataFrame con los resultados por cada punto de validación
results_df = pd.DataFrame({
    'PointID': validation_points['CID'],  # Cambia 'CID' por la columna que identifica los puntos
    'True Temperature (°C)': true_temperature_values,
    'Predicted Temperature (°C)': lst_sat1_celsius,
    'RMSE': rmse_values,
    'MBE': mbe_values,
    'ASE': ase_values,
    'MSE': mse_values
})

# Calcular el MAE (Error Absoluto Medio) y otros errores globales
mae_value = np.mean(ase_values)  # Error Absoluto Medio (global)
global_rmse = np.sqrt(mean_squared_error(true_temperature_values, lst_sat1_celsius))  # RMSE Global
global_mbe = np.mean(mbe_values)  # MBE Global
global_mse = np.mean(mse_values)  # MSE Global
global_r2 = r2_score(true_temperature_values, lst_sat1_celsius)  # R² Global

# Agregar los errores globales al DataFrame
global_errors = {
    'Global RMSE': global_rmse,
    'Global MBE': global_mbe,
    'Global MAE': mae_value,  # MAE Global
    'Global MSE': global_mse,
    'R²': global_r2
}

# Convertir los errores globales en DataFrame para exportar
global_errors_df = pd.DataFrame([global_errors])

# Configurar pandas para no usar notación científica
pd.set_option('display.float_format', '{:.2f}'.format)

# Exportar resultados a un archivo CSV
results_df.to_csv('C:/Users/Tidop/Desktop/Interpol_Art_02/LST/validacion_LST/validacion_punto_a_punto.csv', index=False)
global_errors_df.to_csv('C:/Users/Tidop/Desktop/Interpol_Art_02/LST/validacion_LST/validacion_global.csv', index=False)

# Exportar el archivo a formato Excel
results_excel_path = 'C:/Users/Tidop/Desktop/Interpol_Art_02/LST/validacion_LST/validacion_punto_a_punto.xlsx'
results_df.to_excel(results_excel_path, index=False)

# Filtrar las filas donde la diferencia de temperatura (ASE) sea menor o igual a 3 grados
filtered_df = results_df[results_df['ASE'] <= 8]

# Exportar el DataFrame filtrado a un nuevo archivo Excel
filtered_excel_path = 'C:/Users/Tidop/Desktop/Interpol_Art_02/LST/validacion_LST/validacion_filtrada_menor_3_grados.xlsx'
filtered_df.to_excel(filtered_excel_path, index=False)

# Mostrar resultados en consola
print("Errores entre valores reales y lst_sat1.tif (convertido a Celsius):")
print(f"Global RMSE: {global_rmse:.2f}")
print(f"Global MBE: {global_mbe:.2f}")
print(f"Global MAE: {mae_value:.2f}")  # Mostramos MAE en la consola
print(f"R²: {global_r2:.2f}")
print(f"Global MSE: {global_mse:.2f}")

print(f"Archivo Excel exportado en: {results_excel_path}")
print(f"Archivo Excel filtrado exportado en: {filtered_excel_path}")
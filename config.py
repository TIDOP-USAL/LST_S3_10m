# URLs de los catálogos
S3_CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_CATALOG_URL = "https://earth-search.aws.element84.com/v1"

# Colecciones
S3_COLLECTION = "sentinel-3-slstr-lst-l2-netcdf"
S2_COLLECTION = "sentinel-2-l2a"

# Parámetros de proyección
EPSG_INPUT = 4326
EPSG_PROJECTED = 25830

# Directorios (puedes ajustarlos según tus necesidades)
GEOJSON_DIR = "data"
OUTPUT_DIR = "outputs"
CSV_DIR = "csv"

# Fechas de ejemplo (estas pueden ser modificadas o parametrizadas)
DATE_INI_S3 = "2024-05-04"
DATE_END_S3 = "2024-05-06"

DATE_INI_S2 = "2024-05-10"
DATE_END_S2 = "2024-05-12"

# Resolución deseada (en metros, por ejemplo)
S3_RESOLUTION = 1000

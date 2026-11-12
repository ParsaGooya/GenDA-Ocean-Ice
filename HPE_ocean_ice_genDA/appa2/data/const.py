r"""GLORYS constants."""

# GLORYS (GenCast-based)
GLORYS_DATE_START = "2010-01-01"
GLORYS_DATE_END = "2026-12-31"
GLORYS_RESOLUTION = (4320, 787) 
GLORYS_TIME_INTERVAL = 1
GLORYS_NUM_TOTAL_LEVELS = 22
# GLORYS_NUM_COARSENED_LEVELS = 13

# WeatherBench2
# Buckets are available at https://console.cloud.google.com/storage/browser/weatherbench2/datasets/era5
# GLORYS_AVAILABLE_CONFIGURATIONS = {
#     (
#         6,
#         ERA5_NUM_COARSENED_LEVELS,
#         64,
#         32,
#         False,
#     ): "2023_01_10-6h-64x32_equiangular_conservative.zarr",
#     (
#         6,
#         ERA5_NUM_COARSENED_LEVELS,
#         64,
#         32,
#         True,
#     ): "2022-6h-64x32_equiangular_with_poles_conservative.zarr",
#     (6, ERA5_NUM_COARSENED_LEVELS, 64, 33, False): "2022-6h-64x33.zarr",
#     (6, ERA5_NUM_COARSENED_LEVELS, 128, 64, False): "2022-6h-128x64_equiangular_conservative.zarr",
#     (
#         6,
#         ERA5_NUM_COARSENED_LEVELS,
#         128,
#         64,
#         True,
#     ): "2022-6h-128x64_equiangular_with_poles_conservative.zarr",
#     (
#         6,
#         ERA5_NUM_COARSENED_LEVELS,
#         240,
#         121,
#         True,
#     ): "2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
#     (
#         6,
#         ERA5_NUM_COARSENED_LEVELS,
#         512,
#         256,
#         False,
#     ): "2022-6h-512x256_equiangular_conservative.zarr",
#     (6, ERA5_NUM_COARSENED_LEVELS, 1440, 721, True): "2023_01_10-wb13-6h-1440x721.zarr",
#     (6, ERA5_NUM_TOTAL_LEVELS, 1440, 721, True): "2022-full_37-6h-0p25deg-chunk-1.zarr-v2/",
#     (
#         1,
#         ERA5_NUM_COARSENED_LEVELS,
#         240,
#         121,
#         True,
#     ): "2023_01_10-1h-240x121_equiangular_with_poles_conservative.zarr",
#     (
#         1,
#         ERA5_NUM_COARSENED_LEVELS,
#         360,
#         181,
#         True,
#     ): "2022-1h-360x181_equiangular_with_poles_conservative.zarr",
#     (
#         1,
#         ERA5_NUM_TOTAL_LEVELS,
#         512,
#         256,
#         False,
#     ): "2023_01_10-full_37-1h-512x256_equiangular_conservative.zarr",
#     (1, ERA5_NUM_TOTAL_LEVELS, 1440, 721, True): "2022-full_37-1h-0p25deg-chunk-1.zarr-v2",
# }

# fmt: off
GLORYS_DEPTH_LEVELS = [
        0.494025,  1.541375,  2.645669,  3.819495,  5.078224,  6.440614,
        7.92956 ,  9.572997, 11.405   , 13.46714 , 15.81007 , 18.49556 ,
       21.59882 , 25.21141 , 29.44473 , 34.43415 , 40.34405 , 47.37369 ,
       55.76429 , 65.80727 , 77.85385 , 92.32607 
]

SUB_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

BASE_DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# fmt: on
GLORYS_SURFACE_VARIABLES = [
    "zos",
    "siconc",
    "sithick",
    "usi",
    "vsi",
    # "mlotst",
]

GIOPS_SURFACE_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

GLORYS_OCEAN_VARIABLES = [
    "thetao",
    "so",
    "uo",
    "vo",
]

GLORYS_VARIABLES = GLORYS_SURFACE_VARIABLES + GLORYS_OCEAN_VARIABLES

CONTEXT_VARIABLES = [
    "deptho",
    "land_sea_mask",
    # "distance_to_coast",
    # "eastward_bathymetric_gradient",
    # "northward_bathymetric_gradient",
]

# Corresponding atmospheric variables for surface variables
ATM_SURF_VARIABLE_MAPPINGS = {
    "2m_temperature": "temperature",
    "10m_u_component_of_wind": "u_component_of_wind",
    "10m_v_component_of_wind": "v_component_of_wind",
}

DATA_NAN_VALUE = 0.0
DATASET_DATES_TRAINING = ("2010-01-01", "2023-12-31")
DATASET_DATES_VALIDATION = ("2024-01-01", "2024-12-31")
DATASET_DATES_TEST = ("2025-01-01", "2025-12-31")

# Toy dataset
TOY_DATASET_VARIABLES = ["2m_temperature", "temperature"]
TOY_DATASET_DATES_TRAINING = ("2017-07-01", "2017-09-30")
TOY_DATASET_DATES_VALIDATION = ("2018-07-01", "2018-09-30")
TOY_DATASET_DATES_TEST = ("2019-07-01", "2019-09-30")

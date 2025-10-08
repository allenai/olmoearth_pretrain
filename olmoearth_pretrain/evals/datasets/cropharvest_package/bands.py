"""For easier normalization of the band values (instead of needing to recompute
the normalization dict with the addition of new data), we provide maximum
values for each band
"""

S1_BANDS = ["VV", "VH"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_MAX_VALUES = [50.0, 50.0]
S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
S2_MAX_VALUES = [1e4] * len(S2_BANDS)
ERA5_BANDS = ["temperature_2m", "total_precipitation"]
# the hottest temperature ever recorded on Earth is 329.85K
# (https://en.wikipedia.org/wiki/List_of_weather_records#Hottest)
# For rainfall, http://www.bom.gov.au/water/designRainfalls/rainfallEvents/worldRecRainfall.shtml
# and https://www.guinnessworldrecords.com/world-records/greatest-monthly-rainfall- suggest
# 10 m is a good estimate. This is (very likely) an overestimate, since the grid is much
# larger than the areas for which these records were achieved
ERA5_MAX_VALUES = [329.85, 10.0]
SRTM_BANDS = ["elevation", "slope"]
# max elevation provided by the Google Earth Engine estimate.
# slope is calculated in degrees by the ee.Terrain.slope function
SRTM_MAX_VALUES = [6500.0, 90.0]

DYNAMIC_BANDS = S1_BANDS + S2_BANDS + ERA5_BANDS
STATIC_BANDS = SRTM_BANDS

DYNAMIC_BANDS_MAX = S1_MAX_VALUES + S2_MAX_VALUES + ERA5_MAX_VALUES
STATIC_BANDS_MAX = SRTM_MAX_VALUES

# These bands are what is created by the Engineer. If the engineer changes, the bands
# here will need to change (and vice versa)
REMOVED_BANDS = ["B1", "B10"]
RAW_BANDS = DYNAMIC_BANDS + STATIC_BANDS

BANDS = [x for x in DYNAMIC_BANDS if x not in REMOVED_BANDS] + STATIC_BANDS + ["NDVI"]

# NDVI is between 0 and 1
BANDS_MAX = (
    [
        DYNAMIC_BANDS_MAX[i]
        for i, x in enumerate(DYNAMIC_BANDS)
        if x not in REMOVED_BANDS
    ]
    + STATIC_BANDS_MAX
    + [1.0]
)

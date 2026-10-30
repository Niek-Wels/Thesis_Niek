"""Download cloud-free Sentinel-2 chips (15x15 px, 10 m) around the sample points, following verytree_wur.

One chip per point per half-year (Jan-Jun, Jul-Dec): the clear date closest to the middle of the half-year.
"""
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (adds .rio to xarray)
from affine import Affine
from odc.geo.geobox import GeoBox
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
Long_Lat_CSV = ROOT / "Data" / "Merged_Uganda_original_cleaned_reference_data_2019_2023.csv"
OUT_DIR = ROOT / "Data" / "chips_halfyear"
LOG_CSV = OUT_DIR / "cloud_log.csv"

YEARS = range(2019, 2024)
HALF_YEARS = [  # (name, first day, last day, middle: the dates closest to it are tried first)
    ("H1", "01-01", "06-30", "04-01"),
    ("H2", "07-01", "12-31", "10-01"),
]
SIZE, RES = 15, 10
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # Blue, Green, Red, NIR-narrow, SWIR1, SWIR2
MAX_SCENE_CLOUD = 30  # % cloud over the whole tile; the chip itself must still be fully clear (CLEAR_SCL)
CLEAR_SCL = [2, 4, 5, 6, 7]  # dark area, vegetation, bare, water, unclassified (verytree); every chip pixel must be one
N_WORKERS = 8  # points downloaded at the same time

# Optional test subset: 2 stable + 2 changing samples per LC 2019 class.
USE_TEST_SUBSET = False
TEST_SUBSET = {
    #  class 2019               stable        change (-> LC 2023)
    "crop":                  ([210, 451],   [445, 510]),   # grass, shrub
    "grass":                 ([212, 534],   [640, 642]),   # other (bare and moss), trees
    "open water":            ([41, 45],     [100, 101]),   # grass
    "other (bare and moss)": ([720, 1275],  [556, 731]),   # grass
    "shrubs":                ([741, 1503],  [582, 616]),   # open water, trees
    "trees":                 ([43, 446],    [500, 226]),   # crop
    "urban":                 ([1377, 1371], [499, 339]),   # grass
    "wetland":               ([422, 470],   [127, 171]),   # open water
}
TEST_SAMPLE_IDS = [i for stable, change in TEST_SUBSET.values() for i in stable + change]

catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


def chip_grid(lon, lat):
    """15x15 grid in the local UTM zone, aligned to the S2 pixels; the point is in pixel [7, 7]."""
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    half = SIZE // 2
    left = math.floor(x / RES) * RES - half * RES
    top = (math.floor(y / RES) + 1) * RES + half * RES
    return GeoBox((SIZE, SIZE), Affine(RES, 0, left, 0, -RES, top), f"EPSG:{epsg}")


def load_day(items, grid, bands, resampling):
    """Tiles of one day merged into one chip; pixels outside the tiles are NaN."""
    return odc.stac.load(items, bands=bands, geobox=grid, groupby="solar_day", resampling=resampling).isel(time=0)


def cloud_fraction(items, grid):
    """Share of chip pixels that are cloud, shadow or outside the tiles."""
    scl = load_day(items, grid, ["SCL"], "nearest").SCL.values
    return (np.isnan(scl) | ~np.isin(scl, CLEAR_SCL)).mean()


def has_offset(item):
    """Processing baseline 04.00+ (from 2022) adds 1000 to every band value."""
    return float(item.properties["s2:processing_baseline"]) >= 4.0


def download_chip(items, grid, path):
    data = load_day(items, grid, BANDS, "cubic").to_array("band").astype("int32")
    if has_offset(items[0]):
        data = data - 1000
    data.clip(min=0).astype("uint16").rio.write_crs(str(grid.crs)).rio.to_raster(path)


def process_point(point):
    grid = chip_grid(point.lon, point.lat)
    items = list(catalog.search(collections=["sentinel-2-l2a"], datetime="2019-01-01/2023-12-31",
                                bbox=grid.geographic_extent.boundingbox.bbox,
                                query={"eo:cloud_cover": {"lt": MAX_SCENE_CLOUD}}).items())
    dates = sorted({f"{item.datetime:%Y-%m-%d}" for item in items})
    (OUT_DIR / str(point.sample_id)).mkdir(parents=True, exist_ok=True)

    log = []
    for year in YEARS:
        for name, first, last, middle in HALF_YEARS:
            in_half = [d for d in dates if f"{year}-{first}" <= d <= f"{year}-{last}"]
            in_half.sort(key=lambda d: abs(pd.Timestamp(d) - pd.Timestamp(f"{year}-{middle}")))
            for date in in_half:
                day_items = [planetary_computer.sign(i) for i in items if f"{i.datetime:%Y-%m-%d}" == date]
                cloud = cloud_fraction(day_items, grid)
                # Tiles with and without the +1000 offset can't be merged into one chip, so such days are skipped.
                one_baseline = len({has_offset(i) for i in day_items}) == 1
                saved = cloud == 0 and one_baseline
                if saved:
                    path = OUT_DIR / str(point.sample_id) / f"{point.sample_id}_{date.replace('-', '')}.tif"
                    download_chip(day_items, grid, path)
                log.append({"sample_id": point.sample_id, "half_year": f"{year}{name}", "date": date,
                            "cloud_frac": round(cloud, 4), "saved": saved})
                if saved:  # one chip per half-year
                    break
    print(f"point {point.sample_id}: {sum(row['saved'] for row in log)} chips")
    return pd.DataFrame(log)


points = pd.read_csv(Long_Lat_CSV)
if USE_TEST_SUBSET:
    points = points[points.sample_id.isin(TEST_SAMPLE_IDS)]
with ThreadPoolExecutor(N_WORKERS) as pool:
    logs = list(pool.map(process_point, points.itertuples()))
pd.concat(logs).to_csv(LOG_CSV, index=False)
print(f"log saved in {LOG_CSV.relative_to(ROOT)}")

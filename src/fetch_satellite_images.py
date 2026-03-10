"""
fetch_satellite_images.py
-------------------------
Reads data/weather_stations.csv, then for every station fetches a
cloud-free Sentinel-2 surface-reflectance image chip (1 km radius, 10 m
resolution) from Google Earth Engine and saves it as a multi-band GeoTIFF.

Output directory : data/images/
Output filenames : {Station_ID}.tif
Band order in TIF: 1=Red (S2 B4), 2=Green (S2 B3), 3=Blue (S2 B2)
Value range      : uint16, raw Sentinel-2 DN (divide by 10 000 → reflectance 0–1)

Usage
-----
    python src/fetch_satellite_images.py

Prerequisites
-------------
    1. Run `earthengine authenticate` once (or set GEE_KEY_FILE in .env).
    2. Set GEE_PROJECT_ID in .env.
    3. pip install earthengine-api rasterio requests tqdm python-dotenv
"""

from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the script works from any cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIONS_CSV = PROJECT_ROOT / "data" / "weather_stations.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"

# ---------------------------------------------------------------------------
# GEE / imagery parameters
# ---------------------------------------------------------------------------
CHIP_RADIUS_M = 1_000          # 1 km radius around each station
SCALE_M = 10                   # Sentinel-2 native resolution (metres)
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
RGB_BANDS = ["B4", "B3", "B2"]  # Red, Green, Blue for S2

# Cloud-cover thresholds tried in order until one yields imagery
CLOUD_THRESHOLDS = [10, 20, 35]

# Date range: summers (Jun–Aug) 2020–2024
DATE_START = "2020-06-01"
DATE_END = "2024-08-31"

# Network settings
REQUEST_TIMEOUT_S = 180
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = 5   # doubles on each retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: GEE authentication
# ---------------------------------------------------------------------------
def authenticate_gee() -> None:
    """Initialize Google Earth Engine using credentials from config.py."""
    # Import here so the module is importable without GEE installed
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from config import init_gee
    init_gee()


# ---------------------------------------------------------------------------
# Step 2: Load stations
# ---------------------------------------------------------------------------
def load_stations() -> pd.DataFrame:
    """Load the station CSV produced by fetch_weather_data.py."""
    if not STATIONS_CSV.exists():
        log.error(
            "Station CSV not found at %s.\n"
            "Run `python src/fetch_weather_data.py` first.",
            STATIONS_CSV,
        )
        sys.exit(1)

    df = pd.read_csv(STATIONS_CSV)
    required = {"Station_ID", "Latitude", "Longitude"}
    if not required.issubset(df.columns):
        log.error("CSV is missing columns: %s", required - set(df.columns))
        sys.exit(1)

    log.info("Loaded %d station(s) from %s", len(df), STATIONS_CSV.name)
    return df


# ---------------------------------------------------------------------------
# Step 3: Build Sentinel-2 cloud-free composite for a given location
# ---------------------------------------------------------------------------
def build_s2_image(lat: float, lon: float, max_cloud_pct: int):
    """
    Return (image, roi) where image is the median S2 RGB composite clipped
    to a CHIP_RADIUS_M-metre buffer around (lat, lon), or (None, roi) if the
    collection is empty after filtering.
    """
    import ee  # imported after init

    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(CHIP_RADIUS_M)

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.calendarRange(6, 8, "month"))   # summer only
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .select(RGB_BANDS)
    )

    size = collection.size().getInfo()
    if size == 0:
        return None, roi

    image = collection.median().clip(roi)
    return image, roi


# ---------------------------------------------------------------------------
# Step 4: Download zip from GEE and stack bands into a GeoTIFF
# ---------------------------------------------------------------------------
def _sort_key_by_band(filename: str) -> int:
    """
    GEE names zip entries like 'download.B4.tif'.
    Return sort index so files come out in B4, B3, B2 order (R, G, B).
    """
    order = {"B4": 0, "B3": 1, "B2": 2}
    for band, idx in order.items():
        if f".{band}." in filename:
            return idx
    return 99  # unknown band → push to end


def _fetch_url_with_retry(url: str) -> bytes:
    """GET a URL, retrying up to RETRY_ATTEMPTS times with exponential back-off."""
    delay = RETRY_BACKOFF_S
    last_exc: Optional[Exception] = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT_S, stream=True)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                log.warning("  Attempt %d failed (%s). Retrying in %ds …", attempt, exc, delay)
                time.sleep(delay)
                delay *= 2

    raise RuntimeError(f"All {RETRY_ATTEMPTS} download attempts failed") from last_exc


def download_chip(image, roi, output_path: Path) -> None:
    """
    Download a GEE image chip directly into a multi-band GeoTIFF at output_path.

    GEE's getDownloadURL with format='GEO_TIFF' now natively returns the 
    multi-band GeoTIFF file directly (as bytes), rather than a ZIP archive.
    """
    import ee  # imported after init

    url: str = image.getDownloadURL(
        {
            "bands": RGB_BANDS,
            "region": roi,
            "scale": SCALE_M,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        }
    )

    raw_bytes = _fetch_url_with_retry(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(raw_bytes)


# ---------------------------------------------------------------------------
# Step 5: Process one station (with progressive cloud fallback)
# ---------------------------------------------------------------------------
def process_station(station_id: str, lat: float, lon: float) -> bool:
    """
    Try downloading a chip for *station_id*, relaxing the cloud threshold
    progressively if no imagery is found. Returns True on success.
    """
    output_path = IMAGES_DIR / f"{station_id}.tif"

    # Resume: skip stations whose image already exists
    if output_path.exists():
        log.info("  [SKIP] %s — image already exists.", station_id)
        return True

    for threshold in CLOUD_THRESHOLDS:
        image, roi = build_s2_image(lat, lon, max_cloud_pct=threshold)

        if image is None:
            log.debug(
                "  No imagery at cloud<=%d%% for %s. Trying higher threshold …",
                threshold, station_id,
            )
            continue

        try:
            download_chip(image, roi, output_path)
            size_kb = output_path.stat().st_size / 1024
            log.info(
                "  [OK]   %s — saved (%.1f KB, cloud<=%d%%)",
                station_id, size_kb, threshold,
            )
            return True

        except Exception as exc:
            log.warning(
                "  [WARN] %s — download failed at cloud<=%d%% (%s)",
                station_id, threshold, exc,
            )
            # Clean up partial file if it was created
            if output_path.exists():
                output_path.unlink()
            break  # network errors won't improve with a looser cloud filter

    log.error("  [FAIL] %s — could not download a usable image.", station_id)
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("  Houston Satellite Image Acquisition")
    print(f"  Source      : Sentinel-2 SR ({S2_COLLECTION})")
    print(f"  Resolution  : {SCALE_M} m/px")
    print(f"  Date range  : {DATE_START} -> {DATE_END} (summers only)")
    print("=" * 60)

    # ── 1. Authenticate ────────────────────────────────────────────
    log.info("[1/3] Authenticating with Google Earth Engine …")
    authenticate_gee()

    # ── 2. Load stations ───────────────────────────────────────────
    log.info("[2/3] Loading station list …")
    stations = load_stations()
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 3. Download one chip per station ───────────────────────────
    log.info("[3/3] Downloading image chips …\n")

    succeeded, failed, skipped = [], [], []

    for _, row in tqdm(stations.iterrows(), total=len(stations), unit="station"):
        sid = str(row["Station_ID"])
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])

        output_path = IMAGES_DIR / f"{sid}.tif"
        if output_path.exists():
            skipped.append(sid)
            tqdm.write(f"  [SKIP] {sid} — already downloaded.")
            continue

        ok = process_station(sid, lat, lon)
        (succeeded if ok else failed).append(sid)

        # Small pause between requests to respect GEE rate limits
        time.sleep(0.5)

    # ── Summary ────────────────────────────────────────────────────
    total = len(stations)
    print("\n" + "=" * 60)
    print(f"  Results: {total} station(s) processed")
    print(f"    Succeeded : {len(succeeded)}")
    print(f"    Skipped   : {len(skipped)}  (already on disk)")
    print(f"    Failed    : {len(failed)}")
    if failed:
        print("\n  Failed Station IDs:")
        for sid in failed:
            print(f"    - {sid}")
        print(
            "\n  Tip: failed stations may have persistent cloud cover or lie\n"
            "  outside the Sentinel-2 archive. Check GEE Code Editor manually."
        )
    print("=" * 60)
    print("\nDone.")


if __name__ == "__main__":
    main()

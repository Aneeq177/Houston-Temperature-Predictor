"""
fetch_weather_data.py
---------------------
Fetches historical average summer temperatures across a regular grid of
coordinates that covers the Houston, TX metro area.

Data source : Open-Meteo Historical Weather API (ERA5 reanalysis)
              https://open-meteo.com/en/docs/historical-weather-api
              No API key required.
Library     : openmeteo-requests  (official Python client with cache + retry)

Grid
----
A 0.10°-step lat/lon grid (~11 km spacing) is laid over the Houston metro
bounding box, producing ~56 evenly-spaced sample points.  Each point is
treated as one "station" in the downstream pipeline.

Output : data/weather_stations.csv
Columns: Station_ID | Latitude | Longitude | Avg_Summer_Temp (°C)

Summer  = June, July, August
Window  = 2020-06-01 → 2024-08-31  (5 full summers)

Usage
-----
    python src/fetch_weather_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "data" / "weather_stations.csv"
CACHE_PATH   = PROJECT_ROOT / "data" / ".om_cache"   # SQLite cache file

# ---------------------------------------------------------------------------
# Houston metro bounding box
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX =  29.52,  30.16   # degrees N
LON_MIN, LON_MAX = -95.79, -95.02   # degrees W (negative = west)
GRID_STEP        =  0.10            # ~11 km spacing

# ---------------------------------------------------------------------------
# Open-Meteo request parameters
# ---------------------------------------------------------------------------
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATE_START     = "2023-08-01"
DATE_END       = "2023-08-31"
SUMMER_MONTHS  = {8}
TIMEZONE       = "America/Chicago"

# Number of locations per API request (conservative to avoid timeouts)
CHUNK_SIZE = 10

MIN_SUMMER_DAYS = 15   # discard grid points with fewer valid daily readings


# ===========================================================================
# Step 1: Build the coordinate grid
# ===========================================================================

def build_grid() -> tuple[list[float], list[float], list[str]]:
    """
    Return (latitudes, longitudes, station_ids) for a regular grid over
    the Houston metro bounding box.

    Station ID format: OM_N{lat*100:.0f}_W{abs(lon)*100:.0f}
    Example         : OM_N2952_W9579  (29.52°N, 95.79°W)
    """
    lat_ticks = np.arange(LAT_MIN, LAT_MAX + GRID_STEP / 2, GRID_STEP)
    lon_ticks = np.arange(LON_MIN, LON_MAX + GRID_STEP / 2, GRID_STEP)

    lats, lons, ids = [], [], []
    for lat in lat_ticks:
        for lon in lon_ticks:
            lat_r = round(float(lat), 2)
            lon_r = round(float(lon), 2)
            lats.append(lat_r)
            lons.append(lon_r)
            ids.append(f"OM_N{round(lat_r * 100):04d}_W{round(abs(lon_r) * 100):04d}")

    return lats, lons, ids


# ===========================================================================
# Step 2: Open-Meteo client (with disk cache + automatic retry)
# ===========================================================================

def build_client() -> openmeteo_requests.Client:
    """
    Return an openmeteo_requests.Client backed by:
      - requests_cache  → responses are stored in data/.om_cache.sqlite
                          so re-running the script never hits the API twice
                          for the same coordinates + date range.
      - retry_requests  → automatic exponential-backoff retry on 5xx / network
                          errors (up to 5 attempts).
    """
    cache_session = requests_cache.CachedSession(
        str(CACHE_PATH),
        expire_after=-1,        # cache never expires — ERA5 historical data is stable
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
    return openmeteo_requests.Client(session=retry_session)


# ===========================================================================
# Step 3: Fetch temperature data in chunks and compute summer averages
# ===========================================================================

def fetch_chunk(
    client: openmeteo_requests.Client,
    lats: list[float],
    lons: list[float],
) -> list[dict]:
    """
    Fetch daily mean temperature for one chunk of coordinates.
    Returns a list of raw dicts with keys: lat, lon, daily_df.
    """
    params = {
        "latitude":         lats,
        "longitude":        lons,
        "start_date":       DATE_START,
        "end_date":         DATE_END,
        "daily":            "temperature_2m_max",
        "temperature_unit": "celsius",
        "timezone":         TIMEZONE,
    }
    responses = client.weather_api(OM_ARCHIVE_URL, params=params)

    results = []
    for resp in responses:
        daily   = resp.Daily()
        # Build a date index aligned to the API's time vector
        time_ix = pd.date_range(
            start     = pd.to_datetime(daily.Time(),    unit="s", utc=True),
            end       = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq      = pd.Timedelta(seconds=daily.Interval()),
            inclusive = "left",
        ).tz_convert(TIMEZONE).tz_localize(None)   # strip tz for convenience

        temp_values = daily.Variables(0).ValuesAsNumpy()  # temperature_2m_mean
        results.append({
            "lat":      round(float(resp.Latitude()),  2),
            "lon":      round(float(resp.Longitude()), 2),
            "daily_df": pd.DataFrame({"date": time_ix, "temp": temp_values}),
        })
    return results


def compute_summer_avg(daily_df: pd.DataFrame) -> float | None:
    """Filter to summer months and return the mean temperature, or None."""
    summer = daily_df[daily_df["date"].dt.month.isin(SUMMER_MONTHS)]["temp"].dropna()
    if len(summer) < MIN_SUMMER_DAYS:
        return None
    return round(float(summer.mean()), 2)


def fetch_all_temperatures(
    client: openmeteo_requests.Client,
    lats: list[float],
    lons: list[float],
    ids: list[str],
) -> pd.DataFrame:
    """
    Iterate through the grid in CHUNK_SIZE batches, fetch data, compute
    summer averages, and return a combined DataFrame.
    """
    n      = len(lats)
    chunks = range(0, n, CHUNK_SIZE)
    records: list[dict] = []
    skipped = 0

    for start in tqdm(chunks, desc="Fetching chunks", unit="chunk"):
        end         = min(start + CHUNK_SIZE, n)
        chunk_lats  = lats[start:end]
        chunk_lons  = lons[start:end]
        chunk_ids   = ids[start:end]

        try:
            raw = fetch_chunk(client, chunk_lats, chunk_lons)
        except Exception as exc:
            print(f"\n  [WARN] Chunk {start}–{end} failed: {exc}", file=sys.stderr)
            skipped += len(chunk_lats)
            continue

        # Match responses back to IDs by position (API preserves order)
        for i, item in enumerate(raw):
            avg_temp = compute_summer_avg(item["daily_df"])
            if avg_temp is None:
                skipped += 1
                continue
            records.append({
                "Station_ID":      chunk_ids[i],
                "Latitude":        item["lat"],
                "Longitude":       item["lon"],
                "Avg_Summer_Temp": avg_temp,
            })

    if skipped:
        print(f"\n  Skipped {skipped} grid point(s) with insufficient data.")

    if not records:
        print("ERROR: No temperature data retrieved.", file=sys.stderr)
        sys.exit(1)

    return pd.DataFrame(records, columns=["Station_ID", "Latitude", "Longitude", "Avg_Summer_Temp"])


# ===========================================================================
# Step 4: Save
# ===========================================================================

def save(df: pd.DataFrame) -> None:
    df = df.drop_duplicates(subset="Station_ID")
    df = df.sort_values("Avg_Summer_Temp", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n  Saved {len(df)} grid point(s) → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\n  Temperature range: {df['Avg_Summer_Temp'].min():.2f} – "
          f"{df['Avg_Summer_Temp'].max():.2f} °C  "
          f"(mean {df['Avg_Summer_Temp'].mean():.2f} °C)")
    print()
    print(df.to_string(index=False))


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    lats, lons, ids = build_grid()

    print("=" * 60)
    print("  Houston Weather Data Acquisition — Open-Meteo ERA5")
    print(f"  Grid        : {len(lats)} points  "
          f"({GRID_STEP}° step, ~{GRID_STEP * 111:.0f} km spacing)")
    print(f"  Bounding box: {LAT_MIN}°–{LAT_MAX}°N, {LON_MIN}°–{LON_MAX}°W")
    print(f"  Summer window: Jun–Aug  {DATE_START[:4]}–{DATE_END[:4]}")
    print(f"  API cache    : {CACHE_PATH.relative_to(PROJECT_ROOT)}.sqlite")
    print("=" * 60)

    print("\n[1/3] Building coordinate grid …")
    print(f"      {len(lats)} grid points generated.")

    print("\n[2/3] Fetching temperature data from Open-Meteo …")
    print("      (Cached responses are reused on subsequent runs.)\n")
    client = build_client()
    df = fetch_all_temperatures(client, lats, lons, ids)

    print("\n[3/3] Saving results …")
    save(df)

    print("\nDone.")


if __name__ == "__main__":
    main()

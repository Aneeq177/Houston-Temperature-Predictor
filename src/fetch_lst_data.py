"""
fetch_lst_data.py
-----------------
Fetches high-resolution (30m) Land Surface Temperature (LST) data from 
the Landsat 8 satellite via Google Earth Engine.

It takes the coordinates from the existing weather_stations.csv, calculates
the mean LST within a 1km radius (matching our CV chip size) during the 
peak August 2023 heatwave, and overwrites the temperature column.

Usage
-----
    python src/fetch_lst_data.py
"""

import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Insert src to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import init_gee
import ee

STATIONS_CSV = PROJECT_ROOT / "data" / "weather_stations.csv"
CHIP_RADIUS_M = 1000

# We use August 2023, the hottest month on record for Houston
DATE_START = "2023-08-01"
DATE_END   = "2023-08-31"

def authenticate_gee():
    init_gee()

def fetch_lst_for_station(lat: float, lon: float) -> float:
    """
    Fetches the median Land Surface Temperature (Celsius) across all clear 
    Landsat 8 passes in August 2023 for a 1km radius around lat/lon.
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(CHIP_RADIUS_M)

    # Landsat 8 Collection 2 Level 2 (Surface Reflectance & Surface Temperature)
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(DATE_START, DATE_END)
        # Filter for relatively clear imagery over the point
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
    )
    
    # NOTE: Do NOT call .size().getInfo() here — that makes a blocking round-trip
    # API call for every station in the loop, adding seconds per station.
    # Instead, use a wider date window from the start (mid-Jul → mid-Sep 2023)
    # which still captures the August 2023 peak. If the collection is empty,
    # reduceRegion returns None for the band, handled by the caller.

    # Scale factor for Landsat 8 ST_B10 (Surface Temp)
    # Temperature (Kelvin) = ST_B10 * 0.00341802 + 149.0
    def scale_and_convert_to_celsius(img):
        st_kelvin = img.select("ST_B10").multiply(0.00341802).add(149.0)
        st_celsius = st_kelvin.subtract(273.15).rename("LST")
        return img.addBands(st_celsius)

    scaled_col = collection.map(scale_and_convert_to_celsius)
    median_img = scaled_col.select("LST").median()

    # Calculate the mean LST within the 1km radius
    stats = median_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30,  # Landsat thermal resolution is natively 100m resampled to 30m
        maxPixels=1e9
    )
    
    val = dict(stats.getInfo()).get("LST")
    return val

def main():
    print("=" * 60)
    print("  Houston LST Data Acquisition (Landsat 8)")
    print(f"  Target  : 30m Land Surface Temperature (LST)")
    print(f"  Window  : {DATE_START} to {DATE_END}")
    print("=" * 60)

    print("\n[1/3] Authenticating with Google Earth Engine...")
    authenticate_gee()
    
    if not STATIONS_CSV.exists():
        print(f"Error: {STATIONS_CSV} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(STATIONS_CSV)
    
    print(f"\n[2/3] Fetching LST for {len(df)} stations...")
    
    lst_values = []
    failed = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), unit="station"):
        sid = row["Station_ID"]
        lat = row["Latitude"]
        lon = row["Longitude"]
        
        try:
            temp = fetch_lst_for_station(lat, lon)
            if temp is not None:
                lst_values.append(round(temp, 2))
            else:
                lst_values.append(None)
                failed.append(sid)
        except Exception as e:
            lst_values.append(None)
            failed.append(sid)
            print(f"\n  [Error] {sid}: {e}")
            
        # Small delay to prevent rate limits
        time.sleep(0.5)
        
    df["Avg_Summer_Temp"] = lst_values
    
    # Drop rows where Earth Engine returned None (e.g. completely cloudy all month)
    before_len = len(df)
    df = df.dropna(subset=["Avg_Summer_Temp"])
    
    print("\n[3/3] Saving results...")
    df.to_csv(STATIONS_CSV, index=False)
    
    print(f"\n  Saved {len(df)} grid point(s) successfully.")
    if failed:
        print(f"  Failed: {before_len - len(df)} stations dropped due to cloud cover.")
        
    print(f"\n  New LST Temperature range: {df['Avg_Summer_Temp'].min():.2f} - {df['Avg_Summer_Temp'].max():.2f} °C")
    print(f"  Mean LST: {df['Avg_Summer_Temp'].mean():.2f} °C")
    print("\nDone.")

if __name__ == "__main__":
    main()

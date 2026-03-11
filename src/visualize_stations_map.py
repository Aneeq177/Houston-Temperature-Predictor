"""
visualize_stations_map.py
-------------------------
Creates an interactive Folium map of all Houston weather stations,
colour-coded by Land Surface Temperature (LST).

Output : results/houston_stations_map.html

Usage
-----
    python src/visualize_stations_map.py
"""

from pathlib import Path

import pandas as pd
import folium
from folium.plugins import MiniMap

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIONS_CSV = PROJECT_ROOT / "data" / "weather_stations.csv"
OUTPUT_HTML  = PROJECT_ROOT / "results" / "houston_stations_map.html"

# ── Colour scale (cool → hot) ─────────────────────────────────────────────────
# Interpolates between five stops across the LST range
COLOUR_STOPS = [
    (0.00, (70,  130, 220)),   # cool blue
    (0.25, (80,  200, 160)),   # teal-green
    (0.50, (255, 220,  50)),   # yellow
    (0.75, (255, 130,  30)),   # orange
    (1.00, (220,  30,  30)),   # hot red
]


def lerp_colour(t: float) -> str:
    """
    Return a CSS hex colour for a normalised value t in [0, 1].
    Uses piecewise linear interpolation across COLOUR_STOPS.
    """
    t = max(0.0, min(1.0, t))
    for i in range(len(COLOUR_STOPS) - 1):
        t0, c0 = COLOUR_STOPS[i]
        t1, c1 = COLOUR_STOPS[i + 1]
        if t <= t1:
            ratio = (t - t0) / (t1 - t0)
            r = int(c0[0] + ratio * (c1[0] - c0[0]))
            g = int(c0[1] + ratio * (c1[1] - c0[1]))
            b = int(c0[2] + ratio * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{COLOUR_STOPS[-1][1][0]:02x}{COLOUR_STOPS[-1][1][1]:02x}{COLOUR_STOPS[-1][1][2]:02x}"


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    if not STATIONS_CSV.exists():
        raise FileNotFoundError(
            f"Station CSV not found at {STATIONS_CSV}.\n"
            "Run 'python src/fetch_weather_data.py' first."
        )

    df = pd.read_csv(STATIONS_CSV)
    required = {"Station_ID", "Latitude", "Longitude", "Avg_Summer_Temp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df.dropna(subset=["Latitude", "Longitude", "Avg_Summer_Temp"])
    n = len(df)

    temp_min = df["Avg_Summer_Temp"].min()
    temp_max = df["Avg_Summer_Temp"].max()
    temp_mean = df["Avg_Summer_Temp"].mean()

    print(f"  Stations loaded : {n}")
    print(f"  LST range       : {temp_min:.2f} – {temp_max:.2f} °C  (mean {temp_mean:.2f} °C)")

    # ── Build map ─────────────────────────────────────────────────────────────
    center_lat = df["Latitude"].mean()
    center_lon = df["Longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Secondary tile layers the user can toggle
    folium.TileLayer("OpenStreetMap",    name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Matter").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
    ).add_to(m)

    # ── Station markers ───────────────────────────────────────────────────────
    temp_range = temp_max - temp_min if temp_max > temp_min else 1.0

    for _, row in df.iterrows():
        t        = (row["Avg_Summer_Temp"] - temp_min) / temp_range
        colour   = lerp_colour(t)
        temp_val = row["Avg_Summer_Temp"]
        sid      = row["Station_ID"]

        popup_html = f"""
        <div style="font-family: monospace; font-size: 13px; min-width: 210px;">
            <b style="font-size:14px;">{sid}</b><br>
            <hr style="margin:4px 0;">
            <table style="border-collapse:collapse; width:100%;">
                <tr>
                    <td style="padding:2px 6px 2px 0; color:#555;">LST</td>
                    <td style="padding:2px 0;">
                        <b style="color:{colour}; font-size:15px;">{temp_val:.2f} °C</b>
                    </td>
                </tr>
                <tr>
                    <td style="padding:2px 6px 2px 0; color:#555;">Latitude</td>
                    <td style="padding:2px 0;">{row['Latitude']:.4f}°N</td>
                </tr>
                <tr>
                    <td style="padding:2px 6px 2px 0; color:#555;">Longitude</td>
                    <td style="padding:2px 0;">{row['Longitude']:.4f}°W</td>
                </tr>
                <tr>
                    <td style="padding:2px 6px 2px 0; color:#555;">vs. mean</td>
                    <td style="padding:2px 0;">{temp_val - temp_mean:+.2f} °C</td>
                </tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=10,
            color="#333333",
            weight=0.8,
            fill=True,
            fill_color=colour,
            fill_opacity=0.88,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{sid}  |  {temp_val:.2f} °C",
        ).add_to(m)

    # ── Colour legend (HTML element injected into the map) ────────────────────
    gradient_css = ", ".join(
        f"{lerp_colour(i / 10)} {i * 10}%" for i in range(11)
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px; right: 20px; z-index: 1000;
        background: rgba(255,255,255,0.93);
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.18);
        min-width: 180px;
    ">
        <b style="font-size:13px;">Land Surface Temp. (LST)</b>
        <div style="
            margin: 8px 0 4px 0;
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(to right, {gradient_css});
            border: 1px solid #bbb;
        "></div>
        <div style="display:flex; justify-content:space-between;">
            <span>{temp_min:.1f} °C (cool)</span>
            <span>{temp_max:.1f} °C (hot)</span>
        </div>
        <hr style="margin:8px 0 6px 0; border-color:#ddd;">
        <div>n = {n} stations</div>
        <div>Mean LST: <b>{temp_mean:.2f} °C</b></div>
        <div style="font-size:10px; color:#888; margin-top:4px;">
            Source: Landsat 8, August 2023
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title banner ──────────────────────────────────────────────────────────
    title_html = """
    <div style="
        position: fixed;
        top: 12px; left: 50%; transform: translateX(-50%);
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 8px 20px;
        font-family: Arial, sans-serif;
        font-size: 15px;
        font-weight: bold;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.18);
        pointer-events: none;
    ">
        Houston Urban Heat Island — Weather Station LST Map
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Mini-map & layer control ──────────────────────────────────────────────
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUTPUT_HTML))
    print(f"\n  Saved -> {OUTPUT_HTML.relative_to(PROJECT_ROOT)}")
    print("  Open the HTML file in any browser to view the interactive map.")


if __name__ == "__main__":
    main()

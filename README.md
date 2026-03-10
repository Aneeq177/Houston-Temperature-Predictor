# Houston Surface Composition & Temperature Correlation Study

An honors computer vision research project analyzing the relationship between
surface composition (green space vs. impervious surfaces) and historical
temperatures across Houston neighborhoods.

---

## Project Goals

1. **Quantify surface composition** — Use Google Earth Engine satellite imagery
   and computer vision (semantic segmentation / NDVI analysis) to classify land
   cover per neighborhood into green space, concrete, asphalt, water, etc.

2. **Correlate with temperature data** — Pair surface-composition metrics with
   historical temperature records (NOAA / Land Surface Temperature from
   Landsat/MODIS) at the neighborhood level.

3. **Identify heat-island patterns** — Apply statistical models to determine
   which surface features are the strongest predictors of elevated temperatures
   (urban heat island effect).

4. **Visualize findings** — Produce publication-quality maps and charts showing
   spatial temperature gradients overlaid with land-cover classifications.

---

## Project Structure

```
.
├── data/
│   ├── raw/          # Original, unmodified source data (GeoJSON, TIF, CSV)
│   └── processed/    # Cleaned and feature-engineered datasets
├── notebooks/        # Exploratory Jupyter notebooks
├── src/
│   ├── config.py     # Central config (paths, API setup)
│   ├── gee.py        # Google Earth Engine data fetching helpers
│   ├── vision.py     # CV / segmentation pipeline (PyTorch)
│   ├── analysis.py   # Statistical correlation & modelling
│   └── viz.py        # Mapping and plotting utilities
├── .env.template     # API key template (copy to .env)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup

### 1. Clone & create a virtual environment
```bash
git clone <repo-url>
cd <repo-directory>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API keys
```bash
cp .env.template .env
# Edit .env with your actual API keys
```

### 4. Authenticate with Google Earth Engine
```bash
earthengine authenticate
```
Or, for a service account, set `GEE_KEY_FILE` in `.env` and the project's
`src/config.py` will handle initialization automatically.

---

## Key Libraries

| Library | Purpose |
|---|---|
| `geopandas` / `rasterio` | Read/write geospatial vector & raster data |
| `earthengine-api` | Access satellite imagery via Google Earth Engine |
| `torch` / `torchvision` | Deep-learning segmentation models |
| `scikit-learn` | Regression, clustering, model evaluation |
| `pandas` / `numpy` | Tabular data wrangling |
| `matplotlib` / `seaborn` | Statistical visualization |

---

## Data Sources

- **Satellite Imagery** — Landsat 8/9 & Sentinel-2 via Google Earth Engine
- **Land Surface Temperature** — MODIS MOD11A1 / Landsat LST products
- **Neighborhood Boundaries** — City of Houston Open Data / US Census TIGER
- **Historical Weather** — NOAA GHCN / Open-Meteo API

---

## Authors

- Aneeq — Honors Research Project

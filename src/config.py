"""
Central configuration module.
Loads environment variables from .env and initializes Google Earth Engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of where the script is run from)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# -------------------------------------------------------------------
# Directory paths
# -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
RAW_DIR = PROJECT_ROOT / os.getenv("RAW_DIR", "data/raw")
PROCESSED_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DIR", "data/processed")

# Create directories if they don't exist
for _dir in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# API credentials
# -------------------------------------------------------------------
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID")
GEE_SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT")
GEE_KEY_FILE = os.getenv("GEE_KEY_FILE")

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

# -------------------------------------------------------------------
# Google Earth Engine initialization
# -------------------------------------------------------------------
def init_gee():
    """Initialize GEE using a service account key if provided, else interactive OAuth."""
    import ee

    if GEE_SERVICE_ACCOUNT and GEE_KEY_FILE:
        credentials = ee.ServiceAccountCredentials(
            email=GEE_SERVICE_ACCOUNT,
            key_file=GEE_KEY_FILE,
        )
        ee.Initialize(credentials=credentials, project=GEE_PROJECT_ID)
    else:
        # Falls back to `earthengine authenticate` OAuth token
        ee.Initialize(project=GEE_PROJECT_ID)

    print(f"[GEE] Initialized with project: {GEE_PROJECT_ID}")

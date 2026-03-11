"""
segment_surfaces.py
-------------------
Classifies surface type for each GeoTIFF in data/images/ using spectral
indices derived directly from Sentinel-2 bands.

Primary method : NDVI  (Normalized Difference Vegetation Index)
                 Requires a 4-band TIF with B2, B3, B4, B8 bands
                 (produced by fetch_satellite_images.py).

                 NDVI = (NIR – Red) / (NIR + Red)

                 Thresholds:
                   NDVI  > 0.30  → Green Space   (healthy vegetation)
                   NDVI  ≤ 0.30  and > –0.05 → Impervious  (built-up, bare soil)
                   NDVI  ≤ –0.05            → Other       (water, deep shadow)

Fallback method: ExG   (Excess Green Index, RGB-only)
                 Used automatically for legacy 3-band TIFs.

Outputs
-------
  data/surface_analysis.csv        Station_ID | Pct_Green | Pct_Impervious | Pct_Other | Method
  data/visuals/{station_id}.png    3-panel figure: RGB | mask | overlay

Usage
-----
    python src/segment_surfaces.py

    # Skip saving overlay PNGs (faster):
    python src/segment_surfaces.py --no-visuals
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR   = PROJECT_ROOT / "data" / "images"
VISUALS_DIR  = PROJECT_ROOT / "data" / "visuals"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "surface_analysis.csv"

# ---------------------------------------------------------------------------
# NDVI classification thresholds
# ---------------------------------------------------------------------------
NDVI_GREEN_MIN      =  0.30   # NDVI above this → vegetation / green space
NDVI_IMPERVIOUS_MIN = -0.05   # NDVI above this (and ≤ GREEN_MIN) → impervious
                               # NDVI at or below this → Other (water, shadow)

# Colour scheme for visualisations (RGB 0-255)
_C_GREEN      = (34,  139,  34)   # forest green
_C_IMPERVIOUS = (180,  60,  60)   # brick red
_C_OTHER      = (150, 150, 150)   # mid-grey
_ALPHA        = 160               # overlay transparency (0-255)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


# ===========================================================================
# 1. Image I/O
# ===========================================================================

def read_tif_bands(tif_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read a Sentinel-2 GeoTIFF and return (rgb_uint8, ndvi_or_None).

    4-band TIF (B2, B3, B4, B8 — alphabetical GEE order):
        Band 1 = Blue  (B2)
        Band 2 = Green (B3)
        Band 3 = Red   (B4)
        Band 4 = NIR   (B8)
      → NDVI is computed and returned.

    3-band TIF (legacy, B4/B3/B2 only):
      → ndvi is None; caller falls back to ExG.

    The returned rgb_uint8 is a uint8 (H, W, 3) array in R-G-B order,
    stretched for display.
    """
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        if n_bands >= 4:
            data = src.read([1, 2, 3, 4]).astype(np.float32)
            blue, green, red, nir = data[0], data[1], data[2], data[3]
            ndvi = (nir - red) / (nir + red + 1e-6)
        else:
            # Legacy 3-band: GEE sorted B2, B3, B4 → Blue, Green, Red
            data = src.read([1, 2, 3]).astype(np.float32)
            blue, green, red = data[0], data[1], data[2]
            ndvi = None

    # Build RGB for display (Red, Green, Blue channel order)
    rgb_raw = np.stack([red, green, blue], axis=2)   # (H, W, 3)

    valid = rgb_raw[rgb_raw > 0]
    p2, p98 = np.percentile(valid, (2, 98)) if valid.size > 0 else (0, 1)
    p98 = max(p98, p2 + 1)   # avoid zero division

    rgb_uint8 = (np.clip((rgb_raw - p2) / (p98 - p2), 0.0, 1.0) * 255).astype(np.uint8)
    return rgb_uint8, ndvi


# ===========================================================================
# 2. Segmentation
# ===========================================================================

def segment_with_ndvi(ndvi: np.ndarray) -> np.ndarray:
    """
    Classify pixels using NDVI thresholds.

    Returns cat_map (uint8, same shape as ndvi):
      0 = Other      (water, shadow: NDVI ≤ –0.05)
      1 = Green      (vegetation:   NDVI  > +0.30)
      2 = Impervious (built-up:     –0.05 < NDVI ≤ +0.30)
    """
    cat = np.zeros(ndvi.shape, dtype=np.uint8)
    cat[ndvi > NDVI_GREEN_MIN]                                    = 1
    cat[(ndvi > NDVI_IMPERVIOUS_MIN) & (ndvi <= NDVI_GREEN_MIN)] = 2
    return cat


def segment_with_exg(rgb: np.ndarray) -> np.ndarray:
    """
    Fallback for 3-band TIFs: Excess Green Index (ExG = 2G – R – B).

    Returns cat_map:
      1 = Green      (ExG > 0.05)
      2 = Impervious (everything else)
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    total = r + g + b + 1e-6
    exg   = (2.0 * g - r - b) / total

    cat = np.full(exg.shape, 2, dtype=np.uint8)   # default: impervious
    cat[exg > 0.05] = 1
    return cat


# ===========================================================================
# 3. Percentage calculation
# ===========================================================================

def compute_percentages(cat_map: np.ndarray) -> Tuple[float, float]:
    """Return (pct_green, pct_impervious) as percentages (0–100)."""
    n = cat_map.size
    pct_green      = round(100.0 * (cat_map == 1).sum() / n, 2)
    pct_impervious = round(100.0 * (cat_map == 2).sum() / n, 2)
    return pct_green, pct_impervious


# ===========================================================================
# 4. Visualisation
# ===========================================================================

def _build_colour_mask(cat_map: np.ndarray) -> np.ndarray:
    mask = np.full((*cat_map.shape, 3), _C_OTHER, dtype=np.uint8)
    mask[cat_map == 1] = _C_GREEN
    mask[cat_map == 2] = _C_IMPERVIOUS
    return mask


def save_visual(
    rgb: np.ndarray,
    cat_map: np.ndarray,
    station_id: str,
    pct_green: float,
    pct_impervious: float,
    method: str,
) -> None:
    """Save a 3-panel PNG to data/visuals/{station_id}.png."""
    colour_mask = _build_colour_mask(cat_map)

    orig_rgba = Image.fromarray(rgb).convert("RGBA")
    mask_rgba = Image.fromarray(
        np.dstack([colour_mask, np.full(cat_map.shape, _ALPHA, dtype=np.uint8)]),
        mode="RGBA",
    )
    overlay = Image.alpha_composite(orig_rgba, mask_rgba)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Satellite Image (RGB)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(colour_mask)
    axes[1].set_title("Surface Classification", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay  |  Green: {pct_green:.1f}%  |  Impervious: {pct_impervious:.1f}%",
        fontsize=11,
    )
    axes[2].axis("off")

    pct_other = round(100.0 - pct_green - pct_impervious, 2)
    legend_patches = [
        mpatches.Patch(color=[c / 255 for c in _C_GREEN],
                       label=f"Green Space ({pct_green:.1f}%)"),
        mpatches.Patch(color=[c / 255 for c in _C_IMPERVIOUS],
                       label=f"Impervious ({pct_impervious:.1f}%)"),
        mpatches.Patch(color=[c / 255 for c in _C_OTHER],
                       label=f"Other ({pct_other:.1f}%)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Station {station_id}   [{method}]",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = VISUALS_DIR / f"{station_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 5. Per-station processing
# ===========================================================================

def process_station(
    tif_path: Path,
    save_visual_flag: bool,
) -> Optional[dict]:
    """Process a single station TIF. Returns a result dict or None on failure."""
    station_id = tif_path.stem

    try:
        rgb, ndvi = read_tif_bands(tif_path)

        if ndvi is not None:
            cat_map = segment_with_ndvi(ndvi)
            method  = "NDVI"
        else:
            cat_map = segment_with_exg(rgb)
            method  = "ExG-fallback"

        pct_green, pct_impervious = compute_percentages(cat_map)

        if save_visual_flag:
            save_visual(rgb, cat_map, station_id, pct_green, pct_impervious, method)

        log.info(
            "  [OK] %s — Green: %5.1f%%  Impervious: %5.1f%%  Other: %5.1f%%  [%s]",
            station_id, pct_green, pct_impervious,
            100.0 - pct_green - pct_impervious, method,
        )

        return {
            "Station_ID":     station_id,
            "Pct_Green":      pct_green,
            "Pct_Impervious": pct_impervious,
            "Pct_Other":      round(100.0 - pct_green - pct_impervious, 2),
            "Method":         method,
        }

    except Exception as exc:
        log.error("  [FAIL] %s — %s", station_id, exc)
        return None


# ===========================================================================
# 6. Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment satellite images into surface types.")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip saving overlay PNG files (faster).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_visuals = not args.no_visuals

    tif_files = sorted(IMAGES_DIR.glob("*.tif"))
    if not tif_files:
        log.error(
            "No .tif files found in %s.\n"
            "Run `python src/fetch_satellite_images.py` first.",
            IMAGES_DIR,
        )
        sys.exit(1)

    # Peek at the first TIF to report band count
    with rasterio.open(tif_files[0]) as _src:
        n_bands = _src.count
    method_label = "NDVI (4-band)" if n_bands >= 4 else "ExG fallback (3-band)"

    print("=" * 62)
    print("  Houston Surface Segmentation")
    print(f"  Images  : {len(tif_files)} stations found in {IMAGES_DIR.name}/")
    print(f"  Method  : {method_label}")
    print(f"  Visuals : {'data/visuals/' if save_visuals else 'disabled'}")
    print(f"  Output  : {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print("=" * 62)

    results: list[dict] = []
    failed:  list[str]  = []

    for tif_path in tqdm(tif_files, unit="station"):
        record = process_station(tif_path, save_visual_flag=save_visuals)
        if record:
            results.append(record)
        else:
            failed.append(tif_path.stem)

    if results:
        df = pd.DataFrame(results, columns=[
            "Station_ID", "Pct_Green", "Pct_Impervious", "Pct_Other", "Method",
        ])
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        print("\n" + "=" * 62)
        print(f"  Saved {len(df)} record(s) → {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
        print("\n  Summary statistics:")
        print(df[["Pct_Green", "Pct_Impervious", "Pct_Other"]].describe().to_string())

    print("\n" + "=" * 62)
    print(f"  Succeeded : {len(results)}")
    print(f"  Failed    : {len(failed)}")
    if save_visuals and results:
        print(f"  Visuals   : {VISUALS_DIR.relative_to(PROJECT_ROOT)}/")
    if failed:
        print(f"\n  Failed station IDs: {', '.join(failed)}")
    print("=" * 62)
    print("\nDone.")


if __name__ == "__main__":
    main()

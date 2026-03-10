"""
segment_surfaces.py
-------------------
Runs semantic segmentation on each GeoTIFF in data/images/ using a
SegFormer model fine-tuned on ADE20K (150 urban/outdoor land-cover classes).
Classes are collapsed into two research-relevant groups:
  - Green Space     (trees, grass, plants, fields, hills …)
  - Impervious      (roads, buildings, sidewalks, bridges …)

The remaining pixels are labelled "Other" (sky, water, parked cars, etc.).

Primary model : nvidia/segformer-b2-finetuned-ade-512-512  (~80 MB download)
Fallback       : Excess-Green colour index (pure NumPy, no model needed)
                 Used automatically when `transformers` is not installed.

Outputs
-------
  data/surface_analysis.csv        Station_ID | Pct_Green | Pct_Impervious | Method
  data/visuals/{station_id}.png    3-panel figure: RGB | mask | overlay

Usage
-----
    python src/segment_surfaces.py

    # Skip visuals (faster batch run):
    python src/segment_surfaces.py --no-visuals

    # Force colour-index fallback (no GPU/internet needed):
    python src/segment_surfaces.py --fallback
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive — safe for headless servers
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
# SegFormer model (ADE20K, 150 classes, 0-indexed)
# ---------------------------------------------------------------------------
HF_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K class index → human-readable label (abbreviated for the classes we care about)
# Full 150-class list: https://groups.csail.mit.edu/vision/datasets/ADE20K/
GREEN_SPACE_IDS: frozenset[int] = frozenset([
    4,   # tree
    9,   # grass
    13,  # earth, ground          (bare soil — still "soft" surface)
    17,  # plant, flora
    29,  # field
    66,  # flower
    68,  # hill
    72,  # palm tree
    94,  # land, ground, soil
])

IMPERVIOUS_IDS: frozenset[int] = frozenset([
    0,   # wall
    1,   # building, edifice
    6,   # road, route
    11,  # sidewalk, pavement
    20,  # car (proxy for paved surface)
    25,  # house
    32,  # fence
    43,  # signboard, sign
    48,  # skyscraper
    51,  # grandstand
    52,  # path
    54,  # runway
    61,  # bridge
    80,  # bus
    83,  # truck
    84,  # tower
    87,  # streetlight
    91,  # dirt track
    93,  # pole
    101, # stage
    102, # van
    109, # swimming pool
    116, # motorbike
    122, # storage tank
    127, # bicycle
    136, # traffic light
    140, # pier, wharf, dock
])

# Colour scheme for visualisations (RGB 0-255)
_C_GREEN      = (34,  139, 34)    # forest green
_C_IMPERVIOUS = (180, 60,  60)    # brick red
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
# 1. Model loading
# ===========================================================================

def load_segformer():
    """
    Download (once) and return (processor, model, device) for
    nvidia/segformer-b2-finetuned-ade-512-512.
    Raises ImportError if `transformers` is not installed.
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading SegFormer model on %s …", device.upper())
    log.info("  Model : %s", HF_MODEL_ID)
    log.info("  (First run downloads ~80 MB; subsequent runs use local cache.)")

    processor = SegformerImageProcessor.from_pretrained(HF_MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(HF_MODEL_ID)
    model.to(device).eval()

    return processor, model, device


# ===========================================================================
# 2. Image I/O and pre-processing
# ===========================================================================

def read_tif_as_rgb(tif_path: Path) -> np.ndarray:
    """
    Read a uint16 Sentinel-2 GeoTIFF (band order: 1=R, 2=G, 3=B)
    and return a uint8 array of shape (H, W, 3).

    Uses per-channel 2nd–98th percentile stretch so both dark
    (dense vegetation) and bright (concrete, rooftops) surfaces
    have good contrast for the segmentation model.
    """
    with rasterio.open(tif_path) as src:
        # Read all three bands; shape → (3, H, W)
        data = src.read([1, 2, 3]).astype(np.float32)

    rgb = np.transpose(data, (1, 2, 0))  # → (H, W, 3)

    # Earth Engine's getDownloadURL natively sorts selected bands alphabetically
    # (B2=Blue, B3=Green, B4=Red) into the GeoTIFF, so we actually read BGR.
    # Reverse the last dimension to make it RGB.
    rgb = rgb[:, :, ::-1]

    # Apply a GLOBAL 2nd-98th percentile stretch to preserve color ratios
    valid_pixels = rgb[rgb > 0]
    p2, p98 = np.percentile(valid_pixels, (2, 98)) if valid_pixels.size > 0 else (0, 1)
    p98 = max(p98, p2 + 1)  # avoid division by zero

    stretched = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    out = (stretched * 255).astype(np.uint8)

    return out


# ===========================================================================
# 3. Inference
# ===========================================================================

def segment_with_segformer(
    rgb: np.ndarray,
    processor,
    model,
    device: str,
) -> np.ndarray:
    """
    Run SegFormer inference.
    Returns class_map of shape (H, W) with ADE20K class indices (0-indexed).
    """
    import torch
    import torch.nn.functional as F

    H, W = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)

    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, 150, H/4, W/4)

    # Upsample back to original resolution
    logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    class_map = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int16)
    return class_map


def segment_with_exg(rgb: np.ndarray) -> np.ndarray:
    """
    Fallback: Excess Green Index (ExG = 2G – R – B) vegetation classifier.
    Works on any RGB image without a neural network.

    Returns a synthetic class_map using ADE20K-compatible indices:
      ExG > threshold → 4  (tree)
      otherwise       → 6  (road)
    """
    r, g, b = rgb[:, :, 0].astype(np.float32), rgb[:, :, 1].astype(np.float32), rgb[:, :, 2].astype(np.float32)
    total = r + g + b + 1e-6
    exg = 2.0 * (g / total) - (r / total) - (b / total)

    class_map = np.where(exg > 0.05, 4, 6).astype(np.int16)
    return class_map


# ===========================================================================
# 4. Pixel classification and percentage calculation
# ===========================================================================

def make_category_map(class_map: np.ndarray) -> np.ndarray:
    """
    Collapse the 150-class ADE20K map into 3 research categories:
      0 = Other / Unknown
      1 = Green Space / Vegetation
      2 = Impervious Surface / Built Environment
    Returns uint8 array of the same shape.
    """
    cat = np.zeros(class_map.shape, dtype=np.uint8)
    cat[np.isin(class_map, list(GREEN_SPACE_IDS))]  = 1
    cat[np.isin(class_map, list(IMPERVIOUS_IDS))]   = 2
    return cat


def compute_percentages(cat_map: np.ndarray) -> Tuple[float, float]:
    """Return (pct_green, pct_impervious) as percentages (0–100, 2 d.p.)."""
    n = cat_map.size
    pct_green      = round(100.0 * (cat_map == 1).sum() / n, 2)
    pct_impervious = round(100.0 * (cat_map == 2).sum() / n, 2)
    return pct_green, pct_impervious


# ===========================================================================
# 5. Visualisation
# ===========================================================================

def _build_colour_mask(cat_map: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB colour mask from a category map."""
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
    """
    Save a 3-panel PNG to data/visuals/{station_id}.png:
      Panel 1 — Original satellite chip
      Panel 2 — Colour-coded surface classification
      Panel 3 — Semi-transparent overlay on top of the original
    """
    colour_mask = _build_colour_mask(cat_map)

    # Build overlay by compositing RGBA mask over RGB original
    orig_rgba   = Image.fromarray(rgb).convert("RGBA")
    mask_rgba   = Image.fromarray(
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
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Station {station_id}   [{method}]",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out = VISUALS_DIR / f"{station_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 6. Per-station processing
# ===========================================================================

def process_station(
    tif_path: Path,
    save_visual_flag: bool,
    processor=None,
    model=None,
    device: str = "cpu",
    use_fallback: bool = False,
) -> Optional[dict]:
    """
    Process a single station TIF.  Returns a result dict or None on failure.
    """
    station_id = tif_path.stem

    try:
        rgb = read_tif_as_rgb(tif_path)

        if use_fallback:
            class_map = segment_with_exg(rgb)
            method = "ExG-fallback"
        else:
            class_map = segment_with_segformer(rgb, processor, model, device)
            method = "SegFormer-ADE20K"

        cat_map = make_category_map(class_map)
        pct_green, pct_impervious = compute_percentages(cat_map)

        if save_visual_flag:
            save_visual(rgb, cat_map, station_id, pct_green, pct_impervious, method)

        log.info(
            "  [OK] %s — Green: %5.1f%%  Impervious: %5.1f%%  Other: %5.1f%%",
            station_id, pct_green, pct_impervious,
            100.0 - pct_green - pct_impervious,
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
# 7. Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment satellite images into surface types.")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip saving overlay PNG files (faster).")
    parser.add_argument("--fallback", action="store_true",
                        help="Use ExG colour-index instead of the SegFormer model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_visuals = not args.no_visuals
    use_fallback = args.fallback

    # Discover TIF files
    tif_files = sorted(IMAGES_DIR.glob("*.tif"))
    if not tif_files:
        log.error(
            "No .tif files found in %s.\n"
            "Run `python src/fetch_satellite_images.py` first.",
            IMAGES_DIR,
        )
        sys.exit(1)

    print("=" * 62)
    print("  Houston Surface Segmentation")
    print(f"  Images      : {len(tif_files)} stations found in {IMAGES_DIR.name}/")
    print(f"  Model       : {'ExG colour index (fallback)' if use_fallback else HF_MODEL_ID}")
    print(f"  Visuals     : {'data/visuals/' if save_visuals else 'disabled'}")
    print(f"  Output CSV  : {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print("=" * 62)

    # Load model (unless using fallback)
    processor = model = device = None
    if not use_fallback:
        try:
            processor, model, device = load_segformer()
        except ImportError:
            log.warning(
                "`transformers` not installed — switching to ExG fallback.\n"
                "Install with:  pip install transformers"
            )
            use_fallback = True

    # Process all stations
    results: list[dict] = []
    failed:  list[str]  = []

    for tif_path in tqdm(tif_files, unit="station"):
        record = process_station(
            tif_path,
            save_visual_flag=save_visuals,
            processor=processor,
            model=model,
            device=device or "cpu",
            use_fallback=use_fallback,
        )
        if record:
            results.append(record)
        else:
            failed.append(tif_path.stem)

    # Save CSV
    if results:
        df = pd.DataFrame(results, columns=[
            "Station_ID", "Pct_Green", "Pct_Impervious", "Pct_Other", "Method",
        ])
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        print("\n" + "=" * 62)
        print(f"  Saved {len(df)} record(s) to {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
        print("\n  Summary statistics:")
        print(df[["Pct_Green", "Pct_Impervious", "Pct_Other"]].describe().to_string())

    # Final report
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

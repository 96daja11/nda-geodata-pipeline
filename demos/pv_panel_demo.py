"""
PV Panel thermal demo – uses real DJI thermal JPEGs from the
"Thermal PV Panel Detection Dataset for UAV Inspection" dataset.

Dataset location: data/datasets/pv-panel/
Usage:
    python3 demos/pv_panel_demo.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

# Ensure project root is on the path and is the working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pv_panel_demo")

# ── Dataset configuration ──────────────────────────────────────────────────
DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "pv-panel" / \
    "Thermal PV Panel Detection Dataset for UAV Inspection"

# Use train split – 235 images, take a representative 10-image subset
SPLIT = "train"
MAX_IMAGES = 10   # Keep the demo fast; increase for a fuller run

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "pv_panel"

# Geographic centre: Goleta / Santa Barbara CA (common DJI PV survey area)
# Falls back to Gothenburg for the report map if GPS is absent.
CENTER_LAT = 57.7089
CENTER_LON = 11.9746


def main() -> str:
    """Run the PV-panel pipeline demo. Returns path to the generated report."""

    # ── Check dataset is present ───────────────────────────────────────────
    images_dir = DATASET_ROOT / SPLIT / "images"
    if not images_dir.exists():
        logger.error(
            f"Dataset not found at: {images_dir}\n"
            "Expected extracted zip at: data/datasets/pv-panel/\n"
            "The dataset should already be extracted – check the directory."
        )
        sys.exit(1)

    # ── Collect images ─────────────────────────────────────────────────────
    all_images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG"))
    if not all_images:
        logger.error(f"No JPEG images found in {images_dir}")
        sys.exit(1)

    # Select a diverse subset (every Nth image)
    step = max(1, len(all_images) // MAX_IMAGES)
    selected = [str(p) for p in all_images[::step][:MAX_IMAGES]]
    logger.info(f"PV-panel dataset: {len(all_images)} images total, using {len(selected)}")

    # The DJI thermal JPEGs serve as both "thermal" and (placeholder) "RGB" images.
    # The pipeline mock-detector handles the RGB step with synthetic findings,
    # while thermal extraction runs on the real pseudo-colour pixel data.
    thermal_paths = selected
    rgb_paths = selected  # same images – demo-mode detector will generate mock RGB findings

    # ── Run pipeline ───────────────────────────────────────────────────────
    from demos.dataset_adapter import run_pipeline_on_images

    pdf_path = run_pipeline_on_images(
        rgb_paths=rgb_paths,
        thermal_paths=thermal_paths,
        output_dir=OUTPUT_DIR,
        dataset_name="pv_panel",
        inspection_date="2026-03-09",
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
    )

    print(f"\nRapport skapad: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()

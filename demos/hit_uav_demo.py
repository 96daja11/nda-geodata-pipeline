"""
HIT-UAV infrared thermal demo.

Dataset: HIT-UAV – A High-altitude Infrared Thermal Dataset for UAV-based Object Detection
Expected location after extraction: data/datasets/hit-uav/

Extract with:
    cd data/datasets/hit-uav && unzip HIT-UAV.zip

HIT-UAV contains infrared images of outdoor scenes (persons, cars, bicycles)
captured at multiple altitudes. We treat all infrared images as the thermal
channel and run the SmartTek pipeline for structural anomaly detection.

Usage:
    python3 demos/hit_uav_demo.py
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hit_uav_demo")

DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "hit-uav"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "outputs" / "hit_uav"
MAX_IMAGES   = 10

# Harbin, China (where HIT-UAV was collected)
CENTER_LAT = 45.7722
CENTER_LON = 126.6369


def _find_images(root: Path) -> list[Path]:
    images = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        images.extend(root.rglob(ext))
    return sorted(images)


def _check_extracted() -> Path:
    zip_path = DATASET_ROOT / "HIT-UAV.zip"
    images = [p for p in _find_images(DATASET_ROOT)
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if images:
        return DATASET_ROOT

    if zip_path.exists():
        print(
            "\n[HIT-UAV demo] Dataset ZIP found but not extracted.\n"
            "Run the following commands to extract:\n\n"
            f"    cd {DATASET_ROOT}\n"
            f"    unzip HIT-UAV.zip\n\n"
            "Then re-run this demo."
        )
    else:
        print(
            "\n[HIT-UAV demo] Dataset not downloaded.\n"
            "Place HIT-UAV.zip at:\n"
            f"    {zip_path}\n"
            "Then extract and re-run."
        )
    sys.exit(0)


def main() -> str:
    root = _check_extracted()
    all_images = [p for p in _find_images(root)
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    if not all_images:
        logger.error(f"No images found under {root}")
        sys.exit(1)

    # Prefer images from a single consistent sub-folder for a coherent demo
    # (HIT-UAV is organized by altitude/scene – pick first subfolder with images)
    subfolders = sorted({p.parent for p in all_images})
    chosen_folder_images = [p for p in all_images if p.parent == subfolders[0]]
    if len(chosen_folder_images) < 3:
        chosen_folder_images = all_images  # fallback to all

    step = max(1, len(chosen_folder_images) // MAX_IMAGES)
    selected = [str(p) for p in chosen_folder_images[::step][:MAX_IMAGES]]
    logger.info(f"HIT-UAV dataset: {len(all_images)} images total, using {len(selected)}")

    from demos.dataset_adapter import run_pipeline_on_images

    pdf_path = run_pipeline_on_images(
        rgb_paths=selected,
        thermal_paths=selected,
        output_dir=OUTPUT_DIR,
        dataset_name="hit_uav",
        inspection_date="2026-03-09",
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
    )

    print(f"\nRapport skapad: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()

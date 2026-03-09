"""
UAVID3D dataset demo – UAV thermal building inspection.

Dataset: UAVID3D – UAV-based thermal building inspection
Location: data/datasets/uavid3d/

Extracted datasets used:
  Blume_drone_data_capture_may2021/thermal/1_initial/project_data/normalised/
      DJI_0001.jpg – DJI_0131.jpg  (129 normalised thermal JPEGs)

  Olympic_club_drone_data_capture_may2021/thermal_images/
      Project 00016_inputs/  – 90 DJI thermal JPEGs
      Project 00017_inputs/  – 90 DJI thermal JPEGs
      Project 00018_inputs/  – 30 DJI thermal JPEGs
      Project 00019_inputs/  – 35 DJI thermal JPEGs

Images from both building sites are combined for the demo run.
The final selection of MAX_IMAGES images is drawn evenly from both sources.

Usage:
    python3 demos/uavid3d_demo.py
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
logger = logging.getLogger("uavid3d_demo")

DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "uavid3d"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "outputs" / "uavid3d"
MAX_IMAGES   = 15

# Blume commercial building survey location (Bochum / NRW area, Germany)
CENTER_LAT = 51.4818
CENTER_LON = 7.2162

# Known path to Blume normalised thermal images
BLUME_THERMAL_DIR = (
    DATASET_ROOT
    / "Blume_drone_data_capture_may2021"
    / "thermal"
    / "1_initial"
    / "project_data"
    / "normalised"
)

# Olympic club thermal images (4 projects extracted from ZIP)
OLYMPIC_THERMAL_DIR = (
    DATASET_ROOT
    / "Olympic_club_drone_data_capture_may2021"
    / "thermal_images"
)


def _collect_olympic_images() -> list[Path]:
    """
    Collect DJI thermal JPEGs from extracted Olympic project folders.

    Each project zip extracted to a subfolder like:
        thermal_images/Project 00016_inputs/Project 00016_inputs/DJI_*.jpg
    """
    images: list[Path] = []
    if not OLYMPIC_THERMAL_DIR.exists():
        return images

    for project_dir in sorted(OLYMPIC_THERMAL_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        # Images may be directly inside or one level deeper (same name subdir)
        found = sorted(project_dir.rglob("DJI_*.jpg"))
        if found:
            logger.info(
                f"  Olympic {project_dir.name}: {len(found)} thermal images"
            )
            images.extend(found)

    return images


def _collect_thermal_images() -> list[Path]:
    """
    Collect thermal images from all available extracted UAVID3D datasets.
    Returns images from Blume (primary) and Olympic (if extracted).
    """
    images: list[Path] = []

    # 1. Blume normalised thermal images
    if BLUME_THERMAL_DIR.exists():
        blume_imgs = sorted(BLUME_THERMAL_DIR.glob("DJI_*.jpg"))
        if blume_imgs:
            logger.info(f"  Blume dataset: {len(blume_imgs)} thermal images")
            images.extend(blume_imgs)

    # 2. Olympic club thermal images
    olympic_imgs = _collect_olympic_images()
    images.extend(olympic_imgs)

    return images


def _check_dataset() -> list[Path]:
    """Return thermal images, or print instructions and exit."""
    if not DATASET_ROOT.exists():
        print(
            "\n[UAVID3D demo] Dataset directory not found:\n"
            f"    {DATASET_ROOT}\n"
        )
        sys.exit(0)

    images = _collect_thermal_images()
    if images:
        return images

    # No images found
    print(
        "\n[UAVID3D demo] No images found.\n"
        "Expected Blume thermal images at:\n"
        f"    {BLUME_THERMAL_DIR}\n"
        "or Olympic images at:\n"
        f"    {OLYMPIC_THERMAL_DIR}\n"
    )
    sys.exit(0)


def main() -> str:
    all_thermal = _check_dataset()

    logger.info(f"UAVID3D: {len(all_thermal)} thermal images found in total")

    # Select MAX_IMAGES evenly-spaced images across all sources
    step = max(1, len(all_thermal) // MAX_IMAGES)
    selected = all_thermal[::step][:MAX_IMAGES]
    logger.info(f"  Selected {len(selected)} images (every {step}th)")

    thermal_paths = [str(p) for p in selected]

    # UAVID3D only provides thermal imagery; use it for both channels.
    # The pipeline's mock detector will generate RGB findings.
    from demos.dataset_adapter import run_pipeline_on_images

    pdf_path = run_pipeline_on_images(
        rgb_paths=thermal_paths,
        thermal_paths=thermal_paths,
        output_dir=OUTPUT_DIR,
        dataset_name="uavid3d",
        inspection_date="2026-03-09",
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
    )

    print(f"\nRapport skapad: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()

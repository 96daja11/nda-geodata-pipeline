"""
UAVID3D dataset demo.

Dataset: UAVID3D – UAV-based thermal building inspection
Location: data/datasets/uavid3d/

Extracted datasets:
  Blume_drone_data_capture_may2021/thermal/1_initial/project_data/normalised/
      DJI_0001.jpg – DJI_0131.jpg  (129 normalised thermal JPEGs)

Also checks for Olympic_004 if extracted:
  Olympic_004/... (if present)

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

# Blume commercial building survey location (Bochum/NRW area, Germany)
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


def _find_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> list[Path]:
    imgs = []
    for ext in exts:
        imgs.extend(root.rglob(f"*{ext}"))
        imgs.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def _collect_thermal_images() -> list[Path]:
    """
    Collect thermal images from all available extracted UAVID3D datasets.
    Prioritises the known normalised directory; also discovers any other
    extracted sub-datasets.
    """
    images: list[Path] = []

    # 1. Blume normalised thermal images (primary)
    if BLUME_THERMAL_DIR.exists():
        blume_imgs = sorted(BLUME_THERMAL_DIR.glob("DJI_*.jpg"))
        if blume_imgs:
            logger.info(f"  Blume dataset: {len(blume_imgs)} thermal images")
            images.extend(blume_imgs)

    # 2. Look for any other extracted datasets (Olympic_004, etc.)
    for child in sorted(DATASET_ROOT.iterdir()):
        if child.is_dir() and child != BLUME_THERMAL_DIR.parents[4]:
            # Skip the Blume top-level dir (already covered above)
            if "blume" in child.name.lower():
                continue
            extra_imgs = _find_images(child)
            extra_imgs = [p for p in extra_imgs
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
            if extra_imgs:
                logger.info(f"  Additional dataset '{child.name}': {len(extra_imgs)} images")
                images.extend(extra_imgs)

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

    # No images found – check for ZIPs
    zips = list(DATASET_ROOT.glob("*.zip"))
    if zips:
        cmds = "\n".join(f"    unzip {z.name}" for z in sorted(zips))
        print(
            "\n[UAVID3D demo] Dataset ZIPs found but not extracted.\n"
            "Run:\n\n"
            f"    cd {DATASET_ROOT}\n"
            f"{cmds}\n\n"
            "Then re-run this demo."
        )
    else:
        print(
            "\n[UAVID3D demo] No images or ZIPs found.\n"
            "Place Blume_004.zip (and/or Olympic_004.zip) at:\n"
            f"    {DATASET_ROOT}/\n"
            "Then extract and re-run."
        )
    sys.exit(0)


def main() -> str:
    all_thermal = _check_dataset()

    logger.info(f"UAVID3D: {len(all_thermal)} thermal images found in total")

    # Select 15 evenly-spaced images
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

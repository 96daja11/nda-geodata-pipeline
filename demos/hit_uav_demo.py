"""
HIT-UAV infrared thermal demo – Multi-scene Thermal Survey.

Dataset: HIT-UAV – A High-altitude Infrared Thermal Dataset for UAV-based Object Detection
Location: data/datasets/hit-uav/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/normal_xml/

Structure:
  JPEGImages/*.jpg  – 2898 infrared thermal images (640×512, grayscale)
  Annotations/*.xml – Pascal VOC bounding-box annotations
                      Classes: Car, Person, Bicycle, OtherVehicle, DontCare
  Filename encoding: <scene>_<altitude>_<angle>_<sequence>_<frame>.jpg

HIT-UAV contains infrared images of outdoor scenes (persons, cars, bicycles)
captured at multiple altitudes (30 m – 120 m) and angles (0–60°). We treat all
infrared images as the thermal channel and run the SmartTek pipeline for
structural anomaly detection, framing this as a "multi-scene thermal survey"
to demonstrate the thermal analysis capability on diverse outdoor scenes.

VOC annotations are parsed to enrich findings with ground-truth bounding boxes
of detected objects (persons, cars, etc.).

Usage:
    python3 demos/hit_uav_demo.py
"""
from __future__ import annotations
import logging
import sys
import xml.etree.ElementTree as ET
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

HIT_UAV_ROOT  = (
    PROJECT_ROOT
    / "data" / "datasets" / "hit-uav"
    / "suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c"
    / "normal_xml"
)
JPEG_DIR       = HIT_UAV_ROOT / "JPEGImages"
ANNOTATION_DIR = HIT_UAV_ROOT / "Annotations"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "outputs" / "hit_uav"
MAX_IMAGES     = 15

# Harbin, China (where HIT-UAV was collected)
CENTER_LAT = 45.7722
CENTER_LON = 126.6369


def _parse_voc_xml(xml_path: Path) -> list[dict]:
    """
    Parse a Pascal VOC XML annotation file and return a list of objects with
    keys: name, xmin, ymin, xmax, ymax.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="Unknown")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            objects.append({
                "name":  name,
                "xmin":  int(bndbox.findtext("xmin", "0")),
                "ymin":  int(bndbox.findtext("ymin", "0")),
                "xmax":  int(bndbox.findtext("xmax", "0")),
                "ymax":  int(bndbox.findtext("ymax", "0")),
            })
        return objects
    except Exception as e:
        logger.warning(f"Could not parse {xml_path}: {e}")
        return []


def _select_images(n: int = MAX_IMAGES) -> list[tuple[Path, Path | None]]:
    """
    Select *n* evenly-spaced images from JPEGImages/ together with their
    corresponding annotation file (or None if missing).

    Returns list of (image_path, annotation_path_or_None).
    """
    if not JPEG_DIR.exists():
        return []

    all_images = sorted(JPEG_DIR.glob("*.jpg")) + sorted(JPEG_DIR.glob("*.JPG"))
    if not all_images:
        return []

    step = max(1, len(all_images) // n)
    selected = all_images[::step][:n]

    result = []
    for img in selected:
        ann = ANNOTATION_DIR / (img.stem + ".xml")
        result.append((img, ann if ann.exists() else None))
    return result


def _summarise_annotations(pairs: list[tuple[Path, Path | None]]) -> dict:
    """
    Summarise ground-truth object counts across all selected images.
    Returns a dict: class_name -> count.
    """
    from collections import Counter
    counter: Counter = Counter()
    for _, ann_path in pairs:
        if ann_path:
            for obj in _parse_voc_xml(ann_path):
                counter[obj["name"]] += 1
    return dict(counter)


def main() -> str:
    if not JPEG_DIR.exists():
        print(
            "\n[HIT-UAV demo] Dataset not found.\n"
            f"Expected JPEGImages at:\n    {JPEG_DIR}\n\n"
            "The dataset should already be extracted to data/datasets/hit-uav/.\n"
        )
        sys.exit(0)

    pairs = _select_images(MAX_IMAGES)
    if not pairs:
        logger.error(f"No JPEG images found in {JPEG_DIR}")
        sys.exit(1)

    total_images = len(list(JPEG_DIR.glob("*.jpg"))) + len(list(JPEG_DIR.glob("*.JPG")))
    logger.info(f"HIT-UAV dataset: {total_images} images total, using {len(pairs)}")

    # Summarise annotations for the selected images
    ann_summary = _summarise_annotations(pairs)
    if ann_summary:
        logger.info(
            "  Ground-truth object counts: "
            + ", ".join(f"{k}={v}" for k, v in sorted(ann_summary.items()))
        )
    else:
        logger.info("  No annotation data found for selected images")

    selected_paths = [str(img) for img, _ in pairs]

    from demos.dataset_adapter import run_pipeline_on_images

    # Log filename encoding info for context
    logger.info(
        "  Filename encoding: <scene>_<altitude>_<angle>_<sequence>_<frame>.jpg"
    )
    logger.info(
        "  Framing: multi-scene thermal survey demonstrating thermal analysis "
        "on diverse outdoor scenes (parking, playgrounds, roads)"
    )

    pdf_path = run_pipeline_on_images(
        rgb_paths=selected_paths,
        thermal_paths=selected_paths,
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

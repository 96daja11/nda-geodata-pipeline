"""
TASeg thermal building demo.

Dataset: TASeg – Thermal Aerial Segmentation
Location: data/datasets/taseg/manual_set/

Structure:
  image/{train,val,test}/  – *.npy.lz4 files (raw temperature arrays)
  preview/{train,val,test}/ – *.png files (visual previews of thermal data)
  label/{train,val,test}/  – *.png files (segmentation masks, anomaly labels)

If lz4 is installed: loads raw temperature arrays for accurate anomaly detection.
Falls back to preview PNGs (normalized to temperature range) if lz4 is unavailable.
Label PNGs are used to locate ground-truth anomaly regions.

Usage:
    python3 demos/taseg_demo.py
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
logger = logging.getLogger("taseg_demo")

DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "taseg" / "manual_set"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "outputs" / "taseg"
MAX_IMAGES   = 15

# TASeg dataset – European building thermal surveys
CENTER_LAT = 51.5074
CENTER_LON = 7.4653


def _check_lz4() -> bool:
    """Return True if lz4 is importable."""
    try:
        import lz4.frame  # noqa: F401
        return True
    except ImportError:
        return False


def _load_npy_lz4(path: Path):
    """Decompress an .npy.lz4 file and return a numpy float32 array."""
    import lz4.frame
    import numpy as np
    raw = lz4.frame.decompress(path.read_bytes())
    return np.frombuffer(raw, dtype=np.float32)


def _find_preview_images(root: Path) -> list[Path]:
    """Return sorted list of PNG preview images across all splits."""
    images = []
    for split in ("train", "val", "test"):
        split_dir = root / "preview" / split
        if split_dir.exists():
            images.extend(sorted(split_dir.glob("*.png")))
    return images


def _find_npy_lz4_files(root: Path) -> list[Path]:
    """Return sorted list of .npy.lz4 temperature array files across all splits."""
    files = []
    for split in ("train", "val", "test"):
        split_dir = root / "image" / split
        if split_dir.exists():
            files.extend(sorted(split_dir.glob("*.npy.lz4")))
    return files


def _find_label_for(preview_path: Path, dataset_root: Path) -> Path | None:
    """
    Given a preview PNG path, find the corresponding label PNG.
    Preview: dataset_root/preview/{split}/{stem}.png
    Label:   dataset_root/label/{split}/{stem}.png
    """
    # Extract split from grandparent directory
    split = preview_path.parent.name
    label_path = dataset_root / "label" / split / preview_path.name
    return label_path if label_path.exists() else None


def _parse_label_anomalies(label_path: Path) -> list[dict]:
    """
    Parse a TASeg label PNG to find anomaly regions.
    Labels encode two classes:
      background  – dark / black pixels
      thermal     – bright / non-black pixels (anomaly regions)

    Returns a list of anomaly bounding boxes (same format as extract_anomalies).
    """
    try:
        import numpy as np
        from PIL import Image

        label_img = Image.open(label_path).convert("L")
        mask = np.array(label_img, dtype=np.uint8)

        # Anomaly = non-background pixels (threshold > 10 to avoid JPEG artefacts)
        anomaly_mask = mask > 10
        if not anomaly_mask.any():
            return []

        try:
            from scipy import ndimage
            labeled, n = ndimage.label(anomaly_mask)
            regions = range(1, n + 1)
        except ImportError:
            labeled = anomaly_mask.astype(int)
            regions = [1]

        results = []
        for i in regions:
            region = labeled == i
            area = int(region.sum())
            if area < 30:
                continue
            rows, cols = region.nonzero()
            results.append({
                "bbox": [int(cols.min()), int(rows.min()),
                         int(cols.max()), int(rows.max())],
                "center_x": float(cols.mean()),
                "center_y": float(rows.mean()),
                "area_px": area,
                "source": "label_gt",
                "anomaly_type": "thermal_anomaly",
            })
        return results
    except Exception as e:
        logger.warning(f"Could not parse label {label_path}: {e}")
        return []


def _check_dataset() -> Path:
    """Verify the dataset is extracted and return the root path."""
    if not DATASET_ROOT.exists():
        zip_path = DATASET_ROOT.parent / "manual_set.zip"
        if zip_path.exists():
            print(
                "\n[TASeg demo] Dataset ZIP found but not extracted.\n"
                "Run the following to extract:\n\n"
                f"    cd {DATASET_ROOT.parent}\n"
                f"    unzip manual_set.zip\n\n"
                "Then re-run this demo."
            )
        else:
            print(
                "\n[TASeg demo] Dataset not found at:\n"
                f"    {DATASET_ROOT}\n"
                "Download the TASeg manual_set.zip and extract it there."
            )
        sys.exit(0)

    # Verify we can find images
    previews = _find_preview_images(DATASET_ROOT)
    if not previews:
        print(f"\n[TASeg demo] No preview images found under {DATASET_ROOT}/preview/")
        sys.exit(1)

    return DATASET_ROOT


def main() -> str:
    root = _check_dataset()
    has_lz4 = _check_lz4()

    preview_images = _find_preview_images(root)
    npy_files = _find_npy_lz4_files(root) if has_lz4 else []

    logger.info(f"TASeg dataset: {len(preview_images)} preview images found")
    if has_lz4:
        logger.info(f"  lz4 available: {len(npy_files)} raw temperature arrays")
    else:
        logger.info("  lz4 not installed – using preview PNGs (normalized to temperature)")
        logger.info("  Install lz4 for raw temperature data: pip install lz4")

    # Select evenly-spaced images
    step = max(1, len(preview_images) // MAX_IMAGES)
    selected_previews = preview_images[::step][:MAX_IMAGES]

    # Find corresponding labels and log ground-truth anomaly info
    total_gt_anomalies = 0
    for prev in selected_previews:
        label = _find_label_for(prev, root)
        if label:
            gt = _parse_label_anomalies(label)
            total_gt_anomalies += len(gt)

    logger.info(
        f"Using {len(selected_previews)} images "
        f"(ground-truth anomaly regions in labels: {total_gt_anomalies})"
    )

    # Use preview PNGs as both RGB and thermal inputs to the pipeline.
    # The pipeline's ThermalExtractor will normalize pixel values to temperature.
    selected_paths = [str(p) for p in selected_previews]

    from demos.dataset_adapter import run_pipeline_on_images

    pdf_path = run_pipeline_on_images(
        rgb_paths=selected_paths,
        thermal_paths=selected_paths,
        output_dir=OUTPUT_DIR,
        dataset_name="taseg",
        inspection_date="2026-03-09",
        center_lat=CENTER_LAT,
        center_lon=CENTER_LON,
    )

    print(f"\nRapport skapad: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()

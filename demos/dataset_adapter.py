"""
Shared utility module for dataset demo adapters.

Provides helpers to bridge external datasets into the SmartTek pipeline format.
"""
from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logger = logging.getLogger(__name__)


def load_thermal_from_jpeg(path: str | Path) -> np.ndarray:
    """
    Load a thermal JPEG (DJI pseudo-color or grayscale) and return
    a 2-D numpy array of pseudo-temperatures in °C.

    Mapping: pixel luminance 0-255  →  15-85 °C (realistic PV-panel range).
    """
    from PIL import Image

    img = Image.open(path)

    # If RGB pseudo-colour thermal (e.g. iron/inferno LUT), convert to grayscale
    # luminance preserves relative "heat" ordering of most thermal palettes.
    if img.mode != "L":
        img = img.convert("L")

    pixel = np.array(img, dtype=np.float32)
    # Map [0, 255] → [15, 85] °C
    temp = 15.0 + (pixel / 255.0) * 70.0
    return temp


def extract_anomalies(
    temp_array: np.ndarray,
    threshold: float = 10.0,
    min_area_px: int = 50,
) -> list[dict]:
    """
    Detect contiguous regions whose temperature deviates by more than
    *threshold* degrees from the image mean.

    Returns a list of dicts with keys:
        bbox, center_x, center_y, max_temp, mean_temp, delta_temp,
        area_px, anomaly_type
    """
    try:
        from scipy import ndimage
        has_scipy = True
    except ImportError:
        has_scipy = False

    mean_t = float(np.mean(temp_array))
    anomalies = []

    for atype, mask in [
        ("hotspot",     temp_array > mean_t + threshold),
        ("cold_bridge", temp_array < mean_t - threshold),
    ]:
        if not np.any(mask):
            continue

        if has_scipy:
            labeled, n = ndimage.label(mask)
            regions = range(1, n + 1)
        else:
            # Treat the whole mask as one region (no scipy)
            labeled = mask.astype(int)
            regions = [1]

        for i in regions:
            region = labeled == i
            area = int(np.sum(region))
            if area < min_area_px:
                continue

            rows, cols = np.where(region)
            region_temps = temp_array[region]
            anomalies.append({
                "bbox": [int(cols.min()), int(rows.min()),
                         int(cols.max()), int(rows.max())],
                "center_x": float(np.mean(cols)),
                "center_y": float(np.mean(rows)),
                "max_temp":  float(region_temps.max()),
                "mean_temp": float(region_temps.mean()),
                "delta_temp": abs(float(region_temps.mean()) - mean_t),
                "area_px":   area,
                "anomaly_type": atype,
            })

    return anomalies


def mock_gps_coords(
    n: int,
    center_lat: float = 57.7089,
    center_lon: float = 11.9746,
    spread: float = 0.005,
) -> list[tuple[float, float]]:
    """
    Generate *n* pseudo-random GPS coordinates near Gothenburg.

    Returns list of (lat, lon) tuples.
    """
    rng = np.random.default_rng(seed=42)
    lats = center_lat + rng.uniform(-spread, spread, n)
    lons = center_lon + rng.uniform(-spread, spread, n)
    return [(round(float(la), 6), round(float(lo), 6)) for la, lo in zip(lats, lons)]


def run_pipeline_on_images(
    rgb_paths: list[str],
    thermal_paths: list[str],
    output_dir: Path | str,
    dataset_name: str,
    inspection_date: str = "2026-03-09",
    center_lat: float = 57.7089,
    center_lon: float = 11.9746,
) -> str:
    """
    Run SmartTek pipeline steps 1-6 on a set of already-validated image paths.

    *rgb_paths*     – list of absolute paths to RGB/visual images
    *thermal_paths* – list of absolute paths to thermal images
    *output_dir*    – base directory for all outputs
    *dataset_name*  – short identifier used as run_id (e.g. "pv_panel")

    Returns the path to the generated PDF (or HTML fallback).
    """
    from pipeline.ingest.models import IngestResult, ImageMetadata
    from pipeline.photogrammetry.odm_client import ODMClient, PhotogrammetryResult
    from pipeline.thermal.extractor import ThermalExtractor
    from pipeline.detection.detector import Detector
    from pipeline.analysis.analyzer import Analyzer
    from pipeline.report.generator import ReportGenerator

    output_dir = Path(output_dir)
    run_id = dataset_name
    t0 = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info(f"  DATASET: {dataset_name}")
    logger.info("=" * 60)

    # ── Step 1: Build IngestResult manually (images already validated) ──
    logger.info("STEG 1: Ingest")
    from PIL import Image as PILImage

    def _make_meta(p: str, ftype: str) -> Optional[ImageMetadata]:
        try:
            path = Path(p)
            stat = path.stat()
            img = PILImage.open(p)
            w, h = img.size
            return ImageMetadata(
                path=p,
                filename=path.name,
                file_type=ftype,
                format=img.format or path.suffix.upper().lstrip("."),
                width=w,
                height=h,
                gps=None,
                file_size_bytes=stat.st_size,
            )
        except Exception as e:
            logger.warning(f"Could not read {p}: {e}")
            return None

    rgb_meta    = [m for p in rgb_paths    if (m := _make_meta(p, "rgb"))     is not None]
    thermal_meta = [m for p in thermal_paths if (m := _make_meta(p, "thermal")) is not None]

    ingest_result = IngestResult(
        run_id=run_id,
        rgb_images=rgb_meta,
        thermal_images=thermal_meta,
        demo_mode=True,
    )
    logger.info(f"  RGB: {len(rgb_meta)}, Thermal: {len(thermal_meta)}")

    # ── Step 2: Photogrammetry (demo mode, synthetic orthophoto) ─────
    logger.info("STEG 2: Fotogrammetri (demo mode)")
    spread = 0.005
    odm = ODMClient(demo_mode=True)
    photogrammetry_result = odm.process(
        rgb_images=[m.path for m in rgb_meta],
        output_dir=output_dir / "photogrammetry",
        run_id=run_id,
    )
    # Override bbox/center with dataset location
    photogrammetry_result.center_lat = center_lat
    photogrammetry_result.center_lon = center_lon
    photogrammetry_result.bbox = [
        center_lon - spread, center_lat - spread,
        center_lon + spread, center_lat + spread,
    ]

    # ── Step 3: Thermal extraction ────────────────────────────────────
    logger.info("STEG 3: Termisk extraktion")
    extractor = ThermalExtractor(
        anomaly_threshold_c=8.0,   # wider threshold – PV images have high contrast
        min_area_px=30,
        demo_mode=True,
    )
    thermal_result = extractor.process(
        thermal_images=[m.path for m in thermal_meta],
        output_dir=output_dir / "thermal",
        run_id=run_id,
    )
    logger.info(f"  Anomalies: {len(thermal_result.anomalies)}")

    # ── Step 4: Detection ─────────────────────────────────────────────
    logger.info("STEG 4: AI-detektering (mock)")
    detector = Detector(confidence=0.4, demo_mode=True)
    detection_result = detector.process(
        rgb_images=[m.path for m in rgb_meta] if rgb_meta else [m.path for m in thermal_meta],
        thermal_anomalies=thermal_result.anomalies,
        output_dir=output_dir / "detection",
        run_id=run_id,
    )
    logger.info(f"  Findings: {len(detection_result.findings)}")

    # ── Step 5: Analysis ──────────────────────────────────────────────
    logger.info("STEG 5: GIS-analys")
    analyzer = Analyzer(demo_mode=True)
    analysis_result = analyzer.process(
        detection_result=detection_result,
        photogrammetry_result=photogrammetry_result,
        output_dir=output_dir / "analysis",
        run_id=run_id,
        inspection_date=inspection_date,
    )
    s = analysis_result.summary
    logger.info(
        f"  Total={s.total_findings}  "
        f"KRITISK={s.kritisk_count}  HÖG={s.hog_count}  "
        f"MEDEL={s.medel_count}  LÅG={s.lag_count}"
    )

    # ── Step 6: Report ────────────────────────────────────────────────
    logger.info("STEG 6: Rapport")
    generator = ReportGenerator(company_name="SmartTek AB", demo_mode=True)
    report_result = generator.generate(
        analysis_result=analysis_result,
        detection_result=detection_result,
        thermal_result=thermal_result,
        ingest_result=ingest_result,
        output_dir=output_dir / "report",
        run_id=run_id,
    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  KLAR ({elapsed:.1f}s)")
    logger.info(f"  PDF:  {report_result.pdf_path}")
    logger.info(f"  HTML: {report_result.html_path}")
    logger.info("=" * 60)

    return report_result.pdf_path

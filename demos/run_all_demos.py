"""
Master demo runner – discovers which datasets are ready and runs them all.

Usage:
    python3 demos/run_all_demos.py
"""
from __future__ import annotations
import importlib
import logging
import sys
import time
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
logger = logging.getLogger("run_all_demos")

DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"

# Each entry: (module_name, dataset_subdir, description, image_extensions)
DEMOS = [
    (
        "demos.pv_panel_demo",
        "pv-panel/Thermal PV Panel Detection Dataset for UAV Inspection/train/images",
        "PV Panel – DJI Thermal (train split)",
        {".jpg", ".jpeg"},
    ),
    (
        "demos.taseg_demo",
        "taseg",
        "TASeg – Thermal Aerial Building Segmentation",
        {".jpg", ".jpeg", ".png"},
    ),
    (
        "demos.hit_uav_demo",
        "hit-uav",
        "HIT-UAV – High-altitude Infrared Thermal",
        {".jpg", ".jpeg", ".png"},
    ),
    (
        "demos.uavid3d_demo",
        "uavid3d",
        "UAVID3D – UAV Urban 3D (RGB + optional thermal)",
        {".jpg", ".jpeg", ".png", ".tif", ".tiff"},
    ),
]


def _count_images(subdir: str, exts: set[str]) -> int:
    """Count extractable image files in *subdir* under DATASETS_ROOT."""
    target = DATASETS_ROOT / subdir
    if not target.exists():
        return 0
    count = 0
    for p in target.rglob("*"):
        if p.suffix.lower() in exts:
            count += 1
    return count


def _dataset_status() -> list[dict]:
    statuses = []
    for module, subdir, desc, exts in DEMOS:
        n = _count_images(subdir, exts)
        statuses.append({
            "module": module,
            "subdir": subdir,
            "description": desc,
            "image_count": n,
            "ready": n > 0,
        })
    return statuses


def main() -> None:
    print("\n" + "=" * 70)
    print("  SmartTek Demo Runner – Dataset Status")
    print("=" * 70)

    statuses = _dataset_status()
    ready = [s for s in statuses if s["ready"]]
    not_ready = [s for s in statuses if not s["ready"]]

    print("\nDatasets READY to run:")
    if ready:
        for s in ready:
            print(f"  [OK]  {s['description']}  ({s['image_count']} images)")
    else:
        print("  (none – no datasets are extracted yet)")

    if not_ready:
        print("\nDatasets NOT ready (zip not extracted or not downloaded):")
        for s in not_ready:
            print(f"  [--]  {s['description']}  →  data/datasets/{s['subdir']}/")

    if not ready:
        print(
            "\nNo datasets are ready. Extract at least one dataset and re-run.\n"
            "See the individual demo scripts for extraction instructions.\n"
        )
        sys.exit(0)

    print(f"\nRunning {len(ready)} demo(s)...\n")

    results: list[tuple[str, str, float]] = []  # (description, pdf_path, elapsed)

    for s in ready:
        print("=" * 70)
        print(f"  Starting: {s['description']}")
        print("=" * 70)
        t0 = time.time()
        try:
            mod = importlib.import_module(s["module"])
            pdf_path = mod.main()
            elapsed = time.time() - t0
            results.append((s["description"], pdf_path, elapsed))
        except SystemExit:
            # Demo called sys.exit(0) because dataset became unavailable
            logger.warning(f"Demo {s['module']} exited early – skipping.")
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"Demo {s['module']} failed after {elapsed:.1f}s: {exc}")
            results.append((s["description"], f"ERROR: {exc}", elapsed))

    print("\n" + "=" * 70)
    print("  Summary of generated reports")
    print("=" * 70)
    for desc, path, elapsed in results:
        status = "OK  " if not str(path).startswith("ERROR") else "FAIL"
        print(f"  [{status}]  {desc}")
        print(f"         {path}  ({elapsed:.1f}s)")
    print()


if __name__ == "__main__":
    main()

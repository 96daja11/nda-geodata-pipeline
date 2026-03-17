#!/usr/bin/env python3
"""
Steg 3 — Detektera terränganomalier och fördjupningar
Använder lokal avvikelseanalys (DTM vs lokal medelnivå) för att hitta
gropar, höjder, stigar och andra terrängfeature.
GPU-accelererad med CuPy.
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    cp.array([1])
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"🚀 GPU: {GPU_NAME} | {_dev.mem_info[1]//1024**2} MB VRAM")
except (ImportError, Exception) as _e:
    import numpy as cp
    import scipy.ndimage as cpnd
    GPU_AVAILABLE = False
    GPU_NAME = "CPU fallback"
    print(f"⚠️  GPU: Not available ({_e})")

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import uniform_filter, label as scipy_label
from skimage import measure
from tqdm import tqdm

print("=" * 60)
print("STEG 3 — DETEKTERA TERRÄNGANOMALIER")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
RAW      = BASE / "data" / "raw"
OUT_FIGS = BASE / "outputs" / "figures"
OUT_REPS = BASE / "outputs" / "reports"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPS.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"

# ── Load DTM ──────────────────────────────────────────────────────────────────
print("\nLaddar DTM...", end=" ", flush=True)
with rasterio.open(RAW / "dtm.tif") as src:
    dtm     = src.read(1).astype(np.float32)
    nodata  = src.nodata
    pixel_m = src.res[0]
    if nodata is not None:
        dtm[dtm == nodata] = np.nan
print(f"OK  ({dtm.shape[1]}×{dtm.shape[0]} px, {pixel_m:.4f} m/px)")

pixel_area = pixel_m ** 2
dtm_filled = np.where(np.isnan(dtm), np.nanmedian(dtm), dtm)

# ── GPU-accelerated local anomaly detection ───────────────────────────────────
# Method: compare each pixel against its local neighborhood mean
# Depressions: dtm << local_mean (negative deviation)
# Elevations:  dtm >> local_mean (positive deviation)

RADII_M = [5.0, 10.0, 20.0]   # meters — multi-scale analysis
MIN_AREA_M2 = 5.0              # minimum anomaly area to report

all_deviations = {}

print("\nBeräknar lokal avvikelse på GPU (multi-skala)...")
for radius_m in RADII_M:
    radius_px = max(3, int(radius_m / pixel_m))
    size_px   = radius_px * 2 + 1
    label_str = f"{radius_m:.1f}m"
    print(f"  Radius {label_str} ({size_px}×{size_px} px)...", end=" ", flush=True)

    if GPU_AVAILABLE:
        dtm_g    = cp.array(dtm_filled.astype(np.float32))
        # GPU uniform filter for local mean
        local_g  = cpnd.uniform_filter(dtm_g, size=size_px)
        dev_g    = dtm_g - local_g   # positive = higher than surroundings
        dev_cpu  = cp.asnumpy(dev_g)
    else:
        local    = uniform_filter(dtm_filled.astype(np.float64), size=size_px)
        dev_cpu  = (dtm_filled - local).astype(np.float32)

    dev_cpu[np.isnan(dtm)] = 0
    all_deviations[label_str] = dev_cpu
    print(f"OK  (range {dev_cpu.min():.3f} – {dev_cpu.max():.3f} m)")

# Use 5m scale as primary
primary_dev = all_deviations["5.0m"]

# Identify depressions (below surroundings) and elevations (above surroundings)
threshold_low  = -0.50   # 25 cm below surroundings
threshold_high = +0.60   # 30 cm above surroundings

depression_mask = primary_dev < threshold_low
elevation_mask  = primary_dev > threshold_high
depression_mask[np.isnan(dtm)] = False
elevation_mask[np.isnan(dtm)]  = False

print(f"\n  Fördjupningar (<{threshold_low}m): "
      f"{depression_mask.sum():,} pixlar "
      f"({100*depression_mask.sum()/depression_mask.size:.1f}%  =  "
      f"{depression_mask.sum()*pixel_area:.0f} m²)")
print(f"  Upphöjningar (>{threshold_high}m): "
      f"{elevation_mask.sum():,} pixlar "
      f"({100*elevation_mask.sum()/elevation_mask.size:.1f}%  =  "
      f"{elevation_mask.sum()*pixel_area:.0f} m²)")

# ── Label and filter regions ───────────────────────────────────────────────────
print("\nEtiketterar och filtrerar regioner...")

def label_regions(mask, intensity_map, min_area_m2, pixel_area):
    labeled = measure.label(mask, connectivity=2)
    props   = measure.regionprops(labeled, intensity_image=np.abs(intensity_map))
    valid   = []
    for p in props:
        area_m2 = p.area * pixel_area
        if area_m2 >= min_area_m2:
            valid.append({
                "label":       int(p.label),
                "area_m2":     float(area_m2),
                "max_dev_m":   float(p.max_intensity),
                "mean_dev_m":  float(p.mean_intensity),
                "centroid":    (float(p.centroid[0]), float(p.centroid[1])),
                "bbox":        list(p.bbox),
            })
    valid.sort(key=lambda x: -x["max_dev_m"])
    return labeled, valid

labeled_dep, valid_dep = label_regions(depression_mask, primary_dev, MIN_AREA_M2, pixel_area)
labeled_elv, valid_elv = label_regions(elevation_mask,  primary_dev, MIN_AREA_M2, pixel_area)

print(f"  Fördjupningar hittade: {len(valid_dep)} st >= {MIN_AREA_M2} m²")
print(f"  Upphöjningar hittade:  {len(valid_elv)} st >= {MIN_AREA_M2} m²")

# Print top 15
if valid_dep:
    print(f"\n  Topp fördjupningar:")
    print(f"  {'#':<5} {'Area (m²)':<12} {'Max avv. (m)':<14}")
    print(f"  {'─'*40}")
    for d in valid_dep[:15]:
        print(f"  {d['label']:<5} {d['area_m2']:<12.1f} {d['max_dev_m']:<14.3f}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
results = {
    "pixel_size_m": pixel_m,
    "radii_m": RADII_M,
    "threshold_depression_m": threshold_low,
    "threshold_elevation_m":  threshold_high,
    "depressions": [
        {"id":d["label"],"area_m2":d["area_m2"],"max_depth_m":d["max_dev_m"],
         "mean_depth_m":d["mean_dev_m"],"centroid_row":d["centroid"][0],
         "centroid_col":d["centroid"][1],"bbox":d["bbox"]}
        for d in valid_dep
    ],
    "elevations": [
        {"id":d["label"],"area_m2":d["area_m2"],"max_height_m":d["max_dev_m"],
         "mean_height_m":d["mean_dev_m"],"centroid_row":d["centroid"][0],
         "centroid_col":d["centroid"][1],"bbox":d["bbox"]}
        for d in valid_elv
    ],
}

json_path = OUT_REPS / "03_depressions.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ JSON sparad: {json_path}")

# ── Figure: 4-panel ───────────────────────────────────────────────────────────
print("\nGenererar figur 03_depressions.png...")
fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=DARK)

def styled_ax(ax, title, subtitle=""):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    full = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full, color=TEXT, fontsize=10, pad=6)

def add_cbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=7)
    cb.outline.set_edgecolor(BORDER)

# Panel 1: Primary deviation map (diverging colormap)
ax = axes[0, 0]
styled_ax(ax, "Lokal terrängavvikelse (5m radius)",
          "Blå=fördjupning, Röd=upphöjning")
vmax = float(np.nanpercentile(np.abs(primary_dev), 98))
im = ax.imshow(primary_dev, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
               interpolation="bilinear")
add_cbar(im, ax, "Avvikelse från lokal medelnivå (m)")

# Panel 2: Multi-scale comparison
ax = axes[0, 1]
styled_ax(ax, "Multi-skala avvikelse",
          f"Ø {RADII_M[0]}m vs {RADII_M[1]}m vs {RADII_M[2]}m")
# Show difference between finest and coarsest scale
fine   = all_deviations[f"{RADII_M[0]:.1f}m"]
coarse = all_deviations[f"{RADII_M[2]:.1f}m"]
scale_diff = fine - coarse  # high = locally rough, low = smooth
im = ax.imshow(scale_diff, cmap="PiYG", interpolation="bilinear",
               vmin=float(np.nanpercentile(scale_diff, 2)),
               vmax=float(np.nanpercentile(scale_diff, 98)))
add_cbar(im, ax, "Finkornig − grovkornig avvikelse (m)")

# Panel 3: Labeled depressions on DTM
ax = axes[1, 0]
n_dep_show = min(50, len(valid_dep))
styled_ax(ax, f"Etiketterade fördjupningar",
          f"{len(valid_dep)} st identifierade (tröskel < {threshold_low}m), visar topp {n_dep_show}")
ax.imshow(dtm_filled, cmap="terrain", alpha=0.5, interpolation="bilinear")
ax.imshow(np.ma.masked_where(~depression_mask, primary_dev),
          cmap="Blues_r", alpha=0.7, interpolation="bilinear",
          vmin=threshold_low, vmax=0)
cmap_tab = plt.cm.tab20
for d in valid_dep[:n_dep_show]:
    cy, cx = d["centroid"]
    ax.annotate(
        f"#{d['label']}\n{d['max_dev_m']:.2f}m",
        (cx, cy), fontsize=5.5, ha="center", va="center",
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="black",
                  alpha=0.65, edgecolor="none"),
    )

# Panel 4: Labeled elevations on DTM
ax = axes[1, 1]
n_elv_show = min(50, len(valid_elv))
styled_ax(ax, f"Etiketterade upphöjningar",
          f"{len(valid_elv)} st identifierade (tröskel > {threshold_high}m), visar topp {n_elv_show}")
ax.imshow(dtm_filled, cmap="terrain", alpha=0.5, interpolation="bilinear")
ax.imshow(np.ma.masked_where(~elevation_mask, primary_dev),
          cmap="Reds", alpha=0.7, interpolation="bilinear",
          vmin=0, vmax=vmax)
for d in valid_elv[:n_elv_show]:
    cy, cx = d["centroid"]
    ax.annotate(
        f"#{d['label']}\n+{d['max_dev_m']:.2f}m",
        (cx, cy), fontsize=5.5, ha="center", va="center",
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#8B0000",
                  alpha=0.7, edgecolor="none"),
    )

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle(
    f"Steg 3 — Terränganomalier  |  "
    f"{len(valid_dep)} fördjupningar, {len(valid_elv)} upphöjningar  |  GPU={GPU_AVAILABLE}",
    fontsize=14, fontweight="bold", color=TEXT, y=0.98
)
fig.patch.set_facecolor(DARK)
plt.tight_layout()

out_path = OUT_FIGS / "03_depressions.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_path}")
print("\nSteg 3 klar.")
print("=" * 60)

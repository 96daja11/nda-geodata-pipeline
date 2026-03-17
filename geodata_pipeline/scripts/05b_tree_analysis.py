#!/usr/bin/env python3
"""
Steg 5b — Trädanalys: Individuell träddetektering och kronanalys
Detekterar individuella träd via lokala maxima i CHM, segmenterar kronor med
watershed-algoritm och beräknar heltäckande trädstatistik.
GPU-accelererad med CuPy (fallback NumPy).
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

import json
import subprocess
import sys
from pathlib import Path

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    cp.array([1])
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    _smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    GPU_NAME = _smi.stdout.strip() if _smi.returncode == 0 else "GPU"
    print(f"GPU: {GPU_NAME} | {_dev.mem_info[1]//1024**2} MB VRAM")
except (ImportError, Exception) as _e:
    import numpy as cp
    import scipy.ndimage as cpnd
    GPU_AVAILABLE = False
    GPU_NAME = "CPU fallback"
    print(f"GPU: Not available ({_e})")

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import measure
from tqdm import tqdm

# ── CLI & Config ──────────────────────────────────────────────────────────────
import argparse, json as _json

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-dir", type=Path, default=None,
                    help="Dataset root directory (default: two levels above this script)")
    return ap.parse_args()

def _load_config(project_dir: Path) -> dict:
    """Load dataset-specific config.json if present, else return empty dict."""
    cfg_path = project_dir / "config.json"
    if cfg_path.exists():
        return _json.loads(cfg_path.read_text())
    return {}

_args = _parse_args()
BASE = _args.project_dir.resolve() if _args.project_dir else Path(__file__).resolve().parent.parent
_cfg = _load_config(BASE)

# Guard: skip if tree_analysis.enabled == False in config
_tree_cfg     = _cfg.get("tree_analysis", {})
if _tree_cfg.get("enabled", True) == False:
    print("=" * 60)
    print("STEG 5b — TRÄDANALYS")
    print("=" * 60)
    print("\nSkipper steg 5b: tree_analysis.enabled=false i config.json")
    sys.exit(0)

SIGMA         = _tree_cfg.get("sigma",            3.0)
VEG_THRESHOLD = _tree_cfg.get("veg_threshold_m",  1.5)
MIN_DISTANCE_PX = _tree_cfg.get("min_distance_px", 100)
TREE_HEIGHT_MIN = _tree_cfg.get("tree_height_min_m", 3.0)

print("=" * 60)
print("STEG 5b — TRÄDANALYS")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE is set above via --project-dir
RAW       = BASE / "data" / "raw"
DERIVED   = BASE / "data" / "derived"
OUT_FIGS  = BASE / "outputs" / "figures"
OUT_REPS  = BASE / "outputs" / "reports"
DERIVED.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPS.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
ACCENT = "#58a6ff"

# ── Step 1: Load DSM + DTM, compute CHM ───────────────────────────────────────
print("\n[1/8] Laddar DSM och DTM, beräknar CHM...")

with rasterio.open(RAW / "dsm.tif") as src:
    dsm       = src.read(1).astype(np.float32)
    nodata    = src.nodata
    pixel_m   = src.res[0]
    transform = src.transform
    crs       = src.crs
    if nodata is not None:
        dsm[dsm == nodata] = np.nan

with rasterio.open(RAW / "dtm.tif") as src:
    dtm    = src.read(1).astype(np.float32)
    nd_dtm = src.nodata
    if nd_dtm is not None:
        dtm[dtm == nd_dtm] = np.nan

print(f"  DSM: {dsm.shape[1]}×{dsm.shape[0]} px  |  {pixel_m:.4f} m/px")
print(f"  DSM höjd: {np.nanmin(dsm):.2f}–{np.nanmax(dsm):.2f} m")
print(f"  DTM höjd: {np.nanmin(dtm):.2f}–{np.nanmax(dtm):.2f} m")

# CHM = DSM - DTM, mask negative values
if GPU_AVAILABLE:
    dsm_g = cp.array(dsm)
    dtm_g = cp.array(dtm)
    chm_g = dsm_g - dtm_g
    chm_g = cp.where(cp.isnan(chm_g), 0.0, chm_g)
    chm_g = cp.where(chm_g < 0, 0.0, chm_g)
    chm = cp.asnumpy(chm_g)
else:
    chm = dsm - dtm
    chm = np.where(np.isnan(chm), 0.0, chm)
    chm[chm < 0] = 0.0

pixel_area = pixel_m ** 2
chm_pos = chm[chm > 0.5]
print(f"  CHM: max={chm.max():.2f} m  |  mean (>0.5m)={chm_pos.mean():.2f} m  |  täckning={100*len(chm_pos)/chm.size:.1f}%")

# Save CHM to derived/
chm_path = DERIVED / "chm.tif"
with rasterio.open(
    chm_path, "w",
    driver="GTiff",
    height=chm.shape[0], width=chm.shape[1],
    count=1, dtype="float32",
    crs=crs, transform=transform
) as dst:
    dst.write(chm.astype(np.float32), 1)
print(f"  CHM sparad: {chm_path}")

# ── Step 2: Gaussian smoothing ────────────────────────────────────────────────
print(f"\n[2/8] Gaussisk utjämning (sigma={SIGMA} px = {SIGMA*pixel_m:.2f} m)...")

if GPU_AVAILABLE:
    chm_g    = cp.array(chm)
    smooth_g = cpnd.gaussian_filter(chm_g, sigma=SIGMA)
    chm_smooth = cp.asnumpy(smooth_g)
else:
    chm_smooth = gaussian_filter(chm, sigma=SIGMA)

chm_smooth[chm_smooth < VEG_THRESHOLD] = 0.0
print(f"  Utjämnad CHM: max={chm_smooth.max():.2f} m  |  sigma={SIGMA} px")

# ── Step 3: Tree detection via local maxima ───────────────────────────────────
print("\n[3/8] Detekterar trädtoppar via lokala maxima...")

# peak_local_max runs on CPU
local_max = peak_local_max(
    chm_smooth,
    min_distance=MIN_DISTANCE_PX,
    threshold_abs=TREE_HEIGHT_MIN,
    exclude_border=True
)

n_trees = len(local_max)
print(f"  Detekterade trädtoppar: {n_trees:,}")
print(f"  Min avstånd: {MIN_DISTANCE_PX} px ({MIN_DISTANCE_PX * pixel_m:.2f} m)")
print(f"  Höjdtröskel: {TREE_HEIGHT_MIN} m")

# Tree top heights
tree_heights_raw = chm_smooth[local_max[:, 0], local_max[:, 1]]
print(f"  Topphöjder: min={tree_heights_raw.min():.2f} m  |  max={tree_heights_raw.max():.2f} m  |  mean={tree_heights_raw.mean():.2f} m")

# ── Step 4: Crown segmentation via watershed ──────────────────────────────────
print("\n[4/8] Segmenterar kronor (watershed)...")

# Create seed markers from detected tree tops
markers = np.zeros(chm_smooth.shape, dtype=np.int32)
for idx, (r, c) in enumerate(local_max):
    markers[r, c] = idx + 1

# Vegetation mask
veg_mask = chm_smooth > VEG_THRESHOLD

# Watershed on inverted CHM (so tree tops are "basins")
labels = watershed(-chm_smooth, markers, mask=veg_mask, compactness=0.001)

print(f"  Watershed klar: {labels.max()} kronregioner")

# ── Step 5: Tree metrics per crown ────────────────────────────────────────────
print("\n[5/8] Beräknar trädmetrik per krona...")

props = measure.regionprops(labels, intensity_image=chm_smooth)

trees = []
for p in tqdm(props, desc="  Kronanalys", leave=False):
    area_m2 = p.area * pixel_area
    if area_m2 < 1.0:   # filter noise < 1 m²
        continue
    radius_m = float(np.sqrt(area_m2 / np.pi))
    trees.append({
        "label":       int(p.label),
        "area_m2":     float(area_m2),
        "crown_radius_m": radius_m,
        "max_height_m":   float(p.max_intensity),
        "mean_height_m":  float(p.mean_intensity),
        "centroid_row":   float(p.centroid[0]),
        "centroid_col":   float(p.centroid[1]),
    })

n_trees_valid = len(trees)
print(f"  Giltiga träd (krona >= 1 m²): {n_trees_valid:,}")

# ── Step 6: Canopy analysis ────────────────────────────────────────────────────
print("\n[6/8] Krontäckningsanalys...")

# Total area of valid (non-NaN) pixels
valid_total_px  = int(np.sum(~np.isnan(dsm)))
canopy_px       = int(np.sum(veg_mask))
total_area_m2   = valid_total_px * pixel_area
canopy_area_m2  = canopy_px * pixel_area
canopy_pct      = 100.0 * canopy_px / max(valid_total_px, 1)
valid_area_ha   = total_area_m2 / 10_000.0
tree_density_ha = n_trees_valid / max(valid_area_ha, 0.001)

# Height class distribution (from valid tree crowns)
all_heights = np.array([t["max_height_m"] for t in trees])
height_dist = {
    "0.5-2m":  int(np.sum((all_heights >= 0.5) & (all_heights < 2))),
    "2-5m":    int(np.sum((all_heights >= 2) & (all_heights < 5))),
    "5-10m":   int(np.sum((all_heights >= 5) & (all_heights < 10))),
    "10-15m":  int(np.sum((all_heights >= 10) & (all_heights < 15))),
    ">15m":    int(np.sum(all_heights >= 15)),
}

crown_areas = np.array([t["area_m2"] for t in trees])
mean_height_all = float(np.mean(all_heights)) if len(all_heights) > 0 else 0.0
max_height_all  = float(np.max(all_heights))  if len(all_heights) > 0 else 0.0

print(f"  Total yta:        {total_area_m2:.0f} m²  ({valid_area_ha:.2f} ha)")
print(f"  Krontäckning:     {canopy_area_m2:.0f} m²  ({canopy_pct:.1f}%)")
print(f"  Trädtäthet:       {tree_density_ha:.1f} träd/ha")
print(f"  Medelhöjd:        {mean_height_all:.2f} m")
print(f"  Maxhöjd:          {max_height_all:.2f} m")
print(f"\n  Höjdklassfördelning:")
for cls, cnt in height_dist.items():
    pct = 100.0 * cnt / max(n_trees_valid, 1)
    print(f"    {cls:<8}: {cnt:>5} träd  ({pct:.1f}%)")

# ── Step 7: Spatial density map ───────────────────────────────────────────────
print("\n[7/8] Beräknar rumslig täthetskarta (10m grid)...")

GRID_M = 10.0
grid_px = max(1, int(GRID_M / pixel_m))

rows_g = int(np.ceil(chm.shape[0] / grid_px))
cols_g = int(np.ceil(chm.shape[1] / grid_px))

# Bin tree top coordinates into 10m grid
tree_rows = np.array([t["centroid_row"] for t in trees])
tree_cols = np.array([t["centroid_col"] for t in trees])

if len(tree_rows) > 0:
    if GPU_AVAILABLE:
        tr_g = cp.array((tree_rows / grid_px).astype(np.int32))
        tc_g = cp.array((tree_cols / grid_px).astype(np.int32))
        tr_g = cp.clip(tr_g, 0, rows_g - 1)
        tc_g = cp.clip(tc_g, 0, cols_g - 1)
        flat_g = tr_g * cols_g + tc_g
        density_grid_flat = cp.bincount(flat_g, minlength=rows_g * cols_g)
        density_grid = cp.asnumpy(density_grid_flat).reshape(rows_g, cols_g).astype(np.float32)
    else:
        tr_int = np.clip((tree_rows / grid_px).astype(np.int32), 0, rows_g - 1)
        tc_int = np.clip((tree_cols / grid_px).astype(np.int32), 0, cols_g - 1)
        flat_idx = tr_int * cols_g + tc_int
        density_grid = np.bincount(flat_idx, minlength=rows_g * cols_g).reshape(rows_g, cols_g).astype(np.float32)
else:
    density_grid = np.zeros((rows_g, cols_g), dtype=np.float32)

print(f"  10m grid: {cols_g}×{rows_g} celler  |  max {density_grid.max():.0f} träd/cell")

# ── Step 8: Save JSON results ─────────────────────────────────────────────────
print("\n[8/8] Sparar resultat...")

results_json = {
    "n_trees":               n_trees_valid,
    "canopy_coverage_pct":   round(canopy_pct, 2),
    "tree_density_per_ha":   round(tree_density_ha, 1),
    "mean_height_m":         round(mean_height_all, 2),
    "max_height_m":          round(max_height_all, 2),
    "total_canopy_area_m2":  round(canopy_area_m2, 1),
    "total_area_m2":         round(total_area_m2, 1),
    "pixel_size_m":          pixel_m,
    "veg_threshold_m":       VEG_THRESHOLD,
    "height_distribution":   height_dist,
    "crown_area_stats": {
        "mean_m2":   round(float(np.mean(crown_areas)), 2) if len(crown_areas) else 0.0,
        "median_m2": round(float(np.median(crown_areas)), 2) if len(crown_areas) else 0.0,
        "p95_m2":    round(float(np.percentile(crown_areas, 95)), 2) if len(crown_areas) else 0.0,
    },
}

json_path = OUT_REPS / "05b_tree_analysis.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, indent=2, ensure_ascii=False)
print(f"  JSON sparad: {json_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def styled_ax(ax, title, subtitle=""):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    full = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full, color=TEXT, fontsize=9, pad=5)

def add_cbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=6)
    cb.outline.set_edgecolor(BORDER)

# ── Figure 1: 05b_tree_overview.png — 2×3 grid ────────────────────────────────
print("\nGenererar figur 05b_tree_overview.png...")

fig, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor=DARK)

# Downsample factor for display (avoid huge arrays in imshow)
step = max(1, max(chm.shape) // 1500)
chm_disp  = chm[::step, ::step]
chmS_disp = chm_smooth[::step, ::step]

# ── [0,0] CHM heatmap ──────────────────────────────────────────────────────────
ax = axes[0, 0]
styled_ax(ax, "CHM — Canopy Height Model", f"max={chm.max():.1f} m  |  {pixel_m*100:.0f} cm/px")
chm_p95 = float(np.percentile(chm[chm > 0], 95)) if chm_pos.size > 0 else chm.max()
im = ax.imshow(chm_disp, cmap="YlGn", vmin=0, vmax=chm_p95, interpolation="bilinear")
add_cbar(im, ax, "CHM Höjd (m)")

# ── [0,1] Smoothed CHM + tree tops ───────────────────────────────────────────
ax = axes[0, 1]
styled_ax(ax, "Utjämnad CHM + Trädtoppar",
          f"{n_trees_valid:,} toppar detekterade")
chmS_p95 = float(np.percentile(chm_smooth[chm_smooth > 0], 95)) if np.sum(chm_smooth > 0) > 0 else 1.0
im = ax.imshow(chmS_disp, cmap="YlGn", vmin=0, vmax=chmS_p95, interpolation="bilinear")
add_cbar(im, ax, "CHM Utjämnad (m)")
# Plot tree tops (subsampled for display speed)
max_plot_trees = 5000
if n_trees_valid > 0:
    plot_idx = np.random.choice(n_trees_valid, min(n_trees_valid, max_plot_trees), replace=False)
    tc_plot = np.array([trees[i]["centroid_col"] for i in plot_idx]) / step
    tr_plot = np.array([trees[i]["centroid_row"] for i in plot_idx]) / step
    ax.scatter(tc_plot, tr_plot, c="#FF4444", s=1.5, alpha=0.6, linewidths=0)

# ── [0,2] Watershed crown map ────────────────────────────────────────────────
ax = axes[0, 2]
styled_ax(ax, "Watershed Kronkarta",
          f"{n_trees_valid:,} kronor segmenterade")
labels_disp = labels[::step, ::step]
# Colorize: use a random colormap, background=dark
np.random.seed(42)
n_labels = int(labels_disp.max()) + 1
rand_colors = np.random.rand(n_labels, 3) * 0.8 + 0.1
rand_colors[0] = [0.05, 0.07, 0.09]  # background = near-black
crown_rgb = rand_colors[labels_disp.astype(int)]
crown_rgb_masked = np.where(
    (labels_disp == 0)[..., np.newaxis],
    np.array([[[0.05, 0.07, 0.09]]]),
    crown_rgb
)
ax.imshow(crown_rgb_masked, interpolation="nearest")

# ── [1,0] Height class map ───────────────────────────────────────────────────
ax = axes[1, 0]
styled_ax(ax, "Höjdklassificering",
          "0.5–2m / 2–5m / 5–10m / 10–15m / >15m")
height_class = np.zeros(chm.shape, dtype=np.uint8)
height_class[(chm >= 0.5) & (chm < 2.0)]  = 1
height_class[(chm >= 2.0) & (chm < 5.0)]  = 2
height_class[(chm >= 5.0) & (chm < 10.0)] = 3
height_class[(chm >= 10.0) & (chm < 15.0)]= 4
height_class[chm >= 15.0]                  = 5
hc_disp = height_class[::step, ::step]
class_colors = ["#0d1117", "#A8D5A2", "#52B347", "#217A35", "#0D4F1E", "#FFDD44"]
cmap_hc = ListedColormap(class_colors)
im = ax.imshow(hc_disp, cmap=cmap_hc, vmin=0, vmax=5, interpolation="nearest")
legend_patches = [
    mpatches.Patch(color=class_colors[0], label="Ingen vegetation"),
    mpatches.Patch(color=class_colors[1], label="0.5–2 m"),
    mpatches.Patch(color=class_colors[2], label="2–5 m"),
    mpatches.Patch(color=class_colors[3], label="5–10 m"),
    mpatches.Patch(color=class_colors[4], label="10–15 m"),
    mpatches.Patch(color=class_colors[5], label=">15 m"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=6,
          facecolor="#21262d", labelcolor=TEXT, edgecolor=BORDER,
          framealpha=0.9, ncol=2)

# ── [1,1] Canopy cover binary map ────────────────────────────────────────────
ax = axes[1, 1]
styled_ax(ax, "Kronövertäckning",
          f"Täckning: {canopy_pct:.1f}%  |  {canopy_area_m2:.0f} m²")
cover_disp = veg_mask[::step, ::step].astype(np.float32)
cover_rgb = np.zeros((*cover_disp.shape, 3), dtype=np.float32)
cover_rgb[cover_disp == 0] = [0.05, 0.07, 0.09]  # non-veg: dark
cover_rgb[cover_disp == 1] = [0.20, 0.65, 0.25]  # veg: green
ax.imshow(cover_rgb, interpolation="nearest")
ax.text(0.02, 0.97, f"Krontäckning: {canopy_pct:.1f}%\n{canopy_area_m2/10000:.2f} ha vegetation",
        transform=ax.transAxes, color=TEXT, fontsize=8, va="top",
        bbox=dict(facecolor="#21262d", alpha=0.85, pad=3, edgecolor=BORDER))

# ── [1,2] Stats text panel ────────────────────────────────────────────────────
ax = axes[1, 2]
ax.set_facecolor(PANEL)
ax.axis("off")
ax.set_title("Sammanfattning — Trädanalys", color=TEXT, fontsize=9, pad=5)

stats_lines = [
    ("Antal detekterade träd",  f"{n_trees_valid:,}"),
    ("Krontäckning",            f"{canopy_pct:.1f}%"),
    ("Trädtäthet",              f"{tree_density_ha:.0f} träd/ha"),
    ("Medelhöjd",               f"{mean_height_all:.2f} m"),
    ("Maxhöjd",                 f"{max_height_all:.2f} m"),
    ("Total krontäckyta",       f"{canopy_area_m2/10000:.2f} ha"),
    ("Medelkrona",              f"{results_json['crown_area_stats']['mean_m2']:.1f} m²"),
    ("Mediankrona",             f"{results_json['crown_area_stats']['median_m2']:.1f} m²"),
    ("0.5–2 m klassen",         f"{height_dist['0.5-2m']} träd"),
    ("2–5 m klassen",           f"{height_dist['2-5m']} träd"),
    ("5–10 m klassen",          f"{height_dist['5-10m']} träd"),
    ("10–15 m klassen",         f"{height_dist['10-15m']} träd"),
    (">15 m klassen",           f"{height_dist['>15m']} träd"),
]
y_pos = 0.96
for lbl, val in stats_lines:
    ax.text(0.05, y_pos, lbl, transform=ax.transAxes,
            fontsize=7.5, color="#8b949e", va="top")
    ax.text(0.95, y_pos, val, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", ha="right", color=TEXT, va="top")
    y_pos -= 0.073

rect = mpatches.FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
    boxstyle="round,pad=0.02", linewidth=1.5,
    edgecolor=ACCENT, facecolor="#0d1117", zorder=0
)
ax.add_patch(rect)

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle(
    f"Steg 5b — Trädanalys  |  {n_trees_valid:,} träd  |  krontäckning {canopy_pct:.1f}%",
    fontsize=14, fontweight="bold", color=TEXT, y=0.98
)
fig.patch.set_facecolor(DARK)
plt.tight_layout()

out1 = OUT_FIGS / "05b_tree_overview.png"
plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out1}")

# ── Figure 2: 05b_tree_density.png — 1×3 grid ────────────────────────────────
print("Genererar figur 05b_tree_density.png...")

fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor=DARK)

# ── [0] Tree density heatmap (10m grid) ──────────────────────────────────────
ax = axes[0]
styled_ax(ax, "Trädtäthetskarta  (10m grid)",
          f"Max: {density_grid.max():.0f} träd/cell")
dens_smooth = gaussian_filter(density_grid.astype(np.float32), sigma=1.0)
im = ax.imshow(dens_smooth, cmap="hot_r", interpolation="bilinear", aspect="auto")
add_cbar(im, ax, "Träd per 10×10 m cell")

# ── [1] Crown area histogram ─────────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor(PANEL)
ax.set_title("Kronytafördelning", color=TEXT, fontsize=9, pad=5)
if len(crown_areas) > 0:
    ax.hist(crown_areas, bins=50, color="#2E7D32", edgecolor=DARK, alpha=0.85, log=True)
    ax.axvline(float(np.mean(crown_areas)), color="#FF9800", linewidth=1.5,
               linestyle="--", label=f"Medel: {np.mean(crown_areas):.1f} m²")
    ax.axvline(float(np.median(crown_areas)), color="#03A9F4", linewidth=1.5,
               linestyle=":", label=f"Median: {np.median(crown_areas):.1f} m²")
    ax.legend(facecolor="#21262d", labelcolor=TEXT, edgecolor=BORDER, fontsize=8)
ax.set_xlabel("Kronyta (m²)", color=TEXT, fontsize=8)
ax.set_ylabel("Antal träd (log-skala)", color=TEXT, fontsize=8)
ax.tick_params(colors=TEXT, labelsize=7)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(BORDER)

# ── [2] Height distribution histogram + stacked bar ──────────────────────────
ax = axes[2]
ax.set_facecolor(PANEL)
ax.set_title("Höjdfördelning — detekterade träd", color=TEXT, fontsize=9, pad=5)

if len(all_heights) > 0:
    # Gradient fill histogram
    n_bins = 30
    counts, bin_edges = np.histogram(all_heights, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cmap_yl = plt.cm.YlGn
    norm_h  = mcolors.Normalize(vmin=bin_edges[0], vmax=bin_edges[-1])
    for bc, cnt, be0, be1 in zip(bin_centers, counts, bin_edges[:-1], bin_edges[1:]):
        ax.bar(bc, cnt, width=(be1 - be0) * 0.9, color=cmap_yl(norm_h(bc)),
               edgecolor=DARK, linewidth=0.3)
    ax.axvline(mean_height_all, color="#FF9800", linewidth=1.5, linestyle="--",
               label=f"Medel: {mean_height_all:.1f} m")
    ax.axvline(max_height_all, color="#F44336", linewidth=1.2, linestyle=":",
               label=f"Max: {max_height_all:.1f} m")
    ax.legend(facecolor="#21262d", labelcolor=TEXT, edgecolor=BORDER, fontsize=8)

    # Inset stacked bar showing height class percentages
    ax_inset = ax.inset_axes([0.55, 0.55, 0.43, 0.40])
    ax_inset.set_facecolor(PANEL)
    hc_labels = list(height_dist.keys())
    hc_vals   = [height_dist[k] for k in hc_labels]
    hc_colors = ["#A8D5A2", "#52B347", "#217A35", "#0D4F1E", "#FFDD44"]
    hc_pcts   = [100.0 * v / max(n_trees_valid, 1) for v in hc_vals]
    ax_inset.bar(range(len(hc_labels)), hc_pcts, color=hc_colors, edgecolor=DARK, linewidth=0.3)
    ax_inset.set_xticks(range(len(hc_labels)))
    ax_inset.set_xticklabels(hc_labels, rotation=30, ha="right", fontsize=4)
    ax_inset.set_ylabel("%", color=TEXT, fontsize=6)
    ax_inset.tick_params(colors=TEXT, labelsize=5)
    ax_inset.set_ylim(0, max(hc_pcts) * 1.3 if hc_pcts else 1)
    for sp in ["top", "right"]:
        ax_inset.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_inset.spines[sp].set_color(BORDER)

ax.set_xlabel("Trädtopphöjd (m)", color=TEXT, fontsize=8)
ax.set_ylabel("Antal träd", color=TEXT, fontsize=8)
ax.tick_params(colors=TEXT, labelsize=7)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(BORDER)

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle(
    f"Steg 5b — Trädtäthet och kronstatistik  |  {n_trees_valid:,} träd  |  {tree_density_ha:.0f} träd/ha",
    fontsize=13, fontweight="bold", color=TEXT, y=1.00
)
fig.patch.set_facecolor(DARK)
plt.tight_layout()

out2 = OUT_FIGS / "05b_tree_density.png"
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out2}")

# ── Figure 3: 05b_chm_profile.png — profillinje ───────────────────────────────
print("Genererar figur 05b_chm_profile.png...")

profile_fracs = [0.25, 0.50, 0.75]
profile_colors = ["#58a6ff", "#FF9800", "#4CAF50"]

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9), facecolor=DARK,
                           gridspec_kw={"height_ratios": [1.5, 1]})

# Top row: CHM with 3 profile lines
ax_chm = axes2[0, 0]
ax_chm.set_facecolor(PANEL)
ax_chm.axis("off")
ax_chm.set_title("CHM med profillinjers position", color=TEXT, fontsize=9, pad=5)
im_chm = ax_chm.imshow(chm_disp, cmap="YlGn", vmin=0, vmax=chm_p95,
                        interpolation="bilinear", aspect="auto")
add_cbar(im_chm, ax_chm, "CHM (m)")
h_disp, w_disp = chm_disp.shape
for frac, col in zip(profile_fracs, profile_colors):
    row_disp = int(h_disp * frac)
    ax_chm.axhline(row_disp, color=col, linewidth=1.5, linestyle="--", alpha=0.9)

# Second and third subplots in top row — unused, hide them
for ax_empty in [axes2[0, 1], axes2[0, 2]]:
    ax_empty.set_facecolor(PANEL)
    ax_empty.axis("off")

# Bottom row: 3 height profiles
for idx, (frac, col) in enumerate(zip(profile_fracs, profile_colors)):
    ax_p = axes2[1, idx]
    ax_p.set_facecolor(PANEL)

    row = int(chm.shape[0] * frac)
    profile = chm[row, :]
    x_m = np.arange(len(profile)) * pixel_m

    ax_p.fill_between(x_m, 0, profile, alpha=0.35, color=col)
    ax_p.plot(x_m, profile, color=col, linewidth=1.2, alpha=0.9)
    ax_p.axhline(0, color=BORDER, linewidth=0.5)
    ax_p.set_xlabel("Avstånd (m)", color=TEXT, fontsize=7)
    ax_p.set_ylabel("CHM (m)", color=TEXT, fontsize=7)
    ax_p.set_title(f"Profil vid {int(frac*100)}% av bildhöjden  (rad {row})",
                   color=TEXT, fontsize=8, pad=4)
    ax_p.tick_params(colors=TEXT, labelsize=6)
    ax_p.grid(True, alpha=0.12, color=BORDER)
    for sp in ax_p.spines.values():
        sp.set_edgecolor(BORDER)

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig2.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
          fontsize=7, color="#586069", style="italic")
fig2.suptitle(
    f"Steg 5b — CHM Höjdprofiler  |  {n_trees_valid:,} träd  |  täckning {canopy_pct:.1f}%",
    fontsize=13, fontweight="bold", color=TEXT, y=1.01
)
fig2.patch.set_facecolor(DARK)
plt.tight_layout()

out3 = OUT_FIGS / "05b_chm_profile.png"
plt.savefig(out3, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out3}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRÄDANALYS — SAMMANFATTNING")
print("=" * 60)
print(f"  Antal träd:           {n_trees_valid:,}")
print(f"  Krontäckning:         {canopy_pct:.1f}%  ({canopy_area_m2:.0f} m²)")
print(f"  Trädtäthet:           {tree_density_ha:.1f} träd/ha")
print(f"  Medelhöjd:            {mean_height_all:.2f} m")
print(f"  Maxhöjd:              {max_height_all:.2f} m")
print(f"  Medelkrona:           {results_json['crown_area_stats']['mean_m2']:.1f} m²")
print(f"  Mediankrona:          {results_json['crown_area_stats']['median_m2']:.1f} m²")
print(f"  JSON:                 {json_path}")
print("=" * 60)
print("\nSteg 5b klar.")

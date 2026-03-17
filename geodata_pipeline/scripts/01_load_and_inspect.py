#!/usr/bin/env python3
"""
Steg 1 — Ladda och visualisera grunddata
Skapar 2×3 panel med DSM, DTM, CHM, ortofoto, histogram och höjdprofil.
GPU-accelererad statistik och histogram med CuPy (fallback NumPy).
"""
import matplotlib
matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    cp.array([1])
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    import subprocess as _sp
    _smi = _sp.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"], capture_output=True, text=True)
    GPU_NAME = _smi.stdout.strip() if _smi.returncode == 0 else "RTX 3060"
    print(f"GPU: {GPU_NAME} | {_dev.mem_info[1]//1024**2} MB VRAM")
except (ImportError, Exception) as _e:
    import numpy as cp
    GPU_AVAILABLE = False
    print(f"GPU: Not available, using CPU ({_e})")

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

print("=" * 60)
print("STEG 1 — LADDA OCH VISUALISERA GRUNDDATA")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE is set above via --project-dir
RAW      = BASE / "data" / "raw"
OUT_FIGS = BASE / "outputs" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"

# ── Load rasters ──────────────────────────────────────────────────────────────
def load_raster(path):
    """Load single-band raster, replacing nodata with NaN."""
    with rasterio.open(path) as src:
        data     = src.read(1).astype(np.float32)
        nodata   = src.nodata
        tf       = src.transform
        crs      = src.crs
        pixel_m  = src.res[0]
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, tf, crs, pixel_m

print("\nLaddar DSM...", end=" ", flush=True)
dsm, tf, crs, pixel_m = load_raster(RAW / "dsm.tif")
print(f"OK  ({dsm.shape[1]}×{dsm.shape[0]} px, {pixel_m:.4f} m/px)")

print("Laddar DTM...", end=" ", flush=True)
dtm, *_ = load_raster(RAW / "dtm.tif")
print(f"OK  ({dtm.shape[1]}×{dtm.shape[0]} px)")

print("Laddar Orthophoto...", end=" ", flush=True)
with rasterio.open(RAW / "odm_orthophoto.tif") as src:
    ortho_r = src.read(1).astype(np.float32)
    ortho_g = src.read(2).astype(np.float32)
    ortho_b = src.read(3).astype(np.float32)
print(f"OK  ({ortho_r.shape[1]}×{ortho_r.shape[0]} px)")

# ── GPU-accelerated stats ─────────────────────────────────────────────────────
def gpu_stats(arr_np, label):
    if GPU_AVAILABLE:
        g = cp.array(arr_np)
        valid = g[~cp.isnan(g)]
    else:
        valid = arr_np[~np.isnan(arr_np)]
    stats = {
        "min":  float(valid.min()),
        "max":  float(valid.max()),
        "mean": float(valid.mean()),
        "std":  float(valid.std()),
        "p5":   float(cp.percentile(valid, 5)  if GPU_AVAILABLE else np.percentile(valid, 5)),
        "p95":  float(cp.percentile(valid, 95) if GPU_AVAILABLE else np.percentile(valid, 95)),
    }
    print(f"\n  Statistik {label} (GPU={GPU_AVAILABLE}):")
    for k, v in stats.items():
        print(f"    {k:>5}: {v:.3f} m")
    return stats

dsm_stats = gpu_stats(dsm, "DSM")
dtm_stats = gpu_stats(dtm, "DTM")

# ── CHM: GPU-accelerated ──────────────────────────────────────────────────────
if GPU_AVAILABLE:
    dsm_g = cp.array(dsm)
    dtm_g = cp.array(dtm)
    chm_g = dsm_g - dtm_g
    chm_g[chm_g < 0] = 0
    chm = cp.asnumpy(chm_g)
    dsm_valid_cpu = cp.asnumpy(dsm_g[~cp.isnan(dsm_g)])
    dtm_valid_cpu = cp.asnumpy(dtm_g[~cp.isnan(dtm_g)])
else:
    chm = dsm - dtm
    chm[chm < 0] = 0
    dsm_valid_cpu = dsm[~np.isnan(dsm)]
    dtm_valid_cpu = dtm[~np.isnan(dtm)]

chm_pos = chm[chm > 0]
print(f"\n  CHM:  max={chm.max():.2f} m  |  mean (>0)={chm_pos.mean():.2f} m  |  coverage={100*len(chm_pos)/chm.size:.1f}%")

# ── Helper functions ──────────────────────────────────────────────────────────
def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title(title, color=TEXT, fontsize=9, pad=5)

def add_cbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=6)
    cb.outline.set_edgecolor(BORDER)

# ── Figure 1: 2×3 overview panel ─────────────────────────────────────────────
print("\nGenererar figur 01_overview.png...")
fig = plt.figure(figsize=(20, 13), facecolor=DARK)
gs  = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.25,
               top=0.88, bottom=0.05, left=0.03, right=0.97)

# DSM
ax1 = fig.add_subplot(gs[0, 0])
styled_ax(ax1, "DSM — Digital Surface Model\n(inkl. vegetation & strukturer)")
im1 = ax1.imshow(dsm, cmap="terrain", interpolation="bilinear")
add_cbar(im1, ax1, "Höjd (m)")

# DTM
ax2 = fig.add_subplot(gs[0, 1])
styled_ax(ax2, "DTM — Digital Terrain Model\n(markmodell)")
im2 = ax2.imshow(dtm, cmap="terrain", interpolation="bilinear")
add_cbar(im2, ax2, "Höjd (m)")

# CHM
ax3 = fig.add_subplot(gs[0, 2])
styled_ax(ax3, "CHM — Canopy Height Model\n(DSM − DTM)")
im3 = ax3.imshow(chm, cmap="YlGn", interpolation="bilinear")
add_cbar(im3, ax3, "Höjd (m)")

# Orthophoto (wide)
ax4 = fig.add_subplot(gs[1, 0:2])
styled_ax(ax4, f"Ortofoto (RGB)  |  {pixel_m:.2f} cm/pixel")
oh, ow = ortho_r.shape
step = max(1, max(oh, ow) // 2000)
r_norm = np.clip(ortho_r[::step, ::step], 0, 255) / 255.0
g_norm = np.clip(ortho_g[::step, ::step], 0, 255) / 255.0
b_norm = np.clip(ortho_b[::step, ::step], 0, 255) / 255.0
rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
ax4.imshow(rgb)
ax4.set_title(f"Ortofoto (RGB)  |  {pixel_m*100:.1f} cm/pixel  |  {ow}×{oh} px",
              color=TEXT, fontsize=9, pad=5)

# Histogram
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
ax5.set_title("Höjdfördelning — DSM vs DTM", color=TEXT, fontsize=9, pad=5)
ax5.hist(dsm_valid_cpu.flatten(), bins=120, alpha=0.7, color="#2196F3", label="DSM")
ax5.hist(dtm_valid_cpu.flatten(), bins=120, alpha=0.7, color="#4CAF50", label="DTM")
ax5.set_xlabel("Höjd (m)", color=TEXT, fontsize=8)
ax5.set_ylabel("Antal pixlar", color=TEXT, fontsize=8)
ax5.tick_params(colors=TEXT, labelsize=7)
ax5.legend(facecolor="#21262d", labelcolor=TEXT, fontsize=9, edgecolor=BORDER)
for sp in ["top", "right"]:
    ax5.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax5.spines[sp].set_color(BORDER)

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle("Steg 1 — Översikt av ingångsdata",
             fontsize=14, fontweight="bold", color=TEXT, y=0.94)

out1 = OUT_FIGS / "01_overview.png"
plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out1}")

# ── Figure 2: Elevation profile ───────────────────────────────────────────────
print("Genererar figur 01_elevation_profile.png...")
fig2, axes2 = plt.subplots(
    2, 1, figsize=(15, 9),
    gridspec_kw={"height_ratios": [1.2, 2]},
    facecolor=DARK
)

row = int(dsm.shape[0] * 0.5)
profile_dsm = dsm[row, :]
profile_dtm = dtm[row, :]
x_m = np.arange(len(profile_dsm)) * pixel_m

# DSM position map
ax_map = axes2[0]
ax_map.set_facecolor(PANEL)
ax_map.imshow(dsm, cmap="terrain", interpolation="bilinear", aspect="auto")
ax_map.axhline(row, color="#F44336", linewidth=2.5, linestyle="--",
               label=f"Profillinje (rad {row})")
ax_map.set_title("Profillinjens position i DSM", color=TEXT, fontsize=11)
ax_map.legend(loc="upper right", facecolor="#21262d",
              labelcolor=TEXT, edgecolor=BORDER, fontsize=9)
ax_map.axis("off")

# Profile curves
ax_prof = axes2[1]
ax_prof.set_facecolor(PANEL)
floor = min(np.nanmin(profile_dsm), np.nanmin(profile_dtm)) - 0.5
ax_prof.fill_between(x_m, floor, profile_dsm, alpha=0.20, color="#2196F3")
ax_prof.fill_between(x_m, floor, profile_dtm, alpha=0.35, color="#4CAF50")
ax_prof.fill_between(x_m, profile_dtm, profile_dsm,
                     alpha=0.45, color="#FFC107", label="CHM (vegetationshöjd)")
ax_prof.plot(x_m, profile_dsm, color="#2196F3", linewidth=1.6, label="DSM")
ax_prof.plot(x_m, profile_dtm, color="#4CAF50", linewidth=1.6, label="DTM")
ax_prof.set_xlabel("Avstånd längs profil (m)", color=TEXT, fontsize=9)
ax_prof.set_ylabel("Höjd (m)", color=TEXT, fontsize=9)
ax_prof.set_title("Höjdprofil — DSM, DTM och CHM (vegetationshöjd)", color=TEXT, fontsize=11)
ax_prof.legend(facecolor="#21262d", labelcolor=TEXT, edgecolor=BORDER, fontsize=9)
ax_prof.tick_params(colors=TEXT, labelsize=8)
ax_prof.grid(True, alpha=0.15, color=BORDER)
for sp in ax_prof.spines.values():
    sp.set_edgecolor(BORDER)

fig2.suptitle("Steg 1 — Höjdprofil tvärs undersökningsområdet",
              fontsize=12, fontweight="bold", color=TEXT, y=1.01)
fig2.patch.set_facecolor(DARK)
plt.tight_layout()

out2 = OUT_FIGS / "01_elevation_profile.png"
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out2}")
print("\nSteg 1 klar.")
print("=" * 60)

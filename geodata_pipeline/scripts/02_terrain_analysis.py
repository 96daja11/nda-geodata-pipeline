#!/usr/bin/env python3
"""
Steg 2 — Terränganalys
Beräknar slope, aspekt och hillshade. Försöker GPU-accelererad gradient med CuPy,
faller tillbaka på richdem/scipy vid behov.
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
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

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
print("STEG 2 — TERRÄNGANALYS")
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

# ── Load DTM ──────────────────────────────────────────────────────────────────
print("\nLaddar DTM...", end=" ", flush=True)
with rasterio.open(RAW / "dtm.tif") as src:
    dtm     = src.read(1).astype(np.float64)
    nodata  = src.nodata
    pixel_m = src.res[0]
    if nodata is not None:
        dtm[dtm == nodata] = np.nan
print(f"OK  ({dtm.shape[1]}×{dtm.shape[0]} px, {pixel_m:.4f} m/px)")

# ── Compute slope & aspect ────────────────────────────────────────────────────
def compute_slope_aspect_gpu(dtm_np, px_size):
    """GPU-accelerated slope & aspect via CuPy gradient."""
    print("  Metod: CuPy GPU-gradient")
    g = cp.array(dtm_np.copy())
    g = cp.where(cp.isnan(g), 0.0, g)
    gy, gx = cp.gradient(g, px_size)
    slope_rad = cp.arctan(cp.sqrt(gx**2 + gy**2))
    slope_deg = cp.degrees(slope_rad)
    aspect    = cp.degrees(cp.arctan2(-gx, gy)) % 360
    return cp.asnumpy(slope_deg), cp.asnumpy(aspect)

def compute_slope_aspect_richdem(dtm_np, px_size):
    """richdem-based slope & aspect (hydro-correct)."""
    try:
        import richdem as rd
        print("  Metod: richdem (CPU)")
        dtm_fill = np.where(np.isnan(dtm_np), -9999.0, dtm_np)
        rda = rd.rdarray(dtm_fill, no_data=-9999.0)
        slope_rr  = rd.TerrainAttribute(rda, attrib="slope_riserun")
        slope_deg = np.degrees(np.arctan(np.array(slope_rr)))
        aspect    = np.array(rd.TerrainAttribute(rda, attrib="aspect"))
        return slope_deg, aspect
    except Exception:
        print("  Metod: scipy.ndimage sobel (CPU fallback)")
        from scipy.ndimage import sobel
        dtm_nf = np.where(np.isnan(dtm_np), 0.0, dtm_np)
        gx = sobel(dtm_nf, axis=1) / (8.0 * px_size)
        gy = sobel(dtm_nf, axis=0) / (8.0 * px_size)
        slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        aspect    = np.degrees(np.arctan2(-gx, gy)) % 360
        return slope_deg, aspect

print("\nBeräknar slope och aspekt...")
if GPU_AVAILABLE:
    slope, aspect = compute_slope_aspect_gpu(dtm, pixel_m)
else:
    slope, aspect = compute_slope_aspect_richdem(dtm, pixel_m)

# Mask NaN regions
nan_mask = np.isnan(dtm)
slope[nan_mask]  = np.nan
aspect[nan_mask] = np.nan

# ── Hillshade ─────────────────────────────────────────────────────────────────
print("Beräknar hillshade...")
ls = LightSource(azdeg=315, altdeg=45)
dtm_filled = np.where(np.isnan(dtm), np.nanmin(dtm), dtm)
hillshade = ls.hillshade(dtm_filled, vert_exag=2.0, dx=pixel_m, dy=pixel_m)

# ── Statistics ────────────────────────────────────────────────────────────────
print("\n--- Terrängstatistik ---")
valid_slope = slope[~np.isnan(slope)]
print(f"  Medelutning:            {np.nanmean(valid_slope):.2f}°")
print(f"  Maxlutning:             {np.nanmax(valid_slope):.2f}°")
print(f"  Medianlutnng:           {np.nanmedian(valid_slope):.2f}°")
print(f"  Andel plant (<5°):      {100*(valid_slope<5).mean():.1f}%")
print(f"  Andel måttlig (5-15°):  {100*((valid_slope>=5)&(valid_slope<15)).mean():.1f}%")
print(f"  Andel brant (15-30°):   {100*((valid_slope>=15)&(valid_slope<30)).mean():.1f}%")
print(f"  Andel mycket brant >30°:{100*(valid_slope>=30).mean():.1f}%")

# ── Figure: 2×2 panel ─────────────────────────────────────────────────────────
print("\nGenererar figur 02_terrain_analysis.png...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=DARK)

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title(title, color=TEXT, fontsize=10, pad=6)

def add_cbar(im, ax, label, ticks=None, ticklabels=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=7)
    cb.outline.set_edgecolor(BORDER)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)

# Panel 1: Hillshade + DTM overlay
ax = axes[0, 0]
styled_ax(ax, "Hillshade + DTM  (NV-belysning, vert. exagg. ×2)")
ax.imshow(hillshade, cmap="gray", interpolation="bilinear")
im = ax.imshow(dtm, cmap="terrain", alpha=0.45, interpolation="bilinear")
add_cbar(im, ax, "Höjd (m)")

# Panel 2: Slope
ax = axes[0, 1]
styled_ax(ax, "Lutning (Slope)  |  Rött=brant, Grönt=plant")
im = ax.imshow(slope, cmap="RdYlGn_r", vmin=0, vmax=45, interpolation="bilinear")
add_cbar(im, ax, "Lutning (°)")

# Panel 3: Aspect
ax = axes[1, 0]
styled_ax(ax, "Aspekt (Exposition)  |  Kompassriktning för sluttning")
im = ax.imshow(aspect, cmap="hsv", vmin=0, vmax=360, interpolation="bilinear")
add_cbar(im, ax, "Aspekt (°)",
         ticks=[0, 90, 180, 270, 360],
         ticklabels=["N", "Ö", "S", "V", "N"])

# Panel 4: Slope classification
ax = axes[1, 1]
styled_ax(ax, "Lutningsklassificering")
slope_cls = np.zeros_like(slope)
slope_cls[slope < 5]                      = 1  # plant
slope_cls[(slope >= 5)  & (slope < 15)]   = 2  # måttlig
slope_cls[(slope >= 15) & (slope < 30)]   = 3  # brant
slope_cls[slope >= 30]                    = 4  # mycket brant
slope_cls[nan_mask]                       = 0

cmap_cls = mcolors.ListedColormap(["#1a1a2e", "#A5D6A7", "#FFF176", "#FFB74D", "#EF5350"])
im = ax.imshow(slope_cls, cmap=cmap_cls, vmin=-0.5, vmax=4.5, interpolation="nearest")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_ticks([0, 1, 2, 3, 4])
cb.set_ticklabels(["Nodata", "Plant (<5°)", "Måttlig (5-15°)",
                   "Brant (15-30°)", "Mycket brant (>30°)"], fontsize=7)
cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
cb.outline.set_edgecolor(BORDER)

# Area percentages
percents = [
    f"Plant:         {100*(slope_cls==1).sum()/slope_cls.size:.1f}%",
    f"Måttlig:       {100*(slope_cls==2).sum()/slope_cls.size:.1f}%",
    f"Brant:         {100*(slope_cls==3).sum()/slope_cls.size:.1f}%",
    f"Mycket brant:  {100*(slope_cls==4).sum()/slope_cls.size:.1f}%",
]
for i, txt in enumerate(percents):
    ax.text(0.02, 0.96 - i*0.07, txt, transform=ax.transAxes,
            color=TEXT, fontsize=7, va="top",
            bbox=dict(facecolor="#21262d", alpha=0.7, pad=2, edgecolor="none"))

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle("Steg 2 — Terränganalys",
             fontsize=15, fontweight="bold", color=TEXT, y=0.97)
fig.patch.set_facecolor(DARK)
plt.tight_layout()

out_path = OUT_FIGS / "02_terrain_analysis.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_path}")
print("\nSteg 2 klar.")
print("=" * 60)

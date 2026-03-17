#!/usr/bin/env python3
"""
Steg 6 — Rapportklara sammanfattningsfigurer
Sammanställer alla analysresultat i ett professionellt 4×3-rutnät.
Läser om alla rådata och reproducerar nyckelvisualiseringar.
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, ListedColormap
from skimage import measure
import laspy

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
print("STEG 6 — RAPPORTKLARA SAMMANFATTNINGSFIGURER")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE is set above via --project-dir
RAW      = BASE / "data" / "raw"
OUT_FIGS = BASE / "outputs" / "figures"
OUT_REPS = BASE / "outputs" / "reports"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
ACCENT = "#58a6ff"

# ── Load all rasters ──────────────────────────────────────────────────────────
def load_raster(path):
    with rasterio.open(path) as src:
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        pixel_m = src.res[0]
        crs     = str(src.crs)
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, pixel_m, crs

print("\nLaddar data...", end=" ", flush=True)
dsm, pixel_m, crs_str = load_raster(RAW / "dsm.tif")
dtm, *_              = load_raster(RAW / "dtm.tif")

with rasterio.open(RAW / "odm_orthophoto.tif") as src:
    ortho_r = src.read(1).astype(np.float32)
    ortho_g = src.read(2).astype(np.float32)
    ortho_b = src.read(3).astype(np.float32)

print("OK")

# ── Derived layers ────────────────────────────────────────────────────────────
print("Beräknar avledda lager...", end=" ", flush=True)

# CHM
chm = dsm - dtm
chm[chm < 0] = 0

# Hillshade
ls = LightSource(azdeg=315, altdeg=45)
dtm_filled = np.where(np.isnan(dtm), np.nanmin(dtm), dtm)
hillshade = ls.hillshade(dtm_filled.astype(np.float64), vert_exag=2.0,
                          dx=pixel_m, dy=pixel_m)

# Slope
if GPU_AVAILABLE:
    dtm_g = cp.array(dtm_filled.astype(np.float64))
    gy, gx = cp.gradient(dtm_g, pixel_m)
    slope = cp.asnumpy(cp.degrees(cp.arctan(cp.sqrt(gx**2 + gy**2))))
else:
    gy, gx = np.gradient(dtm_filled.astype(np.float64), pixel_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))

slope[np.isnan(dtm)] = np.nan

# Depressions
def fill_depressions(dtm_np):
    try:
        import richdem as rd
        dtm_f = np.where(np.isnan(dtm_np), -9999.0, dtm_np)
        rda = rd.rdarray(dtm_f.copy(), no_data=-9999.0)
        f = rd.FillDepressions(rda, in_place=False)
        return np.array(f)
    except Exception:
        from scipy.ndimage import grey_dilation
        dtm_f = np.where(np.isnan(dtm_np), np.nanmin(dtm_np)-1, dtm_np)
        seed = dtm_f.copy()
        b = np.zeros_like(seed, dtype=bool)
        b[0,:]=b[-1,:]=b[:,0]=b[:,-1]=True
        seed[~b] = dtm_f.min()
        prev = seed.copy()
        for _ in range(50):
            d = grey_dilation(prev, size=(3,3))
            r = np.minimum(d, dtm_f)
            if np.allclose(r, prev, atol=1e-6):
                break
            prev = r
        return prev

filled = fill_depressions(dtm)
dep_depth = filled - dtm
dep_depth[np.isnan(dep_depth)] = 0
dep_depth[dep_depth < 0.05]    = 0

print("OK")

# Label depressions
binary = dep_depth > 0
labeled = measure.label(binary, connectivity=2)
props   = measure.regionprops(labeled, intensity_image=dep_depth)
pixel_area = pixel_m ** 2
valid_deps = [p for p in props if p.area * pixel_area >= 50.0]
valid_deps.sort(key=lambda p: -p.max_intensity)

# Load pointcloud
print("Laddar punktmoln...", end=" ", flush=True)
laz_path = RAW / "georeferenced_model.laz"
las = laspy.read(str(laz_path))
pc_x = las.x.scaled_array().astype(np.float32)
pc_y = las.y.scaled_array().astype(np.float32)
pc_z = las.z.scaled_array().astype(np.float32)
pc_cls = np.array(las.classification, dtype=np.uint8)
n_pts = len(pc_x)
# Build 1m density
xmin, xmax = float(pc_x.min()), float(pc_x.max())
ymin, ymax = float(pc_y.min()), float(pc_y.max())
pc_cols = int(np.ceil((xmax-xmin))) + 1
pc_rows = int(np.ceil((ymax-ymin))) + 1
xi_pc = np.clip(((pc_x - xmin)).astype(np.int32), 0, pc_cols-1)
yi_pc = np.clip(((ymax - pc_y)).astype(np.int32), 0, pc_rows-1)
flat  = (yi_pc * pc_cols + xi_pc).astype(np.int64)
dens  = np.bincount(flat, minlength=pc_rows*pc_cols).reshape(pc_rows, pc_cols).astype(np.float32)
print(f"OK  ({n_pts:,} pts)")

# Load volumes JSON if available
vol_json = OUT_REPS / "04_volumes.json"
volumes_data = []
if vol_json.exists():
    with open(vol_json) as f:
        volumes_data = json.load(f)

# Orthophoto thumbnail
oh, ow = ortho_r.shape
step = max(1, max(oh, ow) // 1500)
rgb_thumb = np.stack([
    np.clip(ortho_r[::step, ::step], 0, 255) / 255.0,
    np.clip(ortho_g[::step, ::step], 0, 255) / 255.0,
    np.clip(ortho_b[::step, ::step], 0, 255) / 255.0,
], axis=-1)

# ── Figure: 4×3 grid ──────────────────────────────────────────────────────────
print("\nGenererar figur 06_final_report.png...")
fig = plt.figure(figsize=(24, 30), facecolor=DARK)
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.25,
                        top=0.93, bottom=0.03, left=0.04, right=0.97)

def sax(ax, title, subtitle=""):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    full = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full, color=TEXT, fontsize=9, pad=5)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

def cbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=6)
    cb.outline.set_edgecolor(BORDER)

# ── Row 0: Input data overview ────────────────────────────────────────────────
# 0,0 DSM
ax = fig.add_subplot(gs[0, 0])
sax(ax, "DSM — Digital Surface Model")
im = ax.imshow(dsm, cmap="terrain", interpolation="bilinear")
cbar(im, ax, "Höjd (m)")

# 0,1 DTM
ax = fig.add_subplot(gs[0, 1])
sax(ax, "DTM — Digital Terrain Model")
im = ax.imshow(dtm, cmap="terrain", interpolation="bilinear")
cbar(im, ax, "Höjd (m)")

# 0,2 CHM
ax = fig.add_subplot(gs[0, 2])
sax(ax, "CHM — Canopy Height Model", "(DSM − DTM)")
im = ax.imshow(chm, cmap="YlGn", interpolation="bilinear")
cbar(im, ax, "Höjd (m)")

# ── Row 1: Terrain analysis ───────────────────────────────────────────────────
# 1,0 Hillshade
ax = fig.add_subplot(gs[1, 0])
sax(ax, "Hillshade + DTM", "(NV-belysning, exagg. ×2)")
ax.imshow(hillshade, cmap="gray", interpolation="bilinear")
im = ax.imshow(dtm, cmap="terrain", alpha=0.4, interpolation="bilinear")
cbar(im, ax, "Höjd (m)")

# 1,1 Slope
ax = fig.add_subplot(gs[1, 1])
sax(ax, "Lutning (Slope)", f"Medel: {np.nanmean(slope):.1f}°  Max: {np.nanmax(slope):.1f}°")
im = ax.imshow(slope, cmap="RdYlGn_r", vmin=0, vmax=45, interpolation="bilinear")
cbar(im, ax, "Lutning (°)")

# 1,2 Orthophoto
ax = fig.add_subplot(gs[1, 2])
sax(ax, "Ortofoto (RGB)", f"{pixel_m*100:.1f} cm/px  |  {ow}×{oh} px")
ax.imshow(rgb_thumb)

# ── Row 2: Depression / anomaly ───────────────────────────────────────────────
# 2,0 DTM + depression overlay
ax = fig.add_subplot(gs[2, 0])
sax(ax, "Fördjupningar (blå overlay)", f"{len(valid_deps)} fördjupningar >= 50 m²")
ax.imshow(dtm, cmap="terrain", interpolation="bilinear")
masked_d = np.ma.masked_where(dep_depth == 0, dep_depth)
im = ax.imshow(masked_d, cmap="Blues", alpha=0.85, interpolation="bilinear")
cbar(im, ax, "Djup (m)")

# 2,1 Labeled depressions
ax = fig.add_subplot(gs[2, 1])
sax(ax, "Etiketterade fördjupningar", f"Topp {min(10, len(valid_deps))} visas")
ax.imshow(dtm, cmap="gray", alpha=0.4, interpolation="bilinear")
cmap_tab = plt.cm.tab20
for i, p in enumerate(valid_deps[:10]):
    mask = labeled == p.label
    color = cmap_tab(i % 20)
    ax.contourf(mask.astype(float), levels=[0.5, 1.5], colors=[color], alpha=0.65)
    cy, cx = p.centroid
    ax.text(cx, cy, f"#{p.label}\n{p.max_intensity:.2f}m",
            ha="center", va="center", fontsize=5.5, color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

# 2,2 Depth map
ax = fig.add_subplot(gs[2, 2])
sax(ax, "Fördjupningsdjupskarta")
im = ax.imshow(dep_depth, cmap="Blues", interpolation="bilinear")
cbar(im, ax, "Djup (m)")
contours = measure.find_contours(dep_depth, 0.1)
for c in contours:
    ax.plot(c[:, 1], c[:, 0], "-", color="#F44336", linewidth=0.4, alpha=0.5)

# ── Row 3: Point cloud + volume summary ──────────────────────────────────────
# 3,0 Point cloud density
ax = fig.add_subplot(gs[3, 0])
sax(ax, "Punkttäthetskarta  (log-skala)", f"{n_pts:,} punkter  |  1 m/cell")
im = ax.imshow(np.log1p(dens), cmap="inferno", interpolation="bilinear", aspect="auto")
cbar(im, ax, "log(täthet + 1)")

# 3,1 Volume bar chart (if JSON exists)
ax = fig.add_subplot(gs[3, 1])
ax.set_facecolor(PANEL)
if volumes_data:
    obj_labels  = [f"#{d['object_id']}" for d in volumes_data]
    mean_vols   = [np.mean(list(d["volumes_m3"].values())) for d in volumes_data]
    max_depths  = [d["max_depth_m"] for d in volumes_data]
    bar_colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(obj_labels)))
    bars = ax.bar(obj_labels, mean_vols, color=bar_colors, edgecolor=DARK, width=0.6)
    for bar, vol, dep in zip(bars, mean_vols, max_depths):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(mean_vols)*0.02,
                f"{vol:.1f} m³\n({dep:.2f} m)",
                ha="center", fontsize=7, color=TEXT, fontweight="bold")
    ax.set_ylabel("Medel volym (m³)", color=TEXT, fontsize=8)
    ax.set_xlabel("Objekt-ID", color=TEXT, fontsize=8)
    ax.set_title(f"Volymer per fördjupning  (topp {len(volumes_data)})",
                 color=TEXT, fontsize=9, pad=5)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_ylim(0, max(mean_vols) * 1.35 if mean_vols else 1)
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color(BORDER)
else:
    ax.axis("off")
    ax.text(0.5, 0.5, "Kör 04_volume_analysis.py\nför volymdata",
            transform=ax.transAxes, ha="center", va="center",
            color="#586069", fontsize=11)
    ax.set_title("Volymer (ej tillgängliga)", color=TEXT, fontsize=9, pad=5)

# 3,2 Summary stats panel
ax = fig.add_subplot(gs[3, 2])
ax.set_facecolor(PANEL)
ax.axis("off")
ax.set_title("Sammanfattning", color=TEXT, fontsize=10, pad=5)

valid_slope_flat = slope[~np.isnan(slope)]
stats = [
    ("Dataset",            crs_str.split(":")[-1] if ":" in crs_str else crs_str),
    ("DSM höjdspann",      f"{float(np.nanmin(dsm)):.1f} — {float(np.nanmax(dsm)):.1f} m"),
    ("DTM höjdspann",      f"{float(np.nanmin(dtm)):.1f} — {float(np.nanmax(dtm)):.1f} m"),
    ("Upplösning",         f"{pixel_m:.4f} m/px"),
    ("Medelutning",        f"{float(np.nanmean(valid_slope_flat)):.1f}°"),
    ("Maxlutning",         f"{float(np.nanmax(valid_slope_flat)):.1f}°"),
    ("Fördjupningar",      f"{len(valid_deps)} st (>= 50 m²)"),
    ("Punktmoln",          f"{n_pts:,} punkter"),
    ("GPU",                GPU_NAME if GPU_AVAILABLE else "CPU"),
]
y_pos = 0.95
for label, value in stats:
    ax.text(0.04, y_pos, label, transform=ax.transAxes,
            fontsize=8, color="#8b949e", va="top")
    ax.text(0.96, y_pos, value, transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="right", color=TEXT, va="top")
    y_pos -= 0.105

rect = mpatches.FancyBboxPatch(
    (0.01, 0.01), 0.98, 0.98,
    transform=ax.transAxes,
    boxstyle="round,pad=0.02",
    linewidth=1.5,
    edgecolor=ACCENT,
    facecolor="#0d1117",
    zorder=0
)
ax.add_patch(rect)

# ── Header ────────────────────────────────────────────────────────────────────
gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.005, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle("Steg 6 — Geodataanalys: Sammanfattning  |  Alla pipeline-steg",
             fontsize=16, fontweight="bold", color=TEXT, y=0.965)
fig.patch.set_facecolor(DARK)

out_path = OUT_FIGS / "06_final_report.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_path}")
print("\nSteg 6 klar.")
print("=" * 60)

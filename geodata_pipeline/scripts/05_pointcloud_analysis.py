#!/usr/bin/env python3
"""
Steg 5 — Punktmolnsanalys
Laddar .laz, skapar täthetskarta och höjdkarta (1 m raster).
GPU-accelererad binning med CuPy. Sparar 3D matplotlib scatter (headless).
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
import matplotlib.pyplot as plt
import laspy
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

print("=" * 60)
print("STEG 5 — PUNKTMOLNSANALYS")
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

RESOLUTION = 1.0    # m per cell for density / height map
MAX_3D_PTS = 200_000

CLASS_NAMES  = {0:"Oklass", 1:"Oklass2", 2:"Mark", 3:"Lågveg",
                4:"Medelveg", 5:"Högveg", 6:"Byggnad"}
CLASS_COLORS = {0:"#9E9E9E", 1:"#78909C", 2:"#795548", 3:"#C5E1A5",
                4:"#66BB6A", 5:"#1B5E20", 6:"#1565C0"}

# ── Load point cloud ──────────────────────────────────────────────────────────
laz_path = RAW / "georeferenced_model.laz"
print(f"\nLaddar {laz_path.name}  ({laz_path.stat().st_size/1e6:.1f} MB)...")
las = laspy.read(str(laz_path))

x   = las.x.scaled_array().astype(np.float32)
y   = las.y.scaled_array().astype(np.float32)
z   = las.z.scaled_array().astype(np.float32)
classification = np.array(las.classification, dtype=np.uint8)
n_pts = len(x)

print(f"  Antal punkter: {n_pts:,}")
print(f"  X:  {x.min():.2f} — {x.max():.2f} m  (spann {x.max()-x.min():.2f} m)")
print(f"  Y:  {y.min():.2f} — {y.max():.2f} m  (spann {y.max()-y.min():.2f} m)")
print(f"  Z:  {z.min():.2f} — {z.max():.2f} m  (spann {z.max()-z.min():.2f} m)")

unique_classes = np.unique(classification)
print("\n  Klassfördelning:")
for k in unique_classes:
    cnt = int((classification == k).sum())
    lbl = CLASS_NAMES.get(int(k), f"Klass {k}")
    print(f"    Klass {k} ({lbl}): {cnt:,}  ({100*cnt/n_pts:.1f}%)")

# ── Build density & height maps ───────────────────────────────────────────────
print(f"\nByggger täthetskarta och höjdkarta  ({RESOLUTION} m/cell)...")

x_min, x_max = float(x.min()), float(x.max())
y_min, y_max = float(y.min()), float(y.max())

cols = int(np.ceil((x_max - x_min) / RESOLUTION)) + 1
rows = int(np.ceil((y_max - y_min) / RESOLUTION)) + 1
print(f"  Raster: {cols} × {rows} celler")

if GPU_AVAILABLE:
    # GPU binning
    x_g = cp.array(x)
    y_g = cp.array(y)
    z_g = cp.array(z)

    xi_g = cp.clip(((x_g - x_min) / RESOLUTION).astype(cp.int32), 0, cols-1)
    yi_g = cp.clip(((y_max - y_g) / RESOLUTION).astype(cp.int32), 0, rows-1)

    # Flat index for 2D bincount
    flat_idx = yi_g * cols + xi_g
    density_flat = cp.bincount(flat_idx, minlength=rows*cols)
    density = cp.asnumpy(density_flat).reshape(rows, cols).astype(np.float32)

    # Height map: max Z per cell (CPU, scatter add GPU is complex)
    xi_cpu = cp.asnumpy(xi_g).astype(np.int32)
    yi_cpu = cp.asnumpy(yi_g).astype(np.int32)
    z_cpu  = cp.asnumpy(z_g)
else:
    xi_cpu = np.clip(((x - x_min) / RESOLUTION).astype(np.int32), 0, cols-1)
    yi_cpu = np.clip(((y_max - y) / RESOLUTION).astype(np.int32), 0, rows-1)
    z_cpu  = z

    flat_idx = (yi_cpu * cols + xi_cpu).astype(np.int64)
    density_flat = np.bincount(flat_idx, minlength=rows*cols)
    density = density_flat.reshape(rows, cols).astype(np.float32)

# Height map (max Z per cell) — CPU scatter
print("  Beräknar höjdkarta (max Z per cell)...", end=" ", flush=True)
height_map = np.full((rows, cols), np.nan, dtype=np.float32)
chunk = 500_000
for start in tqdm(range(0, n_pts, chunk), desc="  Z-binning", leave=False):
    end = min(start + chunk, n_pts)
    xi_c = xi_cpu[start:end]
    yi_c = yi_cpu[start:end]
    z_c  = z_cpu[start:end]
    # vectorised per-chunk update
    order = np.lexsort((z_c,))  # sort by z ascending
    xi_s, yi_s, z_s = xi_c[order], yi_c[order], z_c[order]
    height_map[yi_s, xi_s] = z_s   # last write (highest z) wins
print("OK")

# ── Figure: 3-panel ───────────────────────────────────────────────────────────
print("\nGenererar figur 05_pointcloud_analysis.png...")
fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor=DARK)

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title(title, color=TEXT, fontsize=10, pad=6)

def add_cbar(im, ax, label):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, color=TEXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=7)
    cb.outline.set_edgecolor(BORDER)

# Panel 1: Density map (log scale)
ax = axes[0]
styled_ax(ax, "Punkttäthetskarta  (log-skala)")
im = ax.imshow(np.log1p(density), cmap="inferno", interpolation="bilinear",
               aspect="auto")
add_cbar(im, ax, "log(täthet + 1)")

# Panel 2: Height map from point cloud
ax = axes[1]
styled_ax(ax, "Höjdkarta från punktmoln  (max Z per cell)")
im = ax.imshow(height_map, cmap="terrain", interpolation="bilinear", aspect="auto")
add_cbar(im, ax, "Höjd (m)")

# Panel 3: Classification bar chart
ax = axes[2]
ax.set_facecolor(PANEL)
counts = []
for k in unique_classes:
    cnt = int((classification == k).sum())
    if cnt > 0:
        lbl = CLASS_NAMES.get(int(k), f"Klass {k}")
        color = CLASS_COLORS.get(int(k), "#607D8B")
        counts.append((f"Klass {k}\n({lbl})", cnt, color))

names_  = [c[0] for c in counts]
vals_   = [c[1] for c in counts]
colors_ = [c[2] for c in counts]

bars = ax.barh(names_, vals_, color=colors_, edgecolor=DARK, height=0.6)
for bar, val in zip(bars, vals_):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2,
            f"{val/1000:.0f}k", va="center", fontsize=8, color=TEXT)
ax.set_xlabel("Antal punkter", color=TEXT, fontsize=9)
ax.set_title("Punktfördelning per klass", color=TEXT, fontsize=10, pad=6)
ax.tick_params(colors=TEXT, labelsize=8)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
for sp in ["top","right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom","left"]:
    ax.spines[sp].set_color(BORDER)

gpu_lbl = f"GPU: {GPU_NAME}" if GPU_AVAILABLE else "GPU: CPU fallback"
fig.text(0.99, 0.01, gpu_lbl, ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.suptitle(f"Steg 5 — Punktmolnsanalys  |  {n_pts:,} punkter  |  {RESOLUTION} m/cell",
             fontsize=14, fontweight="bold", color=TEXT, y=1.00)
fig.patch.set_facecolor(DARK)
plt.tight_layout()

out_main = OUT_FIGS / "05_pointcloud_analysis.png"
plt.savefig(out_main, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_main}")

# ── 3D scatter (matplotlib, headless) ────────────────────────────────────────
print("Genererar 3D scatter-figur (matplotlib)...")
if n_pts > MAX_3D_PTS:
    idx3d = np.random.choice(n_pts, MAX_3D_PTS, replace=False)
    x3, y3, z3, c3 = x[idx3d], y[idx3d], z[idx3d], classification[idx3d]
else:
    x3, y3, z3, c3 = x, y, z, classification

colors_rgb = np.array([
    [int(CLASS_COLORS.get(int(k), "#607D8B").lstrip("#")[i:i+2], 16)/255
     for i in (0, 2, 4)]
    for k in c3
])

fig3d = plt.figure(figsize=(12, 9), facecolor=DARK)
ax3d  = fig3d.add_subplot(111, projection="3d", facecolor=PANEL)
ax3d.scatter(x3, y3, z3, c=colors_rgb, s=0.5, alpha=0.6, rasterized=True)
ax3d.set_title(f"3D Punktmoln  ({len(x3):,} av {n_pts:,} punkter)",
               color=TEXT, fontsize=11, pad=8)
ax3d.tick_params(colors=TEXT, labelsize=6)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
# Build legend patches
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color=CLASS_COLORS.get(int(k), "#607D8B"),
                   label=f"Klass {k} — {CLASS_NAMES.get(int(k), '?')}")
    for k in unique_classes if (classification==k).sum() > 0
]
ax3d.legend(handles=legend_patches, loc="upper left",
            facecolor="#21262d", labelcolor=TEXT, fontsize=7,
            edgecolor=BORDER, framealpha=0.8)
fig3d.patch.set_facecolor(DARK)

out_3d = OUT_FIGS / "05_pointcloud_3d.png"
plt.savefig(out_3d, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_3d}")
print("\nSteg 5 klar.")
print("=" * 60)

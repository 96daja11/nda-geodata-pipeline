#!/usr/bin/env python3
"""
Steg 0 — Validera och inspektera ingångsdata
Kontrollerar att alla filer finns, CRS-match, nodata, upplösning och värdeintervall.
Visar GPU-info och sparar bounding-box-figur med DSM-thumbnail som bakgrund.
"""
import matplotlib
matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")

from pathlib import Path

# ── GPU setup ─────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    cp.array([1])
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    _vram_total = _dev.mem_info[1] // 1024**2
    _vram_free  = _dev.mem_info[0] // 1024**2
    import subprocess as _sp
    _smi = _sp.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"], capture_output=True, text=True)
    GPU_NAME = _smi.stdout.strip() if _smi.returncode == 0 else "RTX 3060"
    print(f"GPU: {GPU_NAME} | {_vram_total} MB total | {_vram_free} MB free")
except (ImportError, Exception) as _e:
    import numpy as cp
    GPU_AVAILABLE = False
    GPU_NAME = "CPU fallback"
    print(f"GPU: Not available ({_e})")

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import colormaps
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
print("STEG 0 — VALIDERA INGÅNGSDATA")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE is set above via --project-dir
DATA_RAW  = BASE / "data" / "raw"
OUT_FIGS  = BASE / "outputs" / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)

RASTER_FILES = {
    "DSM":        DATA_RAW / "dsm.tif",
    "DTM":        DATA_RAW / "dtm.tif",
    "Orthophoto": DATA_RAW / "odm_orthophoto.tif",
}
LAZ_FILE = DATA_RAW / "georeferenced_model.laz"

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
PALETTE = ["#2196F3", "#4CAF50", "#FF9800"]

# ── GPU info ──────────────────────────────────────────────────────────────────
print("\n--- GPU Information ---")
if GPU_AVAILABLE:
    _free2, _total2 = cp.cuda.Device(0).mem_info
    print(f"  CUDA available:   True")
    print(f"  GPU name:         {GPU_NAME}")
    print(f"  VRAM total:       {_total2 // 1024**2} MB")
    print(f"  VRAM free:        {_free2  // 1024**2} MB")
    print(f"  CuPy version:     {cp.__version__}")
else:
    print("  CUDA available:   False  |  Fallback: NumPy (CPU)")

# ── Collect raster metadata ────────────────────────────────────────────────────
print("\n--- Raster File Validation ---")
crs_registry = {}
meta_rows    = []
raster_data  = {}   # name → (data_downsampled, transform, bounds, color)
all_ok       = True

for (name, path), color in zip(RASTER_FILES.items(), PALETTE):
    if not path.exists():
        print(f"\n  [SAKNAS] {path}")
        all_ok = False
        continue

    with rasterio.open(path) as src:
        b       = src.bounds
        crs     = src.crs
        res     = src.res
        nodata  = src.nodata
        n_bands = src.count
        w, h    = src.width, src.height
        sz_mb   = path.stat().st_size / 1e6

        # Downsample for display (max 512px on longest side)
        scale = max(1, max(w, h) // 512)
        out_w, out_h = w // scale, h // scale

        raw = src.read(1, out_shape=(out_h, out_w),
                       resampling=rasterio.enums.Resampling.average).astype(np.float64)
        if nodata is not None:
            raw[raw == nodata] = np.nan

        # Full-res band 1 stats (using rasterio's overview or direct read)
        full = src.read(1).astype(np.float64)
        if nodata is not None:
            full[full == nodata] = np.nan

        valid_mask = ~np.isnan(full)
        valid_px   = int(valid_mask.sum())
        total_px   = full.size
        vmin  = float(np.nanmin(full))
        vmax  = float(np.nanmax(full))
        vmean = float(np.nanmean(full))
        vstd  = float(np.nanstd(full))

        print(f"\n  {'─'*50}")
        print(f"  {name}")
        print(f"  Fil:         {path.name}  ({sz_mb:.1f} MB)")
        print(f"  CRS:         {crs}")
        print(f"  Upplösning:  {res[0]:.4f} m/pixel")
        print(f"  Storlek:     {w} x {h} px  ({n_bands} band(er))")
        print(f"  Bounds:      W={b.left:.2f}  S={b.bottom:.2f}  E={b.right:.2f}  N={b.top:.2f}")
        print(f"  NoData:      {nodata}")
        print(f"  Giltiga px:  {valid_px:,} / {total_px:,}  ({100*valid_px/total_px:.1f}%)")
        if name != "Orthophoto":
            print(f"  Min:         {vmin:.3f} m")
            print(f"  Max:         {vmax:.3f} m")
            print(f"  Medel:       {vmean:.3f} m")
            print(f"  Std dev:     {vstd:.3f} m")

        crs_registry[name] = str(crs)
        meta_rows.append([name,
                          f"{res[0]:.4f}",
                          f"{w}×{h}",
                          f"{n_bands}",
                          f"{vmin:.2f}" if name != "Orthophoto" else "—",
                          f"{vmax:.2f}" if name != "Orthophoto" else "—",
                          f"{vmean:.2f}" if name != "Orthophoto" else "—"])
        raster_data[name] = (raw, b, color)

# ── CRS check ─────────────────────────────────────────────────────────────────
print("\n--- CRS Consistency Check ---")
unique_crs = set(crs_registry.values())
if len(unique_crs) == 1:
    print(f"  OK — alla rasters i samma CRS: {list(unique_crs)[0]}")
elif len(unique_crs) > 1:
    print("  VARNING: CRS mismatch!")
    for k, v in crs_registry.items():
        print(f"    {k}: {v}")
    all_ok = False

# ── LAZ validation ────────────────────────────────────────────────────────────
print("\n--- LAZ Point Cloud Validation ---")
laz_info = None
if not LAZ_FILE.exists():
    print(f"  [SAKNAS] {LAZ_FILE}")
    all_ok = False
else:
    las = laspy.read(str(LAZ_FILE))
    x   = las.x.scaled_array()
    y   = las.y.scaled_array()
    z   = las.z.scaled_array()
    classification = np.array(las.classification)
    n_pts  = len(x)
    sz_mb  = LAZ_FILE.stat().st_size / 1e6

    print(f"  Fil:           {LAZ_FILE.name}  ({sz_mb:.1f} MB)")
    print(f"  Antal punkter: {n_pts:,}")
    print(f"  X:  {x.min():.2f} — {x.max():.2f} m  (spann {x.max()-x.min():.2f} m)")
    print(f"  Y:  {y.min():.2f} — {y.max():.2f} m  (spann {y.max()-y.min():.2f} m)")
    print(f"  Z:  {z.min():.2f} — {z.max():.2f} m  (spann {z.max()-z.min():.2f} m)")

    class_names = {0:"Oklass",1:"Oklass2",2:"Mark",3:"Lågveg",4:"Medelveg",5:"Högveg",6:"Byggnad"}
    class_counts = {}
    for k in np.unique(classification):
        cnt  = int((classification == k).sum())
        name = class_names.get(int(k), f"Klass {k}")
        class_counts[name] = cnt
        print(f"    Klass {k} ({name}): {cnt:,}  ({100*cnt/n_pts:.1f}%)")

    laz_info = dict(x_min=x.min(), x_max=x.max(), y_min=y.min(), y_max=y.max(),
                    z_min=z.min(), z_max=z.max(), n_pts=n_pts, class_counts=class_counts)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: 2×2 grid
#   [0,0] DSM thumbnail with all bbox overlays
#   [0,1] DTM thumbnail
#   [1,0] Orthophoto RGB thumbnail
#   [1,1] Stats table + LAZ class bar
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenererar figur 00_input_validation.png...")

fig = plt.figure(figsize=(18, 13), facecolor=DARK)
gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                       top=0.91, bottom=0.05, left=0.05, right=0.97)

def style_ax(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=10, pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT, labelsize=7)

def add_bbox_overlay(ax, b, color, label, lw=2):
    """Draw a bounding-box rectangle on a pixel-coordinate axes."""
    rect = mpatches.Rectangle(
        (b.left, b.bottom), b.right - b.left, b.top - b.bottom,
        linewidth=lw, edgecolor=color, facecolor="none",
        linestyle="--", label=label, zorder=5
    )
    ax.add_patch(rect)

# ── Panel [0,0]: DSM thumbnail + all bounding boxes ───────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
style_ax(ax0, "DSM — Thumbnail med bounding boxes")

if "DSM" in raster_data:
    dsm_thumb, b_dsm, _ = raster_data["DSM"]
    norm = Normalize(vmin=np.nanpercentile(dsm_thumb, 2),
                     vmax=np.nanpercentile(dsm_thumb, 98))
    ax0.imshow(dsm_thumb, cmap="terrain", norm=norm, interpolation="bilinear",
               extent=[b_dsm.left, b_dsm.right, b_dsm.bottom, b_dsm.top],
               origin="upper", aspect="auto")

    # Overlay bounding boxes for all datasets
    for (name, path), color in zip(RASTER_FILES.items(), PALETTE):
        if name in raster_data:
            _, b, _ = raster_data[name]
            add_bbox_overlay(ax0, b, color, name, lw=2.5)
    if laz_info:
        class LazBounds:
            left   = laz_info["x_min"]
            right  = laz_info["x_max"]
            bottom = laz_info["y_min"]
            top    = laz_info["y_max"]
        add_bbox_overlay(ax0, LazBounds(), "#9C27B0", f"LAZ ({laz_info['n_pts']//1000}k pts)", lw=1.5)

    ax0.set_xlim(b_dsm.left - 5, b_dsm.right + 5)
    ax0.set_ylim(b_dsm.bottom - 5, b_dsm.top + 5)
    ax0.legend(loc="upper right", fontsize=7, facecolor="#21262d",
               labelcolor=TEXT, edgecolor=BORDER)
    ax0.set_xlabel("Easting (m)", color=TEXT, fontsize=8)
    ax0.set_ylabel("Northing (m)", color=TEXT, fontsize=8)

# ── Panel [0,1]: DTM thumbnail ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
style_ax(ax1, "DTM — Digital Terrain Model (markmodell)")

if "DTM" in raster_data:
    dtm_thumb, b_dtm, _ = raster_data["DTM"]
    norm_dtm = Normalize(vmin=np.nanpercentile(dtm_thumb, 2),
                         vmax=np.nanpercentile(dtm_thumb, 98))
    im1 = ax1.imshow(dtm_thumb, cmap="terrain", norm=norm_dtm, interpolation="bilinear",
                     extent=[b_dtm.left, b_dtm.right, b_dtm.bottom, b_dtm.top],
                     origin="upper", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label("Höjd (m)", color=TEXT, fontsize=7)
    cb1.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT, labelsize=6)
    cb1.outline.set_edgecolor(BORDER)
    ax1.set_xlabel("Easting (m)", color=TEXT, fontsize=8)
    ax1.set_ylabel("Northing (m)", color=TEXT, fontsize=8)

# ── Panel [1,0]: Orthophoto RGB ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Ortofoto — RGB (4989×4071 px, 0.05 m/px)")

ortho_path = RASTER_FILES.get("Orthophoto")
if ortho_path and ortho_path.exists():
    with rasterio.open(ortho_path) as src:
        scale = max(1, max(src.width, src.height) // 512)
        out_w = src.width  // scale
        out_h = src.height // scale
        r    = src.read(1, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average)
        g    = src.read(2, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average)
        b_ch = src.read(3, out_shape=(out_h, out_w),
                        resampling=rasterio.enums.Resampling.average)
        b_ext = src.bounds
    rgb = np.stack([r, g, b_ch], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb[rgb > 0], [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    ax2.imshow(rgb, extent=[b_ext.left, b_ext.right, b_ext.bottom, b_ext.top],
               origin="upper", aspect="auto", interpolation="bilinear")
    ax2.set_xlabel("Easting (m)", color=TEXT, fontsize=8)
    ax2.set_ylabel("Northing (m)", color=TEXT, fontsize=8)

# ── Panel [1,1]: Stats table + LAZ class bar ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(PANEL)
ax3.axis("off")
ax3.set_title("Filsammanfattning", color=TEXT, fontsize=10, pad=8)

if meta_rows:
    col_labels = ["Dataset", "m/px", "Storlek", "Band", "Min(m)", "Max(m)", "Medel(m)"]
    tbl = ax3.table(cellText=meta_rows, colLabels=col_labels,
                    loc="upper center", cellLoc="center",
                    bbox=[0.0, 0.45, 1.0, 0.52])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        else:
            cell.set_facecolor(PANEL if r % 2 == 0 else "#1c2128")
            cell.set_text_props(color=TEXT)
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.5)

# LAZ class bar chart in lower half of panel [1,1]
if laz_info and laz_info["class_counts"]:
    ax3b = fig.add_axes([ax3.get_position().x0 + 0.01,
                         ax3.get_position().y0 + 0.02,
                         ax3.get_position().width - 0.02,
                         ax3.get_position().height * 0.40])
    ax3b.set_facecolor(PANEL)
    class_colors = {"Mark": "#795548", "Lågveg": "#C5E1A5", "Medelveg": "#66BB6A",
                    "Högveg": "#1B5E20", "Byggnad": "#1565C0", "Oklass": "#9E9E9E",
                    "Oklass2": "#BDBDBD"}
    names = list(laz_info["class_counts"].keys())
    vals  = [laz_info["class_counts"][n] for n in names]
    total = sum(vals)
    colors_bar = [class_colors.get(n, "#607D8B") for n in names]
    bars = ax3b.barh(names, vals, color=colors_bar, edgecolor=DARK, height=0.6)
    for bar, v in zip(bars, vals):
        ax3b.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                  f"{v/1e6:.1f}M ({100*v/total:.0f}%)",
                  va="center", fontsize=7, color=TEXT)
    ax3b.set_xlim(0, max(vals) * 1.35)
    ax3b.set_xlabel(f"Antal punkter  (totalt {total/1e6:.1f}M)", color=TEXT, fontsize=7)
    ax3b.set_title("Punktmoln – klassfördelning", color=TEXT, fontsize=8, pad=4)
    ax3b.tick_params(colors=TEXT, labelsize=7)
    ax3b.set_facecolor(PANEL)
    for sp in ax3b.spines.values():
        sp.set_edgecolor(BORDER)
    ax3b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

# ── Suptitle and footer ───────────────────────────────────────────────────────
crs_str = list(crs_registry.values())[0] if crs_registry else "—"
fig.suptitle(
    f"Steg 0 — Validering av ingångsdata  |  CRS: {crs_str}  |  {'✓ Alla filer OK' if all_ok else '⚠ Problem detekterade'}",
    fontsize=13, fontweight="bold", color=TEXT, y=0.96
)
fig.text(0.99, 0.01, f"GPU: {GPU_NAME}", ha="right", va="bottom",
         fontsize=7, color="#586069", style="italic")
fig.patch.set_facecolor(DARK)

out_path = OUT_FIGS / "00_input_validation.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"  Figur sparad: {out_path}")

print("\n--- Sammanfattning ---")
print(f"  {'ALLA FILER OK — pipeline redo att köras.' if all_ok else 'PROBLEM — kontrollera saknade filer ovan.'}")
print("=" * 60)

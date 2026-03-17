#!/usr/bin/env python3
"""
3D Point Cloud Viewers — interaktiva HTML-filer med Plotly WebGL
Genererar separata vyar per segment för båda datasets:
  1. Hela klassificerade molnet (RGB + klasskolorer)
  2. Vegetation/träd (klass 3-5) — färgat per höjd
  3. Terränganomali (klass 2, mark) — färgat per lokal avvikelse
  4. Byggnader (klass 6) + markkontext
  5. Wietrznia: enskilda träd (watershed-segment, unikt ID per träd)
"""
import warnings; warnings.filterwarnings("ignore")
import json, math
from pathlib import Path

import numpy as np
import laspy
import plotly.graph_objects as go
import plotly.io as pio
from scipy.ndimage import uniform_filter

# ── Dataset definitions ────────────────────────────────────────────────────────
DATASETS = [
    {
        "name":    "Güterweg Ritzing",
        "slug":    "guterweg",
        "base":    Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata"),
        "laz":     Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata/data/raw/georeferenced_model.laz"),
        "dep_json": Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata/outputs/reports/03_depressions.json"),
        "tree_json": None,
        "chm_tif": None,
    },
    {
        "name":    "Wietrznia",
        "slug":    "wietrznia",
        "base":    Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia"),
        "laz":     Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia/data/raw/georeferenced_model.laz"),
        "dep_json": Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia/outputs/reports/03_depressions.json"),
        "tree_json": Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia/outputs/reports/05b_tree_analysis.json"),
        "chm_tif": Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia/data/derived/chm.tif"),
    },
]

OUT_BASE = Path("/mnt/storage4tb/smarttek-demo/outputs/3d_viewers")
OUT_BASE.mkdir(parents=True, exist_ok=True)

MAX_PTS_VIEW = 400_000   # max points per viewer (WebGL performance)

CLASS_COLORS = {
    1: (150, 150, 150),   # Oklass — grå
    2: (120,  85,  60),   # Mark — brun
    3: (100, 200,  80),   # Lågveg — grön
    4: ( 50, 160,  50),   # Medelveg — mörkgrön
    5: ( 20, 100,  20),   # Högveg — skogsgrönt
    6: ( 70, 130, 200),   # Byggnad — blå
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_laz(path):
    print(f"  Laddar {Path(path).name} ...", end=" ", flush=True)
    las = laspy.read(str(path))
    x   = las.x.scaled_array().astype(np.float32)
    y   = las.y.scaled_array().astype(np.float32)
    z   = las.z.scaled_array().astype(np.float32)
    cls = np.array(las.classification, dtype=np.uint8)
    try:
        r = np.array(las.red,   dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue,  dtype=np.float32)
        # Normalise: ODM typically stores 16-bit (0-65535) or 8-bit (0-255)
        if r.max() > 256:
            r, g, b = r / 65535, g / 65535, b / 65535
        else:
            r, g, b = r / 255,   g / 255,   b / 255
        has_rgb = True
    except Exception:
        has_rgb = False
        r = g = b = None
    print(f"OK  ({len(x):,} pts)")
    return x, y, z, cls, (r, g, b) if has_rgb else (None, None, None)

def subsample(mask, n_max, *arrays):
    """Random subsample of points selected by mask, capped at n_max."""
    idx = np.where(mask)[0]
    if len(idx) > n_max:
        idx = np.random.choice(idx, n_max, replace=False)
        idx.sort()
    return tuple(a[idx] for a in arrays)

def height_colorscale(z_arr, cmap_name="Viridis"):
    """Map z values to hex colors using a matplotlib colormap."""
    import matplotlib.pyplot as plt
    cm   = plt.get_cmap(cmap_name)
    zmin, zmax = z_arr.min(), z_arr.max()
    norm = (z_arr - zmin) / max(zmax - zmin, 0.001)
    rgba = cm(norm)
    return [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in rgba]

def rgb_to_hex_list(r, g, b):
    return [f"rgb({int(ri*255)},{int(gi*255)},{int(bi*255)})" for ri, gi, bi in zip(r, g, b)]

def anomaly_colors(dev_arr):
    """Blue→white→red diverging for deviation values."""
    import matplotlib.pyplot as plt
    cm   = plt.get_cmap("RdBu_r")
    lim  = max(abs(dev_arr.min()), abs(dev_arr.max()), 0.001)
    norm = (dev_arr + lim) / (2 * lim)
    norm = np.clip(norm, 0, 1)
    rgba = cm(norm)
    return [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b, _ in rgba]

def make_scatter(x, y, z, colors, name, size=1.5, opacity=1.0):
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=size, color=colors, opacity=opacity),
        name=name,
        hovertemplate=f"<b>{name}</b><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Z: %{{z:.2f}}<extra></extra>",
    )

LAYOUT_BASE = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="DM Mono, monospace", size=11),
    margin=dict(l=0, r=0, t=50, b=0),
    legend=dict(
        bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
        font=dict(size=10, color="#e6edf3"),
        x=0.01, y=0.99,
    ),
    scene=dict(
        bgcolor="#0d1117",
        xaxis=dict(showgrid=True, gridcolor="#21262d", zerolinecolor="#30363d",
                   tickfont=dict(color="#8b949e", size=8), title=""),
        yaxis=dict(showgrid=True, gridcolor="#21262d", zerolinecolor="#30363d",
                   tickfont=dict(color="#8b949e", size=8), title=""),
        zaxis=dict(showgrid=True, gridcolor="#21262d", zerolinecolor="#30363d",
                   tickfont=dict(color="#8b949e", size=8), title="Höjd (m)"),
        aspectmode="data",
        camera=dict(eye=dict(x=1.3, y=-1.3, z=0.9)),
    ),
)

def save_html(fig, path, title):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color="#58a6ff"),
                                  x=0.5, xanchor="center"))
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn",
                       config=dict(displayModeBar=True, scrollZoom=True))
    path.write_text(html, encoding="utf-8")
    print(f"  → {path.name}  ({path.stat().st_size/1e6:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("3D POINT CLOUD VIEWERS — Plotly WebGL Interactive")
print("=" * 65)

np.random.seed(42)

for ds in DATASETS:
    name  = ds["name"]
    slug  = ds["slug"]
    out   = OUT_BASE / slug
    out.mkdir(exist_ok=True)

    print(f"\n{'─'*65}")
    print(f"  Dataset: {name}")
    print(f"{'─'*65}")

    x, y, z, cls, (r, g, b) = load_laz(ds["laz"])
    has_rgb = r is not None

    # Centre coordinates for nicer axis labels
    x0, y0 = x.mean(), y.mean()
    xc, yc = (x - x0).astype(np.float32), (y - y0).astype(np.float32)

    # ── 1. Full classified cloud ───────────────────────────────────────────────
    print("\n  [1/5] Full klassificerat molnet...")
    traces = []
    class_labels = {1:"Oklassificerad",2:"Mark",3:"Lågvegetation",
                    4:"Medelvegetation",5:"Högvegetation",6:"Byggnad"}
    pts_per_class = MAX_PTS_VIEW // max(len(np.unique(cls)), 1)

    for k in sorted(np.unique(cls)):
        mask = cls == k
        n_cls = mask.sum()
        n_samp = min(n_cls, pts_per_class)
        idx = np.where(mask)[0]
        if len(idx) > n_samp:
            idx = np.random.choice(idx, n_samp, replace=False)

        if has_rgb:
            cols = rgb_to_hex_list(r[idx], g[idx], b[idx])
        else:
            cc = CLASS_COLORS.get(int(k), (180,180,180))
            cols = f"rgb{cc}"

        lbl = class_labels.get(int(k), f"Klass {k}")
        traces.append(make_scatter(xc[idx], yc[idx], z[idx], cols,
                                   f"{lbl} ({n_cls//1000}k pts)", size=1.2))

    fig = go.Figure(traces)
    fig.update_layout(**LAYOUT_BASE)
    save_html(fig, out / "01_full_klassificerat.html",
              f"{name} — Fullt klassificerat punktmoln (RGB + LAS-klasser)")

    # ── 2. Vegetation / trees only ─────────────────────────────────────────────
    print("\n  [2/5] Vegetation / träd...")
    veg_mask = (cls >= 3) & (cls <= 5)
    n_veg = veg_mask.sum()
    xi, yi, zi, cli = subsample(veg_mask, MAX_PTS_VIEW, xc, yc, z, cls)

    # colour by height within vegetation
    cols_h = height_colorscale(zi, "YlGn")

    # also add ground as thin grey context layer
    gnd_mask = cls == 2
    n_gnd_samp = min(gnd_mask.sum(), 80_000)
    gnd_idx = np.random.choice(np.where(gnd_mask)[0], n_gnd_samp, replace=False)

    fig = go.Figure([
        make_scatter(xc[gnd_idx], yc[gnd_idx], z[gnd_idx],
                     "rgb(80,60,45)", "Mark (kontext)", size=0.8, opacity=0.3),
        make_scatter(xi, yi, zi, cols_h,
                     f"Vegetation ({n_veg//1000}k pts) — färg=höjd", size=2.0),
    ])
    fig.update_layout(**LAYOUT_BASE)
    save_html(fig, out / "02_vegetation_trad.html",
              f"{name} — Vegetation & träd (färgat per höjd)")

    # ── 3. Terrain anomaly (ground pts coloured by local deviation) ────────────
    print("\n  [3/5] Terränganomali (markpunkter per avvikelse)...")
    gnd_mask2 = cls == 2
    n_gnd = gnd_mask2.sum()
    n_samp_gnd = min(n_gnd, MAX_PTS_VIEW)
    gnd_all_idx = np.where(gnd_mask2)[0]
    gnd_idx2 = np.random.choice(gnd_all_idx, n_samp_gnd, replace=False)
    gnd_idx2.sort()

    xg, yg, zg = xc[gnd_idx2], yc[gnd_idx2], z[gnd_idx2]

    # Fast local deviation: bin into 1m grid, compute mean, subtract
    xr, yr = x[gnd_idx2], y[gnd_idx2]
    cell  = 5.0   # 5m neighbourhood
    xi_b  = ((xr - xr.min()) / cell).astype(int)
    yi_b  = ((yr - yr.min()) / cell).astype(int)
    cols_b = max(xi_b.max() + 1, 1)
    rows_b = max(yi_b.max() + 1, 1)
    grid_sum   = np.zeros((rows_b, cols_b), dtype=np.float64)
    grid_cnt   = np.zeros((rows_b, cols_b), dtype=np.int32)
    np.add.at(grid_sum, (yi_b, xi_b), zg)
    np.add.at(grid_cnt, (yi_b, xi_b), 1)
    grid_mean  = np.where(grid_cnt > 0, grid_sum / np.maximum(grid_cnt, 1), 0)
    # Smooth the grid
    from scipy.ndimage import gaussian_filter as gf
    grid_mean_s = gf(grid_mean, sigma=2)
    local_mean  = grid_mean_s[yi_b, xi_b]
    deviation   = (zg - local_mean).astype(np.float32)

    cols_dev = anomaly_colors(deviation)

    # Load depressions JSON for annotation markers
    dep_markers = []
    if ds["dep_json"] and ds["dep_json"].exists():
        with open(ds["dep_json"]) as f:
            dep_raw = json.load(f)
        deps = sorted(dep_raw.get("depressions", []),
                      key=lambda d: -d.get("max_depth_m", 0))[:10]
        # We don't have pixel→world transform here, skip centroid pins

    fig = go.Figure([
        make_scatter(xg, yg, zg, cols_dev,
                     f"Mark — lokal avvikelse ({n_samp_gnd//1000}k pts)<br>"
                     "Blå=fördjupning  Röd=upphöjning", size=1.8),
    ])
    # Add colorbar via a dummy scatter with colorscale
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers",
        marker=dict(color=[deviation.min(), 0, deviation.max()],
                    colorscale="RdBu", showscale=True,
                    colorbar=dict(title=dict(text="Avvikelse (m)",
                                              font=dict(color="#e6edf3")),
                                  x=1.02,
                                  tickfont=dict(color="#e6edf3"))),
        showlegend=False,
    ))
    fig.update_layout(**LAYOUT_BASE)
    save_html(fig, out / "03_terrang_anomali.html",
              f"{name} — Terränganomali: lokal höjdavvikelse (blå=grop, röd=upphöjning)")

    # ── 4. Buildings ───────────────────────────────────────────────────────────
    print("\n  [4/5] Byggnader...")
    bld_mask = cls == 6
    n_bld = bld_mask.sum()

    traces_bld = []
    if n_bld > 0:
        xi_b2, yi_b2, zi_b2 = subsample(bld_mask, 120_000, xc, yc, z)
        # Height-coloured buildings
        bld_cols = height_colorscale(zi_b2, "Blues")
        traces_bld.append(make_scatter(xi_b2, yi_b2, zi_b2, bld_cols,
                                       f"Byggnader ({n_bld//1000}k pts)", size=2.0))

    # Ground context
    gnd_idx3 = np.random.choice(np.where(gnd_mask2)[0],
                                 min(gnd_mask2.sum(), 100_000), replace=False)
    if has_rgb:
        gnd_c = rgb_to_hex_list(r[gnd_idx3], g[gnd_idx3], b[gnd_idx3])
    else:
        gnd_c = "rgb(100,75,55)"
    traces_bld.insert(0, make_scatter(xc[gnd_idx3], yc[gnd_idx3], z[gnd_idx3],
                                       gnd_c, "Mark (kontext)", size=0.8, opacity=0.35))
    # Vegetation thin context
    if veg_mask.sum() > 0:
        veg_idx3 = np.random.choice(np.where(veg_mask)[0],
                                     min(veg_mask.sum(), 60_000), replace=False)
        traces_bld.append(make_scatter(xc[veg_idx3], yc[veg_idx3], z[veg_idx3],
                                        "rgb(60,130,60)", "Vegetation (kontext)",
                                        size=0.8, opacity=0.25))

    fig = go.Figure(traces_bld)
    fig.update_layout(**LAYOUT_BASE)
    save_html(fig, out / "04_byggnader.html",
              f"{name} — Byggnader isolerade (blå gradient=höjd)")

    # ── 5. Dataset-specific extra view ────────────────────────────────────────
    print(f"\n  [5/5] Specialvy ({name})...")

    if slug == "wietrznia" and ds["chm_tif"]:
        # Individual tree crowns from watershed segmentation
        # Recompute quickly from CHM + LAZ vegetation
        try:
            import rasterio
            from scipy.ndimage import gaussian_filter
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed

            with rasterio.open(ds["chm_tif"]) as src:
                chm      = src.read(1).astype(np.float32)
                chm_tf   = src.transform
                chm_crs  = src.crs

            chm[chm < 0] = 0
            chm_s = gaussian_filter(chm, sigma=3.0)
            veg_m = chm > 1.5

            peaks   = peak_local_max(chm_s, min_distance=100,
                                      threshold_abs=3.0, labels=veg_m.astype(np.uint8))
            markers = np.zeros(chm.shape, dtype=np.int32)
            for i, (r2, c2) in enumerate(peaks):
                markers[r2, c2] = i + 1
            labels  = watershed(-chm_s, markers, mask=veg_m)

            n_ids = labels.max()
            print(f"    Träd detekterade: {n_ids}")

            # Assign each vegetation LAZ point a tree ID via pixel lookup
            veg_pts = veg_mask  # cls 3-5
            n_veg_pts = veg_pts.sum()
            max_veg = min(n_veg_pts, MAX_PTS_VIEW)
            vi = np.random.choice(np.where(veg_pts)[0], max_veg, replace=False)
            vi.sort()

            xv, yv, zv = x[vi], y[vi], z[vi]

            # pixel coords
            inv_tf = ~chm_tf
            col_f, row_f = inv_tf * (xv, yv)
            row_i = np.clip(row_f.astype(int), 0, chm.shape[0]-1)
            col_i = np.clip(col_f.astype(int), 0, chm.shape[1]-1)
            tree_id = labels[row_i, col_i]   # 0 = unassigned

            # Colour by tree ID (categorical, cycle through palette)
            import matplotlib.pyplot as plt
            n_colors = max(n_ids, 1)
            cmap_trees = plt.get_cmap("tab20", n_colors)
            cols_tree = []
            for tid in tree_id:
                if tid == 0:
                    cols_tree.append("rgb(40,40,40)")
                else:
                    rgba = cmap_trees((tid - 1) % n_colors)
                    cols_tree.append(f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})")

            # Add ground for context
            gnd_i2 = np.random.choice(np.where(gnd_mask2)[0],
                                       min(gnd_mask2.sum(), 80_000), replace=False)
            fig = go.Figure([
                make_scatter((x[gnd_i2]-x0), (y[gnd_i2]-y0), z[gnd_i2],
                              "rgb(70,55,45)", "Mark", size=0.8, opacity=0.3),
                make_scatter((xv-x0), (yv-y0), zv, cols_tree,
                              f"{n_ids} individuella trädkronor", size=2.0),
            ])
            fig.update_layout(**LAYOUT_BASE)
            save_html(fig, out / "05_individuella_trad.html",
                      f"{name} — {n_ids} individuella trädkronor (varje krona = unik färg)")
        except Exception as e:
            print(f"    Trädvy fel: {e}")

    elif slug == "guterweg":
        # Depression depth view — ground coloured by depression depth
        # Reload ground points
        gnd_all = np.where(gnd_mask2)[0]
        n_g = min(len(gnd_all), MAX_PTS_VIEW)
        gi  = np.random.choice(gnd_all, n_g, replace=False); gi.sort()

        xgi, ygi, zgi = x[gi], y[gi], z[gi]

        # Local deviation (same as above but finer grid 1m)
        xi_g  = ((xgi - xgi.min()) / 1.0).astype(int)
        yi_g  = ((ygi - ygi.min()) / 1.0).astype(int)
        cb, rb = max(xi_g.max()+1,1), max(yi_g.max()+1,1)
        gs2, gc2 = np.zeros((rb,cb)), np.zeros((rb,cb),dtype=int)
        np.add.at(gs2, (yi_g, xi_g), zgi)
        np.add.at(gc2, (yi_g, xi_g), 1)
        gm2 = np.where(gc2>0, gs2/np.maximum(gc2,1), 0)
        from scipy.ndimage import gaussian_filter as gf2
        gm2s = gf2(gm2, sigma=5)
        lm2  = gm2s[yi_g, xi_g]
        dev2 = (zgi - lm2).astype(np.float32)

        # Separate traces: deep depressions, flat, high
        deep  = dev2 < -0.3
        high  = dev2 >  0.3
        flat  = (~deep) & (~high)

        fig = go.Figure([
            make_scatter((xgi[flat]-x0),  (ygi[flat]-y0),  zgi[flat],
                          "rgb(120,95,70)", "Plant mark", size=1.2, opacity=0.5),
            make_scatter((xgi[high]-x0),  (ygi[high]-y0),  zgi[high],
                          "rgb(220,100,40)", f"Upphöjningar (>{0.3}m)", size=2.0),
            make_scatter((xgi[deep]-x0),  (ygi[deep]-y0),  zgi[deep],
                          "rgb(40,100,220)", f"Fördjupningar (<{-0.3}m)", size=2.0),
        ])
        # Add buildings on top
        if n_bld > 0:
            bld_i3 = np.random.choice(np.where(bld_mask)[0],
                                       min(n_bld, 60_000), replace=False)
            fig.add_trace(make_scatter((x[bld_i3]-x0), (y[bld_i3]-y0), z[bld_i3],
                                        "rgb(200,220,255)", "Byggnader", size=2.5))

        fig.update_layout(**LAYOUT_BASE)
        save_html(fig, out / "05_fordjupningar_byggnader.html",
                  f"{name} — Fördjupningar (blå) & upphöjningar (orange) + byggnader (vit)")

# ── Index HTML ─────────────────────────────────────────────────────────────────
print("\nGenererar index.html...")

index_html = """<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8"/>
<title>3D Pointcloud Viewers</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@300;400;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'Instrument Sans',sans-serif;padding:30px}
  h1{font-family:'DM Mono',monospace;color:#58a6ff;font-size:1.3rem;letter-spacing:.1em;margin-bottom:6px}
  .sub{color:#8b949e;font-size:.85rem;margin-bottom:30px}
  .ds{margin-bottom:40px}
  h2{font-size:1rem;color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:8px;margin-bottom:16px}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;text-decoration:none;
        color:#e6edf3;transition:border-color .2s,background .2s}
  .card:hover{border-color:#58a6ff;background:#1c2128}
  .card-icon{font-size:1.6rem;margin-bottom:8px}
  .card-title{font-size:.9rem;font-weight:600;margin-bottom:4px;color:#58a6ff}
  .card-desc{font-size:.78rem;color:#8b949e;line-height:1.4}
  footer{margin-top:40px;color:#30363d;font-family:'DM Mono',monospace;font-size:.7rem}
</style>
</head>
<body>
<h1>3D POINT CLOUD VIEWERS</h1>
<p class="sub">Interaktiva WebGL-visualiseringar · Roteras med vänsterklick · Zooma med scroll · Panorera med högerklick</p>
"""

viewer_meta = {
    "01_full_klassificerat.html": ("🗂️", "Fullt klassificerat moln", "Alla punkter färgade med RGB-ortofoto och LAS-klassöverlager"),
    "02_vegetation_trad.html":    ("🌲", "Vegetation & träd", "Klass 3–5 isolerad, färgad per höjd (gul→grön→mörk)"),
    "03_terrang_anomali.html":    ("🌊", "Terränganomali", "Markpunkter färgade per lokal höjdavvikelse · Blå=fördjupning · Röd=upphöjning"),
    "04_byggnader.html":          ("🏠", "Byggnader isolerade", "LAS-klass 6 med höjdgradient · Mark och vegetation som kontext"),
    "05_individuella_trad.html":  ("🌳", "Individuella trädkronor", "Varje träd en unik färg · Watershed-segmentering av CHM"),
    "05_fordjupningar_byggnader.html": ("🕳️", "Fördjupningar & upphöjningar", "Blå=fördjupning, orange=upphöjning, vit=byggnader"),
}

for ds in DATASETS:
    slug = ds["slug"]
    index_html += f'<div class="ds"><h2>{ds["name"]}</h2><div class="cards">'
    d = OUT_BASE / slug
    for fname, (icon, title, desc) in viewer_meta.items():
        fp = d / fname
        if fp.exists():
            index_html += f'<a class="card" href="{slug}/{fname}" target="_blank"><div class="card-icon">{icon}</div><div class="card-title">{title}</div><div class="card-desc">{desc}</div></a>'
    index_html += "</div></div>"

index_html += "<footer>Geodata Analysis Pipeline · 3D Viewers · WebGL via Plotly</footer></body></html>"

(OUT_BASE / "index.html").write_text(index_html, encoding="utf-8")
print(f"  → index.html")

print("\n" + "="*65)
print(f"  KLART! Alla viewers sparade i:")
print(f"  {OUT_BASE}")
print(f"  Öppna: {OUT_BASE}/index.html")
print("="*65)

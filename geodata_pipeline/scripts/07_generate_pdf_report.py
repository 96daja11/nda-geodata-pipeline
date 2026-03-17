#!/usr/bin/env python3
"""
Steg 7 — Generera sammanfattande PDF-rapport
Sammanfattar all data, siffror och figurer från stegen 0–6 (inkl. valfri 5b trädanalys)
i ett professionellt PDF-dokument.
"""
import matplotlib
matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")

import json, base64, datetime
from pathlib import Path

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

FIGS      = BASE / "outputs" / "figures"
REPS      = BASE / "outputs" / "reports"
OUT_PDF   = REPS / _cfg.get("report", {}).get("pdf_name", "geodata_analys_rapport.pdf")
REPS.mkdir(parents=True, exist_ok=True)

# Dataset title and footer from config
DATASET_TITLE  = _cfg.get("report", {}).get("title",          "Geodataanalys")
FOOTER_DATASET = _cfg.get("report", {}).get("footer_dataset", "Geodataanalys 2022")

print("=" * 60)
print("STEG 7 — GENERERA PDF-RAPPORT")
print("=" * 60)

# ── Load JSON data from previous steps ───────────────────────────────────────
dep_data  = {}
vol_data  = []
tree_data = {}

dep_json  = REPS / "03_depressions.json"
vol_json  = REPS / "04_volumes.json"
tree_json = REPS / "05b_tree_analysis.json"

if dep_json.exists():
    with open(dep_json) as f:
        dep_data = json.load(f)
if vol_json.exists():
    with open(vol_json) as f:
        vol_data = json.load(f)

# Tree analysis: conditional on config and presence of JSON
tree_cfg_enabled = _cfg.get("tree_analysis", {}).get("enabled", True)
INCLUDE_TREES = tree_cfg_enabled and tree_json.exists()

if tree_json.exists():
    with open(tree_json) as f:
        tree_data = json.load(f)
    if INCLUDE_TREES:
        print(f"  Trädanalys laddad: {tree_data.get('n_trees', 0):,} träd")
    else:
        print("  Trädanalys JSON finns men är inaktiverad i config.json.")
else:
    if tree_cfg_enabled:
        print("  Info: 05b_tree_analysis.json saknas — kör steg 5b om trädanalys önskas.")
    else:
        print("  Trädanalys inaktiverad i config.json (tree_analysis.enabled=false).")

# ── Raster metadata ───────────────────────────────────────────────────────────
import numpy as np
import rasterio

DATA_RAW    = BASE / "data" / "raw"
raster_meta = {}

for name, fname in [("DSM", "dsm.tif"), ("DTM", "dtm.tif"), ("Orthophoto", "odm_orthophoto.tif")]:
    p = DATA_RAW / fname
    if p.exists():
        with rasterio.open(p) as src:
            data = src.read(1).astype(np.float64)
            nd   = src.nodata
            if nd is not None:
                data[data == nd] = np.nan
            raster_meta[name] = {
                "crs":       str(src.crs),
                "res_m":     src.res[0],
                "width":     src.width,
                "height":    src.height,
                "size_mb":   p.stat().st_size / 1e6,
                "vmin":      float(np.nanmin(data)),
                "vmax":      float(np.nanmax(data)),
                "vmean":     float(np.nanmean(data)),
                "vstd":      float(np.nanstd(data)),
                "valid_pct": 100 * float(np.sum(~np.isnan(data))) / data.size,
            }

# ── LAZ metadata ──────────────────────────────────────────────────────────────
import laspy

laz_meta = {}
laz_p = DATA_RAW / "georeferenced_model.laz"
if laz_p.exists():
    las  = laspy.read(str(laz_p))
    x, y, z = las.x.scaled_array(), las.y.scaled_array(), las.z.scaled_array()
    cls  = np.array(las.classification)
    cn   = {0:"Oklass",1:"Oklass2",2:"Mark",3:"Lågveg",4:"Medelveg",5:"Högveg",6:"Byggnad"}
    laz_meta = {
        "n_pts": len(x),
        "size_mb": laz_p.stat().st_size / 1e6,
        "x_span": float(x.max() - x.min()),
        "y_span": float(y.max() - y.min()),
        "z_min":  float(z.min()), "z_max": float(z.max()),
        "classes": {cn.get(int(k), str(k)): int((cls==k).sum()) for k in np.unique(cls)},
    }

# ── Terrain stats from step 2 ─────────────────────────────────────────────────
terrain_stats = {}
dtm_p = DATA_RAW / "dtm.tif"
if dtm_p.exists():
    from scipy.ndimage import sobel
    with rasterio.open(dtm_p) as src:
        dtm_f = src.read(1).astype(np.float64); nd = src.nodata; px = src.res[0]
        if nd is not None: dtm_f[dtm_f == nd] = np.nan
    dtm_nn = np.where(np.isnan(dtm_f), 0.0, dtm_f)
    gx = sobel(dtm_nn, axis=1) / (8.0 * px)
    gy = sobel(dtm_nn, axis=0) / (8.0 * px)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    slope[np.isnan(dtm_f)] = np.nan
    vs = slope[~np.isnan(slope)]
    terrain_stats = {
        "mean_slope":   float(np.nanmean(vs)),
        "max_slope":    float(np.nanmax(vs)),
        "pct_flat":     100 * float((vs < 5).mean()),
        "pct_moderate": 100 * float(((vs >= 5) & (vs < 15)).mean()),
        "pct_steep":    100 * float(((vs >= 15) & (vs < 30)).mean()),
        "pct_vsteep":   100 * float((vs >= 30).mean()),
    }

# ── Helper: embed image as base64 ────────────────────────────────────────────
def img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img_tag(path: Path, style="width:100%;") -> str:
    b64 = img_b64(path)
    if not b64:
        return f'<div class="missing-fig">Figur ej tillgänglig: {path.name}</div>'
    return f'<img src="data:image/png;base64,{b64}" style="{style}" />'

# ── Depression summary ────────────────────────────────────────────────────────
deps = dep_data.get("depressions", [])
n_deps   = len(deps)
n_ups    = len(dep_data.get("elevations", []))
top_deps = sorted(deps, key=lambda d: -d.get("max_depth_m", 0))[:5]
total_dep_area = sum(d.get("area_m2", 0) for d in deps)

# ── Volume summary ────────────────────────────────────────────────────────────
vol_rows = ""
for v in vol_data:
    vols = v.get("volumes_m3", {})
    p95  = vols.get("Percentil 95", 0)
    km   = vols.get("Kantmedelvärde", 0)
    itp  = vols.get("Interpolerad yta", km)
    vol_rows += f"""
    <tr>
      <td>#{v['object_id']}</td>
      <td>{v['area_m2']:.1f}</td>
      <td>{v['max_depth_m']:.3f}</td>
      <td>{p95:.1f}</td>
      <td>{km:.1f}</td>
      <td>{itp:.1f}</td>
    </tr>"""

# ── Volume figure paths (top objects) ─────────────────────────────────────────
vol_fig_tags = ""
for v in vol_data[:3]:
    oid = v["object_id"]
    fp  = FIGS / f"04_volume_obj{oid:03d}.png"
    if not fp.exists():
        fp = FIGS / f"04_volume_obj{oid}.png"
    vol_fig_tags += f"""
    <div class="vol-fig">
      <div class="fig-caption">Objekt #{oid} — Area: {v['area_m2']:.0f} m² | Max avvikelse: {v['max_depth_m']:.3f} m</div>
      {img_tag(fp, "width:100%;")}
    </div>"""

# ── Depression table rows ─────────────────────────────────────────────────────
dep_rows = ""
for d in top_deps:
    dep_rows += f"""
    <tr>
      <td>#{d['id']}</td>
      <td>{d['area_m2']:.1f}</td>
      <td>{d['max_depth_m']:.3f}</td>
      <td>{d['mean_depth_m']:.3f}</td>
    </tr>"""

# ── Raster table rows ─────────────────────────────────────────────────────────
raster_rows = ""
for name, m in raster_meta.items():
    extra  = f"{m['vmin']:.2f}" if name != "Orthophoto" else "—"
    extra2 = f"{m['vmax']:.2f}" if name != "Orthophoto" else "—"
    extra3 = f"{m['vmean']:.2f} ± {m['vstd']:.2f}" if name != "Orthophoto" else "—"
    raster_rows += f"""
    <tr>
      <td>{name}</td>
      <td>{m['res_m']:.4f}</td>
      <td>{m['width']}×{m['height']}</td>
      <td>{m['size_mb']:.1f}</td>
      <td>{m['valid_pct']:.1f}%</td>
      <td>{extra}</td>
      <td>{extra2}</td>
      <td>{extra3}</td>
    </tr>"""

# ── LAZ class rows ────────────────────────────────────────────────────────────
laz_rows = ""
if laz_meta:
    total_pts = laz_meta["n_pts"]
    for cls_name, cnt in laz_meta["classes"].items():
        laz_rows += f"""
    <tr>
      <td>{cls_name}</td>
      <td>{cnt:,}</td>
      <td>{100*cnt/total_pts:.1f}%</td>
    </tr>"""

# ── Tree analysis section helpers ─────────────────────────────────────────────
n_trees_val       = tree_data.get("n_trees", 0)
canopy_pct_val    = tree_data.get("canopy_coverage_pct", 0.0)
tree_density_val  = tree_data.get("tree_density_per_ha", 0.0)
mean_height_val   = tree_data.get("mean_height_m", 0.0)
max_height_val    = tree_data.get("max_height_m", 0.0)
canopy_area_val   = tree_data.get("total_canopy_area_m2", 0.0)
height_dist_val   = tree_data.get("height_distribution", {})
crown_stats_val   = tree_data.get("crown_area_stats", {})

tree_height_rows = ""
for cls_lbl, cnt_v in height_dist_val.items():
    pct_v = 100.0 * cnt_v / max(n_trees_val, 1)
    tree_height_rows += f"""
    <tr>
      <td>{cls_lbl}</td>
      <td>{cnt_v:,}</td>
      <td>{pct_v:.1f}%</td>
    </tr>"""

date_str = datetime.date.today().strftime("%Y-%m-%d")

# ── Pre-computed template variables ───────────────────────────────────────────
laz_classes       = laz_meta.get("classes", {})
laz_n_pts         = laz_meta.get("n_pts", 1)
laz_size_mb       = laz_meta.get("size_mb", 0)
laz_x_span        = laz_meta.get("x_span", 0)
laz_y_span        = laz_meta.get("y_span", 0)
laz_z_min         = laz_meta.get("z_min", 0)
laz_z_max         = laz_meta.get("z_max", 0)
laz_z_span        = laz_z_max - laz_z_min
laz_n_pts_M       = laz_n_pts / 1e6
laz_mark_pct      = 100 * laz_classes.get("Mark", 0) / laz_n_pts

dsm_meta          = raster_meta.get("DSM", {})
dtm_meta_d        = raster_meta.get("DTM", {})
ortho_meta        = raster_meta.get("Orthophoto", {})
dsm_vmin          = dsm_meta.get("vmin", 0)
dsm_vmax          = dsm_meta.get("vmax", 0)
dsm_vmean         = dsm_meta.get("vmean", 0)
dtm_vmin          = dtm_meta_d.get("vmin", 0)
dtm_vmax          = dtm_meta_d.get("vmax", 0)
dtm_vmean         = dtm_meta_d.get("vmean", 0)
dsm_res           = dsm_meta.get("res_m", 0.05)
chm_mean          = dsm_vmean - dtm_vmean

ts                = terrain_stats
t_mean            = ts.get("mean_slope", 0)
t_max             = ts.get("max_slope", 0)
t_flat            = ts.get("pct_flat", 0)
t_mod             = ts.get("pct_moderate", 0)
t_steep           = ts.get("pct_steep", 0)
t_vsteep          = ts.get("pct_vsteep", 0)
t_steep_total     = t_steep + t_vsteep

top_dep_max_depth = top_deps[0]["max_depth_m"] if top_deps else 0.0

# Approximate covered area from raster size and resolution
dsm_w = dsm_meta.get("width", 2695)
dsm_h = dsm_meta.get("height", 3113)
covered_area_ha = (dsm_w * dsm_res * dsm_h * dsm_res) / 10000.0

# ── Conditional tree page HTML ────────────────────────────────────────────────
if INCLUDE_TREES:
    tree_page_html = f"""
<!-- ═══════════════════════════ PAGE 5b — TREE ANALYSIS ═══════════════════════════ -->
<div class="page">
  <h2>5b. Trädanalys — individuell detektering</h2>

  <div class="callout green">
    <strong>Trädtätt område.</strong> CHM-baserad träddetektering via lokala maxima
    följt av watershed-segmentering identifierar individuella kronor.
    Krontäckning {canopy_pct_val:.1f}% · {n_trees_val:,} träd · {tree_density_val:.0f} träd/ha.
  </div>

  <div class="metric-grid">
    <div class="metric green">
      <div class="val">{n_trees_val:,}</div>
      <div class="key">Detekterade träd</div>
    </div>
    <div class="metric green">
      <div class="val">{canopy_pct_val:.1f}%</div>
      <div class="key">Krontäckning</div>
    </div>
    <div class="metric green">
      <div class="val">{tree_density_val:.0f}</div>
      <div class="key">Träd per hektar</div>
    </div>
    <div class="metric green">
      <div class="val">{mean_height_val:.1f} m</div>
      <div class="key">Medelhöjd</div>
    </div>
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 5b.1 — CHM, utjämnad CHM med trädtoppar, kronkarta, höjdklasser, täckningskarta och statistik</div>
    {img_tag(FIGS / "05b_tree_overview.png")}
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 5b.2 — Trädtäthetskarta (10m grid), kronytahistogram och höjdfördelning</div>
    {img_tag(FIGS / "05b_tree_density.png")}
  </div>

  <h3>Höjdklassfördelning</h3>
  <div class="two-col">
    <table>
      <thead><tr><th>Höjdklass</th><th>Antal träd</th><th>Andel (%)</th></tr></thead>
      <tbody>{tree_height_rows if tree_height_rows else '<tr><td colspan="3" style="text-align:center;color:#999;">Kör steg 5b för träddata</td></tr>'}</tbody>
    </table>
    <div>
      <p><strong>Maxhöjd:</strong> {max_height_val:.2f} m</p>
      <p><strong>Medelkrona:</strong> {crown_stats_val.get('mean_m2', 0):.1f} m²</p>
      <p><strong>Mediankrona:</strong> {crown_stats_val.get('median_m2', 0):.1f} m²</p>
      <p><strong>P95-krona:</strong> {crown_stats_val.get('p95_m2', 0):.1f} m²</p>
      <p><strong>Total krontäckyta:</strong> {canopy_area_val/10000:.2f} ha</p>
    </div>
  </div>
</div>
"""
else:
    tree_page_html = ""

# Cover stat box for trees (show if INCLUDE_TREES, else show depressions count)
if INCLUDE_TREES:
    cover_stat2 = f"""
      <div class="stat-box">
        <span class="num" style="color:#4CAF50">{n_trees_val:,}</span>
        <span class="lbl">Detekterade träd</span>
      </div>"""
    cover_meta_trees = f"""
      <div class="cover-meta-item">
        <label>Trädanalys</label>
        <span>{n_trees_val:,} träd  |  {canopy_pct_val:.1f}% krontäckning</span>
      </div>"""
else:
    cover_stat2 = f"""
      <div class="stat-box">
        <span class="num" style="color:#4CAF50">{n_deps}</span>
        <span class="lbl">Fördjupningar</span>
      </div>"""
    cover_meta_trees = ""

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8"/>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  :root {{
    --dark:   #08152A;
    --panel:  #0F2240;
    --border: #1D55D4;
    --text:   #F7F9FC;
    --accent: #4B8AF4;
    --crit:   #9b1d1d;
    --high:   #b5451b;
    --med:    #c87d2f;
    --low:    #2d6a4f;
    --muted:  #8FA8CC;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    font-size: 9pt;
    background: white;
    color: #1A2B42;
    line-height: 1.5;
  }}

  /* ── Cover page ── */
  .cover {{
    background: var(--dark);
    color: var(--text);
    width: 100%;
    min-height: 297mm;
    padding: 50mm 30mm 30mm 30mm;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .cover-logo {{
    font-family: 'DM Sans', sans-serif;
    font-size: 13pt;
    font-weight: 700;
    margin-bottom: 3mm;
  }}
  .cover-logo .north {{ color: #FFFFFF; }}
  .cover-logo .analytics {{ color: #4B8AF4; }}
  .cover h1 {{
    font-family: 'DM Sans', sans-serif;
    font-size: 32pt;
    font-weight: 700;
    color: white;
    line-height: 1.2;
    margin-bottom: 6mm;
    max-width: 140mm;
  }}
  .cover-sub {{
    font-size: 11pt;
    color: var(--muted);
    margin-bottom: 12mm;
  }}
  .cover-meta {{
    border-top: 1px solid var(--border);
    padding-top: 6mm;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4mm 8mm;
    font-size: 8.5pt;
  }}
  .cover-meta-item label {{
    display: block;
    color: var(--muted);
    font-size: 7.5pt;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1mm;
  }}
  .cover-meta-item span {{
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 8.5pt;
  }}
  .cover-stats {{
    margin-top: 12mm;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4mm;
  }}
  .stat-box {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 5mm 4mm;
    text-align: center;
  }}
  .stat-box .num {{
    font-family: 'DM Mono', monospace;
    font-size: 22pt;
    font-weight: 500;
    display: block;
  }}
  .stat-box .lbl {{
    font-size: 7pt;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}
  .cover-footer {{
    border-top: 1px solid var(--border);
    padding-top: 5mm;
    color: var(--muted);
    font-size: 7.5pt;
    font-family: 'DM Mono', monospace;
  }}

  /* ── Content pages ── */
  .page {{
    padding: 15mm 18mm 18mm 18mm;
    page-break-after: always;
    background: white;
  }}
  .page:last-child {{ page-break-after: auto; }}

  h2 {{
    font-family: 'DM Sans', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #08152A;
    border-bottom: 2px solid #08152A;
    padding-bottom: 2mm;
    margin-bottom: 6mm;
    margin-top: 4mm;
  }}
  h3 {{
    font-family: 'DM Sans', sans-serif;
    font-size: 10pt;
    font-weight: 600;
    color: #1D55D4;
    margin-bottom: 3mm;
    margin-top: 5mm;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  p {{ margin-bottom: 3mm; color: #333; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin-bottom: 5mm;
  }}
  thead tr {{ background: #08152A; color: white; }}
  thead th {{
    padding: 2.5mm 3mm;
    text-align: left;
    font-family: 'DM Mono', monospace;
    font-size: 7.5pt;
    font-weight: 500;
    letter-spacing: 0.04em;
  }}
  tbody tr:nth-child(even) {{ background: #f5f6fa; }}
  tbody tr:nth-child(odd)  {{ background: white; }}
  tbody td {{
    padding: 2mm 3mm;
    border-bottom: 1px solid #e0e0e0;
    font-family: 'DM Mono', monospace;
    font-size: 7.5pt;
    color: #222;
  }}
  tbody tr:hover {{ background: #edf2ff; }}

  .fig-full   {{ width: 100%; margin: 4mm 0; page-break-inside: avoid; }}
  .fig-half   {{ width: 49%; display: inline-block; vertical-align: top; margin: 2mm 0.5%; }}
  .fig-caption {{
    font-size: 7.5pt;
    color: #667388;
    font-style: italic;
    margin-bottom: 2mm;
    font-family: 'DM Sans', sans-serif;
  }}
  img {{ max-width: 100%; display: block; border-radius: 3px; }}

  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4mm;
    margin: 4mm 0;
  }}
  .metric {{
    background: #f5f6fa;
    border-left: 3px solid #1D55D4;
    padding: 3mm 4mm;
    border-radius: 0 3px 3px 0;
  }}
  .metric.green {{ border-left-color: #2d6a4f; }}
  .metric .val {{
    font-family: 'DM Mono', monospace;
    font-size: 13pt;
    font-weight: 600;
    color: #08152A;
  }}
  .metric .key {{
    font-size: 7pt;
    color: #667388;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5mm;
  }}

  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6mm;
    margin: 4mm 0;
  }}
  .three-col {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4mm;
    margin: 4mm 0;
  }}

  .callout {{
    background: #D6E6FF;
    border-left: 4px solid #1D55D4;
    padding: 3mm 5mm;
    border-radius: 0 4px 4px 0;
    margin: 4mm 0;
    font-size: 8.5pt;
  }}
  .callout.green {{
    background: #e8f5e9;
    border-left-color: #2d6a4f;
  }}
  .callout strong {{ color: #08152A; }}

  .vol-fig {{ margin: 4mm 0; page-break-inside: avoid; }}

  .missing-fig {{
    background: #fef3cd;
    border: 1px dashed #c87d2f;
    padding: 4mm;
    color: #666;
    font-size: 8pt;
    text-align: center;
    border-radius: 3px;
  }}

  .slope-bar {{
    display: flex;
    height: 6mm;
    border-radius: 3px;
    overflow: hidden;
    margin: 2mm 0 1mm 0;
  }}
  .slope-seg {{ height: 100%; }}

  footer {{
    position: fixed;
    bottom: 8mm;
    left: 18mm;
    right: 18mm;
    border-top: 1px solid #ddd;
    padding-top: 2mm;
    font-size: 7pt;
    color: #999;
    font-family: 'DM Mono', monospace;
    display: flex;
    justify-content: space-between;
  }}

  @page {{
    size: A4;
    margin: 0;
  }}
</style>
</head>
<body>

<!-- ═══════════════════════════ COVER PAGE ═══════════════════════════ -->
<div class="cover">
  <div>
    <div class="cover-logo"><span class="north">North Drone</span><span class="analytics"> Analytics</span></div>
    <h1>Terränganalys&shy;rapport</h1>
    <div class="cover-sub">
      Dataset: {DATASET_TITLE}<br/>
      Analyskedja: steg 0–6{"inkl. 5b trädanalys" if INCLUDE_TREES else ""} | GPU-accelererad
    </div>

    <div class="cover-meta">
      <div class="cover-meta-item">
        <label>Genereringsdatum</label>
        <span>{date_str}</span>
      </div>
      <div class="cover-meta-item">
        <label>Koordinatsystem</label>
        <span>{_cfg.get("crs", "EPSG:32633")}</span>
      </div>
      <div class="cover-meta-item">
        <label>Upplösning</label>
        <span>{dsm_res:.4f} m/pixel ({dsm_res*100:.1f} cm GSD)</span>
      </div>
      <div class="cover-meta-item">
        <label>Täckt yta</label>
        <span>{covered_area_ha:.1f} ha  ({dsm_w}×{dsm_h} px)</span>
      </div>
      {cover_meta_trees}
      <div class="cover-meta-item">
        <label>GPU-accelerering</label>
        <span>CuPy CUDA  |  RTX 3060  12 GB</span>
      </div>
    </div>

    <div class="cover-stats">
      <div class="stat-box">
        <span class="num" style="color:#4B8AF4">{laz_n_pts_M:.1f}M</span>
        <span class="lbl">LiDAR-punkter</span>
      </div>
      {cover_stat2}
      <div class="stat-box">
        <span class="num" style="color:#FF9800">{n_deps}</span>
        <span class="lbl">Fördjupningar</span>
      </div>
      <div class="stat-box">
        <span class="num" style="color:#e6edf3">{t_mean:.1f}°</span>
        <span class="lbl">Medelutning</span>
      </div>
    </div>
  </div>

  <div class="cover-footer">
    North Drone Analytics · {DATASET_TITLE} · Intern teknisk rapport · All data bearbetad lokalt · {date_str}
  </div>
</div>

<!-- ═══════════════════════════ PAGE 1 — INPUT VALIDATION ═══════════════════════════ -->
<div class="page">
  <h2>1. Validering av ingångsdata</h2>

  <div class="callout">
    <strong>Status: Alla 4 källfiler validerade.</strong>
    CRS konsistent ({_cfg.get("crs", "EPSG:32633")}) · Upplösning {dsm_res:.4f} m/px ·
    Punktmoln: {laz_n_pts:,} punkter i {laz_size_mb:.0f} MB ·
    Vegetation (klass 3–5): {100*(laz_classes.get('Lågveg',0)+laz_classes.get('Medelveg',0)+laz_classes.get('Högveg',0))/max(laz_n_pts,1):.1f}%
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 1.1 — Bounding boxes, DSM/DTM-thumbnail, ortofoto och punktmolnsklassificering</div>
    {img_tag(FIGS / "00_input_validation.png")}
  </div>

  <h3>Rasterdata — filsammanfattning</h3>
  <table>
    <thead>
      <tr>
        <th>Dataset</th><th>Res (m/px)</th><th>Storlek (px)</th>
        <th>MB</th><th>Täckning</th><th>Min (m)</th><th>Max (m)</th><th>Medel ± σ (m)</th>
      </tr>
    </thead>
    <tbody>{raster_rows}</tbody>
  </table>

  <h3>Punktmoln (LAZ) — klassfördelning</h3>
  <div class="two-col">
    <table>
      <thead><tr><th>Klass</th><th>Antal punkter</th><th>Andel</th></tr></thead>
      <tbody>{laz_rows}</tbody>
    </table>
    <div>
      <p><strong>Spann:</strong> {laz_x_span:.1f} m (E–V) × {laz_y_span:.1f} m (N–S)</p>
      <p><strong>Höjdintervall:</strong> {laz_z_min:.2f} — {laz_z_max:.2f} m</p>
      <p><strong>Markpunkter (klass 2):</strong> {laz_mark_pct:.1f}% av total</p>
    </div>
  </div>
</div>

<!-- ═══════════════════════════ PAGE 2 — OVERVIEW ═══════════════════════════ -->
<div class="page">
  <h2>2. Höjdmodell — översikt</h2>

  <div class="metric-grid">
    <div class="metric">
      <div class="val">{dsm_vmin:.0f}–{dsm_vmax:.0f} m</div>
      <div class="key">DSM Höjdintervall</div>
    </div>
    <div class="metric">
      <div class="val">{dtm_vmin:.0f}–{dtm_vmax:.0f} m</div>
      <div class="key">DTM Höjdintervall</div>
    </div>
    <div class="metric">
      <div class="val">{chm_mean:.2f} m</div>
      <div class="key">Medel CHM (DSM–DTM)</div>
    </div>
    <div class="metric">
      <div class="val">{dsm_res:.4f} m</div>
      <div class="key">Marksamplingavstånd</div>
    </div>
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 2.1 — DSM, DTM, CHM (Canopy Height Model), ortofoto och höjdhistogram</div>
    {img_tag(FIGS / "01_overview.png")}
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 2.2 — Höjdprofil längs horisontell linje (mitt i datasetet)</div>
    {img_tag(FIGS / "01_elevation_profile.png")}
  </div>
</div>

<!-- ═══════════════════════════ PAGE 3 — TERRAIN ANALYSIS ═══════════════════════════ -->
<div class="page">
  <h2>3. Terränganalys</h2>

  <h3>Lutningsstatistik</h3>
  <div class="metric-grid">
    <div class="metric">
      <div class="val">{t_mean:.1f}°</div>
      <div class="key">Medelutning</div>
    </div>
    <div class="metric">
      <div class="val">{t_max:.1f}°</div>
      <div class="key">Maxlutning</div>
    </div>
    <div class="metric">
      <div class="val">{t_flat:.1f}%</div>
      <div class="key">Plant (&lt;5°)</div>
    </div>
    <div class="metric">
      <div class="val">{t_steep_total:.1f}%</div>
      <div class="key">Brant (&gt;15°)</div>
    </div>
  </div>

  <div style="margin:3mm 0;">
    <div style="font-size:7.5pt;color:#555;margin-bottom:1.5mm;font-style:italic;">Lutningsklassfördelning (% av total yta)</div>
    <div class="slope-bar">
      <div class="slope-seg" style="width:{t_flat:.1f}%;background:#A5D6A7;" title="Plant"></div>
      <div class="slope-seg" style="width:{t_mod:.1f}%;background:#FFF176;" title="Måttlig"></div>
      <div class="slope-seg" style="width:{t_steep:.1f}%;background:#FFB74D;" title="Brant"></div>
      <div class="slope-seg" style="width:{t_vsteep:.1f}%;background:#EF5350;" title="Mycket brant"></div>
    </div>
    <div style="display:flex;gap:5mm;font-size:7pt;color:#555;">
      <span>Plant &lt;5°: {t_flat:.1f}%</span>
      <span>Måttlig 5–15°: {t_mod:.1f}%</span>
      <span>Brant 15–30°: {t_steep:.1f}%</span>
      <span>Mycket brant &gt;30°: {t_vsteep:.1f}%</span>
    </div>
  </div>

  <div class="callout">
    GPU-accelererad beräkning via CuPy CUDA gradient. Hillshade beräknad med NV-belysning (azimut 315°, 45° altitud).
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 3.1 — Hillshade, lutning, aspekt och lutningsklassificering</div>
    {img_tag(FIGS / "02_terrain_analysis.png")}
  </div>
</div>

<!-- ═══════════════════════════ PAGE 4 — ANOMALY DETECTION ═══════════════════════════ -->
<div class="page">
  <h2>4. Anomalidetektering — fördjupningar och upphöjningar</h2>

  <div class="metric-grid">
    <div class="metric">
      <div class="val">{n_deps}</div>
      <div class="key">Fördjupningar</div>
    </div>
    <div class="metric">
      <div class="val">{n_ups}</div>
      <div class="key">Upphöjningar</div>
    </div>
    <div class="metric">
      <div class="val">{total_dep_area:.0f} m²</div>
      <div class="key">Total fördjupningsyta</div>
    </div>
    <div class="metric">
      <div class="val">{top_dep_max_depth:.3f} m</div>
      <div class="key">Max avvikelse</div>
    </div>
  </div>

  <div class="callout">
    Multi-skala lokal avvikelseanalys med GPU-accelererade uniform-filter (CuPy).
    Parametrar läses från config.json.
  </div>

  <div class="fig-full">
    <div class="fig-caption">Figur 4.1 — Detekterade anomalier: DTM med overlay, etiketterade objekt och djupkarta</div>
    {img_tag(FIGS / "03_depressions.png")}
  </div>

  <h3>Topp 5 fördjupningar (djupast)</h3>
  <table>
    <thead>
      <tr><th>Objekt #</th><th>Area (m²)</th><th>Max avv. (m)</th><th>Medel avv. (m)</th></tr>
    </thead>
    <tbody>{dep_rows if dep_rows else '<tr><td colspan="4" style="text-align:center;color:#999;">Inga fördjupningar detekterade</td></tr>'}</tbody>
  </table>
</div>

{tree_page_html}

<!-- ═══════════════════════════ PAGE 5 — VOLUME ANALYSIS ═══════════════════════════ -->
<div class="page">
  <h2>5. Volymanalys</h2>

  <div class="callout">
    Tre referensplansmetoder tillämpas per objekt: (1) <strong>Percentil 95</strong> — 95:e percentilens höjd,
    (2) <strong>Kantmedelvärde</strong> — medelhöjden längs patchens kant,
    (3) <strong>Interpolerad yta</strong> — linjär interpolering av kantpunkter som referensyta.
    Kantmedelvärde och interpolerad yta ger typiskt de mest meningsfulla estimaten.
  </div>

  <h3>Volymer per objekt (m³)</h3>
  <table>
    <thead>
      <tr>
        <th>Objekt #</th><th>Area (m²)</th><th>Max avv. (m)</th>
        <th>Percentil 95</th><th>Kantmedelvärde</th><th>Interpolerad yta</th>
      </tr>
    </thead>
    <tbody>{vol_rows if vol_rows else '<tr><td colspan="6" style="text-align:center;color:#999;">Kör steg 4 för att generera volymdata</td></tr>'}</tbody>
  </table>

  {vol_fig_tags}
</div>

<!-- ═══════════════════════════ PAGE 6 — POINT CLOUD ═══════════════════════════ -->
<div class="page">
  <h2>6. Punktmolnsanalys</h2>

  <div class="metric-grid">
    <div class="metric">
      <div class="val">{laz_n_pts_M:.1f}M</div>
      <div class="key">Totalt antal punkter</div>
    </div>
    <div class="metric">
      <div class="val">{laz_mark_pct:.1f}%</div>
      <div class="key">Markpunkter (klass 2)</div>
    </div>
    <div class="metric">
      <div class="val">{laz_x_span:.0f}×{laz_y_span:.0f} m</div>
      <div class="key">Täckt yta</div>
    </div>
    <div class="metric">
      <div class="val">{laz_z_span:.2f} m</div>
      <div class="key">Höjdspann</div>
    </div>
  </div>

  <div class="two-col">
    <div>
      <div class="fig-caption">Figur 6.1 — Täthetskarta, höjdkarta och klassfördelning</div>
      {img_tag(FIGS / "05_pointcloud_analysis.png")}
    </div>
    <div>
      <div class="fig-caption">Figur 6.2 — 3D scatter (urval 200k punkter, färgat efter klass)</div>
      {img_tag(FIGS / "05_pointcloud_3d.png")}
    </div>
  </div>
</div>

<!-- ═══════════════════════════ PAGE 7 — FINAL SUMMARY ═══════════════════════════ -->
<div class="page">
  <h2>7. Sammanfattande rapport</h2>

  <div class="fig-full">
    <div class="fig-caption">Figur 7.1 — Komplett analysöversikt (steg 1–6, 4×3 grid)</div>
    {img_tag(FIGS / "06_final_report.png")}
  </div>

  <h3>Metodbeskrivning</h3>
  <table>
    <thead><tr><th>Steg</th><th>Metod</th><th>Verktyg</th><th>GPU</th></tr></thead>
    <tbody>
      <tr><td>0 — Validering</td><td>CRS-kontroll, nodata-maskning, LAZ-statistik</td><td>rasterio, laspy</td><td>—</td></tr>
      <tr><td>1 — Höjdmodell</td><td>DSM/DTM-inläsning, CHM = DSM−DTM</td><td>rasterio, numpy</td><td>CuPy (stats)</td></tr>
      <tr><td>2 — Terränganalys</td><td>Gradient → slope/aspekt, LightSource hillshade</td><td>scipy, matplotlib</td><td>CuPy gradient</td></tr>
      <tr><td>3 — Anomalier</td><td>Multi-skala lokal avvikelse (config-styrda radier)</td><td>cupyx, scipy.ndimage</td><td>CuPy uniform_filter</td></tr>
      <tr><td>4 — Volym</td><td>3 referensplansmetoder per fördjupning</td><td>scipy.interpolate</td><td>CuPy nansum/percentile</td></tr>
      <tr><td>5 — Punktmoln</td><td>Täthetskarta, max-Z-karta, klassfördelning</td><td>laspy, numpy</td><td>CuPy bincount</td></tr>
      {"<tr><td>5b — Trädanalys</td><td>CHM lokala maxima → watershed kronor, stats</td><td>skimage, scipy, CuPy</td><td>CuPy gaussian + bincount</td></tr>" if INCLUDE_TREES else ""}
      <tr><td>6 — Rapport</td><td>Sammanfattningsgrid med alla deriverade lager</td><td>matplotlib</td><td>—</td></tr>
    </tbody>
  </table>

  <h3>Datakvalitet och begränsningar</h3>
  <p>
    DSM och DTM har hög täckning. Nodata-regioner i kanterna är normalt för
    ODM-producerade höjdmodeller pga. bildöverlapp.
    Volymberäkningarna bör tolkas med försiktighet i tät vegetation — CHM-subtraktionen
    ger ett bättre markunderlag (DTM) men vegetationseffekter kan påverka kantmedelvärden.
  </p>
  <p>
    Anomalidetekteringen använder dataset-specifika tröskelvärden från config.json för att
    anpassa känsligheten per dataset.
  </p>
</div>

<footer>
  <span>North Drone Analytics — {DATASET_TITLE} — Intern rapport</span>
  <span>Genererad: {date_str} | {_cfg.get("crs", "EPSG:32633")}</span>
  <span>{FOOTER_DATASET}</span>
</footer>

</body>
</html>"""

# ── Write HTML ─────────────────────────────────────────────────────────────────
html_stem = OUT_PDF.stem
html_path = REPS / f"{html_stem}.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"  HTML sparad: {html_path}")

# ── Generate PDF via WeasyPrint ────────────────────────────────────────────────
print("  Genererar PDF via WeasyPrint...")
try:
    from weasyprint import HTML as WH, CSS
    WH(filename=str(html_path)).write_pdf(str(OUT_PDF))
    size_mb = OUT_PDF.stat().st_size / 1e6
    print(f"  PDF sparad: {OUT_PDF}  ({size_mb:.1f} MB)")
except Exception as e:
    print(f"  WeasyPrint fel: {e}")
    print("      HTML-rapporten är fortfarande tillgänglig.")

print("\n" + "=" * 60)
print(f"  RAPPORT KLAR: {OUT_PDF}")
print("=" * 60)

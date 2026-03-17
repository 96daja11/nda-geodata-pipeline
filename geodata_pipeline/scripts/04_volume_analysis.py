#!/usr/bin/env python3
"""
Steg 4 — Volymanalys för identifierade fördjupningar
Läser fördjupningar från steg 3, beräknar volym med 3 metoder.
GPU-accelererad via CuPy. Exporterar resultat till JSON.
"""
import matplotlib
matplotlib.use("Agg")
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path

try:
    import cupy as cp
    cp.array([1])
    GPU_AVAILABLE = True
    _dev = cp.cuda.Device(0)
    GPU_NAME = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU: {GPU_NAME} | {_dev.mem_info[1]//1024**2} MB VRAM")
except Exception as _e:
    import numpy as cp
    GPU_AVAILABLE = False; GPU_NAME = "CPU fallback"
    print(f"CPU mode ({_e})")

import numpy as np, rasterio, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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

# BASE is set above via --project-dir
RAW = BASE/"data"/"raw"; OUT_FIGS = BASE/"outputs"/"figures"; OUT_REPS = BASE/"outputs"/"reports"
OUT_FIGS.mkdir(parents=True, exist_ok=True); OUT_REPS.mkdir(parents=True, exist_ok=True)
DARK="#0d1117"; PANEL="#161b22"; BORDER="#30363d"; TEXT="#e6edf3"
TOP_N = 5

print("\n" + "="*60 + "\nSTEG 4 — VOLYMANALYS\n" + "="*60)

with rasterio.open(RAW/"dtm.tif") as src:
    dtm = src.read(1).astype(np.float64); nd = src.nodata; pixel_m = src.res[0]
    if nd is not None: dtm[dtm == nd] = np.nan
print(f"DTM: {dtm.shape[1]}×{dtm.shape[0]} px")

dep_json = OUT_REPS/"03_depressions.json"
if not dep_json.exists():
    print("Kör steg 3 först"); exit(1)
with open(dep_json) as f: dep_data = json.load(f)

depressions_raw = dep_data.get("depressions", [])
print(f"Laddade {len(depressions_raw)} fördjupningar från steg 3")

valid_deps = [{"label":d["id"],"area_m2":d["area_m2"],"max_depth_m":d["max_depth_m"],
               "centroid":(d["centroid_row"],d["centroid_col"]),"bbox":d["bbox"]}
              for d in depressions_raw]
top5 = sorted([d for d in valid_deps if d["area_m2"]>=1.0], key=lambda x:-x["max_depth_m"])[:TOP_N]
print(f"Analyserar topp {len(top5)} fördjupningar")

def compute_volume_methods(patch, px_size):
    results = {}; px_area = px_size**2
    if GPU_AVAILABLE:
        pg = cp.array(patch)
        bv = cp.concatenate([pg[0,:],pg[-1,:],pg[:,0],pg[:,-1]])
        bm = float(cp.nanmean(bv)); rp = float(cp.percentile(pg.ravel(), 95))
        results["Percentil 95"] = float(cp.nansum(cp.maximum(rp-pg,0))*px_area)
        results["Kantmedelvärde"] = float(cp.nansum(cp.maximum(bm-pg,0))*px_area)
    else:
        bv = np.concatenate([patch[0,:],patch[-1,:],patch[:,0],patch[:,-1]])
        bm = float(np.nanmean(bv)); rp = float(np.nanpercentile(patch,95))
        results["Percentil 95"] = float(np.nansum(np.maximum(rp-patch,0))*px_area)
        results["Kantmedelvärde"] = float(np.nansum(np.maximum(bm-patch,0))*px_area)
    rows,cols = patch.shape; y,x = np.mgrid[0:rows,0:cols]
    em = np.zeros_like(patch,dtype=bool); em[0,:]=em[-1,:]=em[:,0]=em[:,-1]=True
    ep = np.column_stack([y[em],x[em]]); ev = patch[em]; vi = ~np.isnan(ev)
    if vi.sum()>=4:
        rs = griddata(ep[vi],ev[vi],(y,x),method="linear")
        d3 = np.where(np.isnan(rs),0,np.maximum(rs-patch,0))
        results["Interpolerad yta"] = float(np.nansum(d3)*px_area)
    else:
        results["Interpolerad yta"] = results["Kantmedelvärde"]
    return results

all_results = []
print()
for dep in tqdm(top5, desc="Volymanalys"):
    r0,c0,r1,c1 = dep["bbox"]; pad=15
    r0p=max(0,r0-pad); r1p=min(dtm.shape[0],r1+pad)
    c0p=max(0,c0-pad); c1p=min(dtm.shape[1],c1+pad)
    dtm_p = dtm[r0p:r1p,c0p:c1p]
    dtm_nn = np.where(np.isnan(dtm_p),float(np.nanmean(dtm_p)),dtm_p)
    if dtm_p.size<25: continue
    volumes = compute_volume_methods(dtm_nn, pixel_m)
    lbl=dep["label"]; area=dep["area_m2"]; mdep=dep["max_depth_m"]
    print(f"\n  Objekt #{lbl}  area={area:.1f}m²  max_avv={mdep:.3f}m")
    for k,v in volumes.items(): print(f"    {k:<25}: {v:.3f} m³")
    all_results.append({"object_id":int(lbl),"area_m2":round(area,2),"max_depth_m":round(mdep,3),
                         "volumes_m3":{k:round(v,3) for k,v in volumes.items()}})

    bv_arr = np.concatenate([dtm_nn[0,:],dtm_nn[-1,:],dtm_nn[:,0],dtm_nn[:,-1]])
    ref_val = float(np.nanmean(bv_arr)); depth_map = np.maximum(ref_val-dtm_nn,0)

    fig = plt.figure(figsize=(18,8),facecolor=DARK)
    ax3d = fig.add_subplot(131,projection="3d")
    rp_,cp_ = dtm_nn.shape; s3=max(1,max(rp_,cp_)//80)
    Xg,Yg = np.meshgrid(np.arange(0,cp_,s3)*pixel_m, np.arange(0,rp_,s3)*pixel_m)
    ax3d.plot_surface(Xg,Yg,dtm_nn[::s3,::s3],cmap="terrain",linewidth=0,antialiased=True,alpha=0.9)
    ax3d.plot_surface(Xg,Yg,np.full_like(dtm_nn[::s3,::s3],ref_val),color="cyan",alpha=0.2)
    ax3d.set_title(f"3D-vy #{lbl}",color=TEXT,fontsize=10)
    ax3d.tick_params(colors=TEXT,labelsize=6)
    for pane in [ax3d.xaxis.pane,ax3d.yaxis.pane,ax3d.zaxis.pane]: pane.fill=False

    ax2 = fig.add_subplot(132); ax2.set_facecolor(PANEL); ax2.axis("off")
    im = ax2.imshow(np.ma.masked_where(depth_map<0.02,depth_map),cmap="Blues_r",interpolation="bilinear")
    cb=plt.colorbar(im,ax=ax2,fraction=0.046,pad=0.02); cb.set_label("Djup (m)",color=TEXT,fontsize=7)
    cb.ax.yaxis.set_tick_params(color=TEXT,labelcolor=TEXT,labelsize=6); cb.outline.set_edgecolor(BORDER)
    ax2.set_title(f"Djupkarta #{lbl}  ({area:.0f}m², avv {mdep:.2f}m)",color=TEXT,fontsize=9,pad=5)

    ax3 = fig.add_subplot(133); ax3.set_facecolor(PANEL)
    methods=list(volumes.keys()); vals=list(volumes.values())
    bars=ax3.bar(methods,vals,color=["#1565C0","#2E7D32","#E65100"][:len(methods)],edgecolor=DARK,width=0.55)
    for bar,vol in zip(bars,vals):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*0.02,
                 f"{vol:.2f} m³",ha="center",fontsize=10,fontweight="bold",color=TEXT)
    ax3.set_ylabel("Volym (m³)",color=TEXT,fontsize=9); ax3.tick_params(colors=TEXT,labelsize=8)
    ax3.set_ylim(0,max(vals)*1.3 if vals else 1); ax3.set_title("Volymer",color=TEXT,fontsize=9,pad=5)
    for sp in ["top","right"]: ax3.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax3.spines[sp].set_color(BORDER)

    fig.patch.set_facecolor(DARK)
    fig.text(0.99,0.01,f"GPU: {GPU_NAME}",ha="right",va="bottom",fontsize=7,color="#586069",style="italic")
    fig.suptitle(f"Steg 4 — Volymanalys: Objekt #{lbl}",fontsize=14,fontweight="bold",color=TEXT,y=0.98)
    plt.tight_layout()
    out_path = OUT_FIGS/f"04_volume_obj{lbl:03d}.png"
    plt.savefig(out_path,dpi=200,bbox_inches="tight",facecolor=DARK); plt.close()
    print(f"  Figur: {out_path}")

json_path = OUT_REPS/"04_volumes.json"
with open(json_path,"w",encoding="utf-8") as f: json.dump(all_results,f,indent=2,ensure_ascii=False)
print(f"\nJSON: {json_path}")
print("Steg 4 klar.\n" + "="*60)

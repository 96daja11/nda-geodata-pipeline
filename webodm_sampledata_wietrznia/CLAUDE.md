# CLAUDE.md — Geodata Analysis Pipeline (WebODM → Python)

## Projektöversikt

Detta projekt analyserar dröndata exporterad från WebODM. Målet är att bygga en fullständig,
visualiseringsdriven analyskedja för höjdmodeller, punktmoln och volymberäkningar — med fokus
på att varje analyssteg ska vara begripligt och visuellt kommunicerat.

Primärt testdataset: **Wietrznia** (grönområde med bergssluttningar och schaktliknande fördjupningar)

Filosofi: Äg hela analysstacken. Inga kommersiella plattformar. Alla beräkningar görs i Python
med öppen källkod. Resultaten ska kunna ingå i kundrapporter.

---

## Datastruktur

```
project/
├── CLAUDE.md               ← denna fil
├── data/
│   ├── raw/
│   │   ├── dsm.tif             # Digital Surface Model från WebODM
│   │   ├── dtm.tif             # Digital Terrain Model (markmodell)
│   │   ├── odm_orthophoto.tif  # Ortofoto (RGB)
│   │   └── georeferenced_model.laz  # Punktmoln (komprimerat LAS)
│   └── derived/
│       ├── chm.tif             # Canopy Height Model (DSM - DTM)
│       ├── slope.tif           # Lutningskarta
│       ├── hillshade.tif       # Skuggningsmodell
│       └── roi/                # Exporterade intresseområden
├── scripts/
│   ├── 00_verify_inputs.py     # Steg 0: Validera och inspektera ingångsdata
│   ├── 01_load_and_inspect.py  # Steg 1: Ladda och visualisera grunddata
│   ├── 02_terrain_analysis.py  # Steg 2: Terränganalys (lutning, aspekt, hillshade)
│   ├── 03_anomaly_detection.py # Steg 3: Detektera fördjupningar och anomalier
│   ├── 04_volume_analysis.py   # Steg 4: Volymberäkning för utvalda objekt
│   ├── 05_pointcloud_analysis.py # Steg 5: Punktmolnsanalys med open3d/laspy
│   └── 06_report_figures.py    # Steg 6: Generera rapportklara figurer
├── outputs/
│   ├── figures/                # Alla figurer (PNG, 300 dpi)
│   └── reports/                # CSV/JSON med beräkningsresultat
└── requirements.txt
```

---

## Krav och installation

```bash
pip install rasterio numpy scipy matplotlib laspy open3d pyproj shapely \
            geopandas richdem scikit-image pandas tqdm seaborn
```

**requirements.txt:**
```
rasterio>=1.3
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
laspy>=2.5
open3d>=0.17
pyproj>=3.5
shapely>=2.0
geopandas>=0.13
richdem>=2.3
scikit-image>=0.21
pandas>=2.0
tqdm>=4.65
seaborn>=0.12
```

---

## Steg 0 — Validera ingångsdata

**Script:** `scripts/00_verify_inputs.py`  
**Syfte:** Kontrollera att alla filer finns, är georefererade, har samma CRS och rimliga värden.  
**Visualisering:** Textutskrift med sammanfattning + enkel figur med filernas bounding boxes.

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FILES = {
    "DSM":        "data/raw/dsm.tif",
    "DTM":        "data/raw/dtm.tif",
    "Orthophoto": "data/raw/odm_orthophoto.tif",
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#2196F3", "#4CAF50", "#FF9800"]

for (name, path), color in zip(FILES.items(), colors):
    p = Path(path)
    if not p.exists():
        print(f"⚠️  SAKNAS: {path}")
        continue

    with rasterio.open(path) as src:
        b = src.bounds
        crs = src.crs
        res = src.res
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"  CRS:        {crs}")
        print(f"  Upplösning: {res[0]:.3f} m/pixel")
        print(f"  Storlek:    {src.width} x {src.height} px")
        print(f"  Bounds:     {b}")

        data = src.read(1, masked=True)
        print(f"  Höjd min:   {data.min():.2f} m")
        print(f"  Höjd max:   {data.max():.2f} m")
        print(f"  Höjd medel: {data.mean():.2f} m")

        rect = mpatches.FancyArrowPatch(
            (b.left, b.bottom), (b.right, b.top),
            label=name, color=color, alpha=0.4
        )
        ax.add_patch(mpatches.Rectangle(
            (b.left, b.bottom),
            b.right - b.left,
            b.top - b.bottom,
            linewidth=2, edgecolor=color,
            facecolor=color, alpha=0.2,
            label=name
        ))

ax.autoscale()
ax.set_aspect("equal")
ax.legend()
ax.set_title("Bounding boxes för ingångsdata", fontsize=14)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.tight_layout()
plt.savefig("outputs/figures/00_input_validation.png", dpi=150)
plt.show()
print("\n✅ Validering klar. Figur sparad.")
```

---

## Steg 1 — Ladda och visualisera grunddata

**Script:** `scripts/01_load_and_inspect.py`  
**Syfte:** Skapa en tydlig översikt av rådata — höjdvärden, ortofoto och höjdprofiler.  
**Visualisering:** 2×2 panel med DSM, DTM, ortofoto och höjdhistogram. Plus interaktiv profillinje.

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

def load_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, transform, crs

def plot_raster_panel(dsm, dtm, ortho_r, ortho_g, ortho_b):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # DSM
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dsm, cmap="terrain", interpolation="bilinear")
    plt.colorbar(im1, ax=ax1, label="Höjd (m)")
    ax1.set_title("DSM — Digital Surface Model\n(inkl. vegetation & strukturer)", fontsize=11)
    ax1.axis("off")

    # DTM
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(dtm, cmap="terrain", interpolation="bilinear")
    plt.colorbar(im2, ax=ax2, label="Höjd (m)")
    ax2.set_title("DTM — Digital Terrain Model\n(markmodell)", fontsize=11)
    ax2.axis("off")

    # CHM = DSM - DTM (vegetationshöjd)
    chm = dsm - dtm
    chm[chm < 0] = 0
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(chm, cmap="YlGn", interpolation="bilinear")
    plt.colorbar(im3, ax=ax3, label="Höjd (m)")
    ax3.set_title("CHM — Canopy Height Model\n(DSM minus DTM)", fontsize=11)
    ax3.axis("off")

    # Ortofoto
    ax4 = fig.add_subplot(gs[1, 0:2])
    rgb = np.stack([ortho_r, ortho_g, ortho_b], axis=-1)
    rgb = np.clip(rgb / rgb.max(), 0, 1)
    ax4.imshow(rgb)
    ax4.set_title("Ortofoto (RGB)", fontsize=11)
    ax4.axis("off")

    # Histogram
    ax5 = fig.add_subplot(gs[1, 2])
    valid_dsm = dsm[~np.isnan(dsm)].flatten()
    valid_dtm = dtm[~np.isnan(dtm)].flatten()
    ax5.hist(valid_dsm, bins=80, alpha=0.6, color="#2196F3", label="DSM")
    ax5.hist(valid_dtm, bins=80, alpha=0.6, color="#4CAF50", label="DTM")
    ax5.set_xlabel("Höjd (m)")
    ax5.set_ylabel("Antal pixlar")
    ax5.set_title("Höjdfördelning\nDSM vs DTM", fontsize=11)
    ax5.legend()

    plt.suptitle("Steg 1 — Översikt av ingångsdata", fontsize=15, fontweight="bold", y=1.01)
    plt.savefig("outputs/figures/01_overview.png", dpi=200, bbox_inches="tight")
    plt.show()

# Höjdprofil längs en linje
def plot_elevation_profile(dsm, transform, row_frac=0.5):
    """Rita höjdprofil längs en horisontell linje mitt i rasterdata."""
    row = int(dsm.shape[0] * row_frac)
    profile = dsm[row, :]
    pixel_width = transform.a  # meter per pixel
    x_m = np.arange(len(profile)) * pixel_width

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1, 2]})

    # Karta med profillinje
    axes[0].imshow(dsm, cmap="terrain")
    axes[0].axhline(row, color="red", linewidth=2, linestyle="--", label=f"Profillinje (rad {row})")
    axes[0].set_title("Profillinjens position i DSM", fontsize=11)
    axes[0].legend(loc="upper right")
    axes[0].axis("off")

    # Profil
    axes[1].fill_between(x_m, profile.min() - 1, profile, alpha=0.4, color="#2196F3")
    axes[1].plot(x_m, profile, color="#1565C0", linewidth=1.5)
    axes[1].set_xlabel("Avstånd längs profil (m)")
    axes[1].set_ylabel("Höjd (m)")
    axes[1].set_title("Höjdprofil", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/figures/01_elevation_profile.png", dpi=200)
    plt.show()
```

---

## Steg 2 — Terränganalys

**Script:** `scripts/02_terrain_analysis.py`  
**Syfte:** Beräkna lutning (slope), aspekt (exposition) och hillshade. Identifiera branta kanter.  
**Verktyg:** `richdem` för hydrologiskt korrekta beräkningar, `scipy` för gradient.  
**Visualisering:** 4-panel med hillshade, slope, aspekt och kombinerad overlay.

```python
import rasterio
import numpy as np
import richdem as rd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

def compute_terrain_attributes(dtm_array, pixel_size):
    """Beräkna slope och aspekt med richdem."""
    rda = rd.rdarray(dtm_array, no_data=np.nan)
    slope_rad = rd.TerrainAttribute(rda, attrib="slope_riserun")
    slope_deg = np.degrees(np.arctan(slope_rad))
    aspect = rd.TerrainAttribute(rda, attrib="aspect")
    return np.array(slope_deg), np.array(aspect)

def compute_hillshade(dtm, pixel_size, azimuth=315, altitude=45):
    """Beräkna hillshade med matplotlib LightSource."""
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    hillshade = ls.hillshade(dtm, vert_exag=2, dx=pixel_size, dy=pixel_size)
    return hillshade

def plot_terrain_analysis(dtm, hillshade, slope, aspect, pixel_size):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Hillshade med DTM overlay
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", interpolation="bilinear")
    im = ax.imshow(dtm, cmap="terrain", alpha=0.5, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Höjd (m)", fraction=0.046)
    ax.set_title("Hillshade + DTM\n(belysning från NV, 45°)", fontsize=11)
    ax.axis("off")

    # Slope
    ax = axes[0, 1]
    im = ax.imshow(slope, cmap="RdYlGn_r", vmin=0, vmax=45, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Lutning (°)", fraction=0.046)
    ax.set_title("Lutning (Slope)\nRött = brant, Grönt = plant", fontsize=11)
    ax.axis("off")

    # Aspekt (kompassriktning för sluttningen)
    ax = axes[1, 0]
    cmap_aspect = plt.cm.hsv
    im = ax.imshow(aspect, cmap=cmap_aspect, vmin=0, vmax=360, interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, label="Aspekt (°)", fraction=0.046)
    cbar.set_ticks([0, 90, 180, 270, 360])
    cbar.set_ticklabels(["N", "Ö", "S", "V", "N"])
    ax.set_title("Aspekt (Exposition)\nSluttningsriktning", fontsize=11)
    ax.axis("off")

    # Lutningsklassificering
    ax = axes[1, 1]
    slope_classes = np.zeros_like(slope)
    slope_classes[slope < 5]  = 1   # plant
    slope_classes[(slope >= 5) & (slope < 15)]  = 2   # måttlig
    slope_classes[(slope >= 15) & (slope < 30)] = 3   # brant
    slope_classes[slope >= 30] = 4   # mycket brant

    cmap_class = mcolors.ListedColormap(["#A5D6A7", "#FFF176", "#FFB74D", "#EF5350"])
    im = ax.imshow(slope_classes, cmap=cmap_class, vmin=0.5, vmax=4.5, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.set_ticks([1, 2, 3, 4])
    cbar.set_ticklabels(["Plant (<5°)", "Måttlig (5-15°)", "Brant (15-30°)", "Mycket brant (>30°)"])
    ax.set_title("Lutningsklassificering", fontsize=11)
    ax.axis("off")

    plt.suptitle("Steg 2 — Terränganalys", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/02_terrain_analysis.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Statistik
    print(f"\n📊 Terrängstatistik:")
    print(f"   Medelutning:      {np.nanmean(slope):.1f}°")
    print(f"   Maxlutning:       {np.nanmax(slope):.1f}°")
    print(f"   Andel plant (<5°): {100*(slope<5).sum()/slope.size:.1f}%")
    print(f"   Andel brant (>15°): {100*(slope>15).sum()/slope.size:.1f}%")
```

---

## Steg 3 — Detektera fördjupningar och anomalier

**Script:** `scripts/03_anomaly_detection.py`  
**Syfte:** Hitta automatiskt gropar, schakt, och avvikande låga/höga punkter mot omgivningen.  
**Verktyg:** `richdem` (Fill Depression / Depression-breaching), `scipy` (local minima), `scikit-image`.  
**Visualisering:** Detekterade fördjupningar med konturoverlays och mätta djup.

```python
import rasterio
import numpy as np
import richdem as rd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage import measure, morphology

def detect_depressions(dtm, pixel_size):
    """
    Identifiera hydrologiska fördjupningar via Fill-Depression metoden.
    Skillnaden (filled - original) = fördjupningarnas djup.
    """
    rda = rd.rdarray(dtm.copy(), no_data=np.nan)
    filled = rd.FillDepressions(rda, in_place=False)
    filled = np.array(filled)

    depression_depth = filled - dtm
    depression_depth[depression_depth < 0.05] = 0  # filtrera brus (<5 cm)

    return depression_depth, filled

def label_and_filter_depressions(depression_depth, min_area_m2, pixel_size):
    """Etikettera sammanhängande fördjupningar och filtrera efter storlek."""
    binary = depression_depth > 0
    labeled = measure.label(binary, connectivity=2)
    props = measure.regionprops(labeled, intensity_image=depression_depth)

    pixel_area = pixel_size ** 2
    valid = []
    for p in props:
        area_m2 = p.area * pixel_area
        if area_m2 >= min_area_m2:
            valid.append({
                "label": p.label,
                "area_m2": area_m2,
                "max_depth_m": p.max_intensity,
                "mean_depth_m": p.mean_intensity,
                "centroid": p.centroid,
                "bbox": p.bbox,
            })

    return labeled, valid

def plot_depressions(dtm, depression_depth, labeled, valid_depressions, pixel_size):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # DTM med fördjupningar overlay
    ax = axes[0]
    ax.imshow(dtm, cmap="terrain", interpolation="bilinear")
    masked_depth = np.ma.masked_where(depression_depth == 0, depression_depth)
    im = ax.imshow(masked_depth, cmap="Blues", alpha=0.8, interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Fördjupningsdjup (m)", fraction=0.046)
    ax.set_title("DTM med detekterade fördjupningar\n(blå = fördjupning)", fontsize=11)
    ax.axis("off")

    # Etiketterade fördjupningar
    ax = axes[1]
    ax.imshow(dtm, cmap="gray", alpha=0.5)
    cmap_labels = plt.cm.tab20
    for d in valid_depressions:
        mask = labeled == d["label"]
        ax.contourf(mask, levels=[0.5, 1.5], colors=[cmap_labels(d["label"] % 20)], alpha=0.6)
        cy, cx = d["centroid"]
        ax.annotate(
            f"#{d['label']}\n{d['max_depth_m']:.2f}m djup\n{d['area_m2']:.0f}m²",
            (cx, cy),
            fontsize=7,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
        )
    ax.set_title(f"Etiketterade fördjupningar\n({len(valid_depressions)} st hittade)", fontsize=11)
    ax.axis("off")

    # Djupkarta
    ax = axes[2]
    im = ax.imshow(depression_depth, cmap="Blues", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Djup (m)", fraction=0.046)
    contours = measure.find_contours(depression_depth, 0.1)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], "r-", linewidth=0.5, alpha=0.5)
    ax.set_title("Fördjupningsdjupskarta\n(röda konturer = kanter)", fontsize=11)
    ax.axis("off")

    plt.suptitle("Steg 3 — Detekterade fördjupningar och schakt", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/03_depressions.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Skriv ut tabell
    print(f"\n{'='*60}")
    print(f"{'#':<5} {'Area (m²)':<12} {'Max djup (m)':<14} {'Medeldjup (m)':<15}")
    print(f"{'-'*60}")
    for d in sorted(valid_depressions, key=lambda x: -x["max_depth_m"]):
        print(f"{d['label']:<5} {d['area_m2']:<12.1f} {d['max_depth_m']:<14.3f} {d['mean_depth_m']:<15.3f}")
```

---

## Steg 4 — Volymanalys

**Script:** `scripts/04_volume_analysis.py`  
**Syfte:** Beräkna exakt volym för enskilda fördjupningar eller manuellt avgränsade ROI:er.  
**Metod:** Tre referensplansmetoder — (1) percentilplan, (2) kantmedelvärde, (3) linjär interpolation av kanter.  
**Visualisering:** 3D-vy av fördjupning + jämförelseplot av metodernas resultat.

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.patches as mpatches

def compute_volume_methods(dtm_patch, pixel_size):
    """
    Beräkna volym med tre metoder för ett utklippt DTM-patch.
    Returnar dict med volymer i m³.
    """
    results = {}
    pixel_area = pixel_size ** 2

    # Metod 1: Referensplan = 95:e percentilen (approximerar kantmedelhöjd)
    ref_p95 = np.nanpercentile(dtm_patch, 95)
    diff1 = ref_p95 - dtm_patch
    diff1[diff1 < 0] = 0
    results["Percentil 95"] = np.nansum(diff1) * pixel_area

    # Metod 2: Medelvärde av yttersta randen
    border = np.concatenate([
        dtm_patch[0, :], dtm_patch[-1, :],
        dtm_patch[:, 0], dtm_patch[:, -1]
    ])
    ref_border = np.nanmean(border)
    diff2 = ref_border - dtm_patch
    diff2[diff2 < 0] = 0
    results["Kantmedelvärde"] = np.nansum(diff2) * pixel_area

    # Metod 3: Linjär interpolation av kant → referensyta
    rows, cols = dtm_patch.shape
    y, x = np.mgrid[0:rows, 0:cols]
    # Kantpunkter
    edge_mask = np.zeros_like(dtm_patch, dtype=bool)
    edge_mask[0, :] = edge_mask[-1, :] = True
    edge_mask[:, 0] = edge_mask[:, -1] = True
    edge_points = np.column_stack([y[edge_mask], x[edge_mask]])
    edge_values = dtm_patch[edge_mask]
    ref_surface = griddata(edge_points, edge_values, (y, x), method="linear")
    diff3 = ref_surface - dtm_patch
    diff3[diff3 < 0] = 0
    results["Interpolerad referensyta"] = float(np.nansum(diff3) * pixel_area)

    return results

def plot_volume_analysis(dtm_patch, pixel_size, label="Objekt"):
    results = compute_volume_methods(dtm_patch, pixel_size)

    rows, cols = dtm_patch.shape
    y, x = np.mgrid[0:rows, 0:cols] * pixel_size

    fig = plt.figure(figsize=(18, 8))

    # 3D-vy
    ax3d = fig.add_subplot(131, projection="3d")
    ax3d.plot_surface(x, y, dtm_patch, cmap="terrain", alpha=0.9,
                      linewidth=0, antialiased=True)
    ref = np.nanpercentile(dtm_patch, 95)
    ax3d.plot_surface(x, y, np.full_like(dtm_patch, ref),
                      color="cyan", alpha=0.25)
    ax3d.set_title(f"3D-vy: {label}", fontsize=11)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")

    # Djupkarta
    ax2 = fig.add_subplot(132)
    ref_border = np.nanmean(np.concatenate([
        dtm_patch[0, :], dtm_patch[-1, :],
        dtm_patch[:, 0], dtm_patch[:, -1]
    ]))
    depth_map = ref_border - dtm_patch
    depth_map[depth_map < 0] = 0
    im = ax2.imshow(depth_map, cmap="Blues_r", interpolation="bilinear")
    plt.colorbar(im, ax=ax2, label="Djup under referens (m)")
    ax2.set_title("Djupkarta\n(relativt kantmedelvärde)", fontsize=11)
    ax2.axis("off")

    # Volymjämförelse
    ax3 = fig.add_subplot(133)
    methods = list(results.keys())
    volumes = list(results.values())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax3.barh(methods, volumes, color=colors, edgecolor="white", height=0.5)
    for bar, vol in zip(bars, volumes):
        ax3.text(bar.get_width() + max(volumes)*0.01, bar.get_y() + bar.get_height()/2,
                 f"{vol:.1f} m³", va="center", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Volym (m³)")
    ax3.set_title("Volymer per metod\n(jämförelse)", fontsize=11)
    ax3.set_xlim(0, max(volumes) * 1.25)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.suptitle(f"Steg 4 — Volymanalys: {label}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"outputs/figures/04_volume_{label.replace(' ', '_')}.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\n📦 Volymresultat — {label}:")
    for method, vol in results.items():
        print(f"   {method:<28} {vol:>8.1f} m³")

    return results
```

---

## Steg 5 — Punktmolnsanalys

**Script:** `scripts/05_pointcloud_analysis.py`  
**Syfte:** Läs .laz-filen, klassificera punkter (mark/vegetation/struktur), beräkna täthetskartor.  
**Verktyg:** `laspy` för att läsa LAZ, `open3d` för 3D-visualisering och statistisk analys.  
**Visualisering:** Täthetskarta, klassificeringsöversikt, 3D-interaktiv vy.

```python
import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def load_pointcloud(laz_path):
    """Ladda LAZ-fil och returnera numpy arrays."""
    las = laspy.read(laz_path)
    x = las.x.scaled_array()
    y = las.y.scaled_array()
    z = las.z.scaled_array()

    # Klassificering (LAS standard)
    # 0=Oklassificerad, 1=Oklassificerad, 2=Mark, 3=Lågvegetation,
    # 4=Medelvegetation, 5=Högvegetation, 6=Byggnad
    classification = np.array(las.classification)

    print(f"✅ Punktmoln laddat: {len(x):,} punkter")
    print(f"   X: {x.min():.2f} — {x.max():.2f} m")
    print(f"   Y: {y.min():.2f} — {y.max():.2f} m")
    print(f"   Z: {z.min():.2f} — {z.max():.2f} m (höjdspann: {z.max()-z.min():.2f} m)")

    class_counts = {}
    class_names = {0:"Oklass", 2:"Mark", 3:"Lågveg", 4:"Medelveg", 5:"Högveg", 6:"Byggnad"}
    for k, v in class_names.items():
        count = (classification == k).sum()
        if count > 0:
            class_counts[v] = count
            print(f"   Klass {k} ({v}): {count:,} punkter ({100*count/len(x):.1f}%)")

    return x, y, z, classification

def plot_density_map(x, y, z, classification, resolution=1.0):
    """Skapa täthetskarta och höjdkarta från punktmolnet."""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    cols = int((x_max - x_min) / resolution) + 1
    rows = int((y_max - y_min) / resolution) + 1

    density = np.zeros((rows, cols))
    height_map = np.full((rows, cols), np.nan)

    xi = ((x - x_min) / resolution).astype(int)
    yi = ((y_max - y) / resolution).astype(int)

    xi = np.clip(xi, 0, cols-1)
    yi = np.clip(yi, 0, rows-1)

    for i in range(len(x)):
        density[yi[i], xi[i]] += 1
        if np.isnan(height_map[yi[i], xi[i]]) or z[i] > height_map[yi[i], xi[i]]:
            height_map[yi[i], xi[i]] = z[i]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Täthetskarta
    ax = axes[0]
    im = ax.imshow(np.log1p(density), cmap="inferno", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="log(punkttäthet + 1)")
    ax.set_title("Punkttäthetskarta\n(log-skala)", fontsize=11)
    ax.axis("off")

    # Höjdkarta från punktmoln
    ax = axes[1]
    im = ax.imshow(height_map, cmap="terrain", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Höjd (m)")
    ax.set_title("Höjdkarta från punktmoln\n(max Z per cell)", fontsize=11)
    ax.axis("off")

    # Klassificeringsöversikt
    ax = axes[2]
    class_names = {0:"Oklass", 2:"Mark", 3:"Lågveg", 4:"Medelveg", 5:"Högveg", 6:"Byggnad"}
    colors_map = {0:"#9E9E9E", 2:"#795548", 3:"#C5E1A5", 4:"#66BB6A", 5:"#1B5E20", 6:"#1565C0"}
    counts = [(class_names.get(k, str(k)), (classification==k).sum())
              for k in np.unique(classification)]
    counts = [(n, c) for n, c in counts if c > 0]
    names, vals = zip(*counts)
    bar_colors = [colors_map.get(k, "#607D8B") for k in np.unique(classification)
                  if (classification==k).sum() > 0]
    ax.barh(names, vals, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Antal punkter")
    ax.set_title("Punktfördelning per klass", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Steg 5 — Punktmolnsanalys", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/figures/05_pointcloud_analysis.png", dpi=200, bbox_inches="tight")
    plt.show()

def view_pointcloud_3d(x, y, z, classification, max_points=500_000):
    """Interaktiv 3D-visualisering i open3d."""
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z, classification = x[idx], y[idx], z[idx], classification[idx]

    # Färglägg efter klassificering
    color_map = {
        0: [0.6, 0.6, 0.6],   # Oklassificerad — grå
        2: [0.47, 0.33, 0.28], # Mark — brun
        3: [0.77, 0.88, 0.65], # Lågvegetation — ljusgrön
        4: [0.40, 0.73, 0.42], # Medelvegetation — grön
        5: [0.11, 0.37, 0.12], # Högvegetation — mörkgrön
        6: [0.08, 0.40, 0.78], # Byggnad — blå
    }
    colors = np.array([color_map.get(int(c), [0.5, 0.5, 0.5]) for c in classification])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x, y, z]))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("🖥️  Öppnar 3D-visualisering i open3d...")
    print("   Styr: vänsterklick+drag=rotera, scroll=zoom, shift+drag=panorera")
    o3d.visualization.draw_geometries([pcd],
                                       window_name="Punktmoln — WebODM Export",
                                       width=1200, height=800)
```

---

## Steg 6 — Rapportfigurer

**Script:** `scripts/06_report_figures.py`  
**Syfte:** Sammanställ ett snyggt figurblad med alla resultat, klart för kundrapport.  
**Visualisering:** En sida per objekt med höjdmodell, djup, volym och statistik.

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np

def generate_report_figure(dtm, depression_depth, volumes, object_id, area_m2,
                            project_name="Wietrznia"):
    """Generera en ensidig rapportfigur per analyserat objekt."""
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                           top=0.82, bottom=0.08, left=0.06, right=0.97)

    # Header
    fig.text(0.06, 0.92, f"Geodataanalys — {project_name}",
             fontsize=18, fontweight="bold", color="#1A237E")
    fig.text(0.06, 0.88, f"Objekt #{object_id}  |  Area: {area_m2:.0f} m²  |  Genererad: 2025",
             fontsize=11, color="#546E7A")
    fig.add_artist(plt.Line2D([0.06, 0.97], [0.86, 0.86],
                               transform=fig.transFigure, color="#1A237E", linewidth=2))

    # DTM
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(dtm, cmap="terrain")
    ax1.set_title("Höjdmodell (DTM)", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Fördjupningsdjup
    ax2 = fig.add_subplot(gs[0, 1])
    masked = np.ma.masked_where(depression_depth < 0.05, depression_depth)
    im = ax2.imshow(masked, cmap="Blues_r")
    plt.colorbar(im, ax=ax2, label="Djup (m)", fraction=0.046)
    ax2.set_title("Fördjupningsdjup", fontsize=10, fontweight="bold")
    ax2.axis("off")

    # 3D-profil
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    rows, cols = dtm.shape
    y, x = np.mgrid[0:rows, 0:cols]
    ax3.plot_surface(x, y, dtm, cmap="terrain", linewidth=0, antialiased=True, alpha=0.9)
    ax3.set_title("3D-vy", fontsize=10, fontweight="bold")
    ax3.axis("off")

    # Volymjämförelse
    ax4 = fig.add_subplot(gs[1, 0:2])
    methods = list(volumes.keys())
    vals = list(volumes.values())
    bar_colors = ["#1565C0", "#2E7D32", "#E65100"]
    bars = ax4.bar(methods, vals, color=bar_colors, edgecolor="white", width=0.5)
    for bar, vol in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                 f"{vol:.1f} m³", ha="center", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Volym (m³)")
    ax4.set_title("Volymer per beräkningsmetod", fontsize=10, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Statistikruta
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    stats = [
        ("Area", f"{area_m2:.1f} m²"),
        ("Max djup", f"{depression_depth.max():.2f} m"),
        ("Medeldjup", f"{depression_depth[depression_depth>0].mean():.2f} m"),
        ("Volym (medel)", f"{np.mean(vals):.1f} m³"),
        ("Volym (min/max)", f"{min(vals):.1f} / {max(vals):.1f} m³"),
    ]
    y_pos = 0.9
    for label, value in stats:
        ax5.text(0.05, y_pos, label, transform=ax5.transAxes,
                 fontsize=10, color="#546E7A")
        ax5.text(0.95, y_pos, value, transform=ax5.transAxes,
                 fontsize=11, fontweight="bold", ha="right", color="#1A237E")
        y_pos -= 0.18
    ax5.set_title("Sammanfattning", fontsize=10, fontweight="bold")
    rect = FancyBboxPatch((0, 0), 1, 1, transform=ax5.transAxes,
                           boxstyle="round,pad=0.02", linewidth=1.5,
                           edgecolor="#1A237E", facecolor="#E8EAF6", zorder=0)
    ax5.add_patch(rect)

    outpath = f"outputs/figures/06_report_object_{object_id}.png"
    plt.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"✅ Rapportfigur sparad: {outpath}")
```

---

## Körordning

```bash
# Skapa mappstruktur
mkdir -p data/raw data/derived outputs/figures outputs/reports scripts

# Placera WebODM-export i data/raw/
# (dsm.tif, dtm.tif, odm_orthophoto.tif, georeferenced_model.laz)

# Kör i ordning
python scripts/00_verify_inputs.py
python scripts/01_load_and_inspect.py
python scripts/02_terrain_analysis.py
python scripts/03_anomaly_detection.py
python scripts/04_volume_analysis.py
python scripts/05_pointcloud_analysis.py
python scripts/06_report_figures.py
```

---

## Vanliga problem och lösningar

| Problem | Orsak | Lösning |
|---|---|---|
| `CRS mismatch` | DSM och DTM i olika koordinatsystem | Reprojectera med `rasterio.warp.reproject` |
| `nodata = -9999` stör statistik | Nodata inte maskerad | Alltid: `data[data == nodata] = np.nan` |
| Punkt moln läses sakta | .laz är komprimerat | `laspy` hanterar detta, men chunked reading hjälper för >1 GB |
| Depression fill hittar inget | DTM är inte rensat | Kör `richdem.BreachDepressions` före `FillDepressions` |
| Volymer skiljer sig kraftigt | Referensplanet är avgörande | Jämför alltid alla tre metoder |

---

## Nästa steg (framtida expansion)

- [ ] Integrera Lantmäteriet-data (fastighetspolygoner via CC0 GeoJSON)
- [ ] Automatisk ROI-identifiering via machine learning (U-Net på ortofoto)
- [ ] PDF-rapportgenerering med `reportlab` eller `weasyprint`
- [ ] Tidsserianalys — jämför två flygningar och beräkna volymförändringar
- [ ] Exportera resultat som GeoJSON för visning i QGIS/Leaflet

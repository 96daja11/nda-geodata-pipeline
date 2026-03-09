# SmartTek – Dataset Demo Runners

Each script in this directory runs the full SmartTek 6-step inspection pipeline
against a specific publicly-available UAV / thermal dataset.
All demos are self-contained and run with:

```bash
python3 demos/<name>_demo.py
```

from the repository root (`/path/to/smarttek-demo/`).

---

## Demos

### 1. `pv_panel_demo.py` – Thermal PV Panel Detection Dataset

| Property | Value |
|---|---|
| Source | Roboflow Universe – "Thermal PV Panel Detection Dataset for UAV Inspection" |
| Sensor | DJI Mavic 3 Thermal (R-JPEG pseudo-colour) |
| Images used | 10 of 235 training images |
| Dataset location | `data/datasets/pv-panel/` |
| Status | **Ready** – dataset extracted |

Real DJI thermal JPEGs of solar-panel arrays are fed through the pipeline.
Temperature data is derived from image luminance (pixel 0 → 15 °C, pixel 255 → 85 °C).
The YOLO-based detector runs in mock mode; thermal anomaly extraction runs on real data.

**Last run findings:** Total=38, KRITISK=19, HOG=13, MEDEL=4, LAG=2

Output: `data/outputs/pv_panel/report/rapport_pv_panel.pdf`

```bash
python3 demos/pv_panel_demo.py
```

---

### 2. `taseg_demo.py` – TASeg Thermal Aerial Building Segmentation

| Property | Value |
|---|---|
| Source | TASeg project – `manual_set.zip` |
| Sensor | Thermal building imagery (NPY + PNG previews) |
| Images used | 15 of 538 preview images (train + val + test splits) |
| Dataset location | `data/datasets/taseg/manual_set/` |
| Status | **Ready** – dataset extracted |

Ground-truth segmentation label PNGs are parsed alongside thermal preview images.
Labels encode anomaly regions (bright pixels = thermal anomaly). The demo reports
the number of ground-truth anomaly regions found in the selected images.

**Last run findings:** Total=59, KRITISK=29, HOG=19, MEDEL=8, LAG=3

Output: `data/outputs/taseg/report/rapport_taseg.pdf`

```bash
python3 demos/taseg_demo.py
```

---

### 3. `hit_uav_demo.py` – HIT-UAV High-altitude Infrared Thermal Survey

| Property | Value |
|---|---|
| Source | HIT-UAV – A High-altitude Infrared Thermal Dataset for UAV-based Object Detection |
| Sensor | Infrared thermal camera, altitudes 30–120 m, angles 0–60° |
| Annotation format | Pascal VOC XML (classes: Car, Person, Bicycle, OtherVehicle, DontCare) |
| Images used | 15 of 2898 images |
| Dataset location | `data/datasets/hit-uav/suojiashun-HIT-UAV-Infrared-Thermal-Dataset-b53106c/normal_xml/` |
| Geographic centre | Harbin, China (45.7722 N, 126.6369 E) |
| Status | **Ready** – dataset extracted |

Demonstrates thermal analysis on diverse outdoor scenes (parking lots, playgrounds, roads).
Framed as a "multi-scene thermal survey" to highlight the pipeline's versatility beyond
building inspection. VOC XML annotations are parsed to log the ground-truth object counts
(persons, cars, etc.) for the selected images, enriching the demo with real detection context.

Filename encoding: `<scene>_<altitude>_<angle>_<sequence>_<frame>.jpg`

**Last run findings:**
- Ground-truth objects: Bicycle=54, Car=47, DontCare=2, Person=69
- Pipeline findings: Total=59, KRITISK=29, HOG=19, MEDEL=8, LAG=3

Output: `data/outputs/hit_uav/report/rapport_hit_uav.pdf`

```bash
python3 demos/hit_uav_demo.py
```

---

### 4. `uavid3d_demo.py` – UAVID3D UAV Thermal Building Inspection

| Property | Value |
|---|---|
| Source | UAVID3D – Blume and Olympic club building surveys (May 2021) |
| Sensor | DJI thermal camera |
| Images used | 15 of 374 total thermal images (Blume + Olympic combined) |
| Dataset location | `data/datasets/uavid3d/` |
| Geographic centre | Bochum / NRW, Germany (51.4818 N, 7.2162 E) |
| Status | **Ready** – all datasets extracted |

Combines thermal imagery from two separate building sites:

- **Blume** (`Blume_drone_data_capture_may2021/thermal/.../normalised/`): 129 normalised DJI thermal JPEGs of a commercial building.
- **Olympic club** (`Olympic_club_drone_data_capture_may2021/thermal_images/`): 245 DJI thermal JPEGs across 4 project captures (Projects 00016–00019, ~90+90+30+35 images).

The 4 Olympic ZIP archives are extracted to sub-folders at import time (Python `zipfile`, no `unzip` required).

**Last run findings:** Total=59, KRITISK=29, HOG=19, MEDEL=8, LAG=3

Output: `data/outputs/uavid3d/report/rapport_uavid3d.pdf`

```bash
python3 demos/uavid3d_demo.py
```

---

## Run all ready demos at once

```bash
python3 demos/run_all_demos.py
```

The master runner checks which datasets are extracted, runs all ready demos,
and prints a summary table with paths to all generated reports.

---

## Shared utilities – `dataset_adapter.py`

| Function | Description |
|---|---|
| `load_thermal_from_jpeg(path)` | Decode a thermal JPEG to a 2-D temperature array (°C) |
| `extract_anomalies(temp_array, threshold)` | Find hotspot / cold-bridge regions |
| `mock_gps_coords(n, center_lat, center_lon)` | Generate GPS coordinates near a centre point |
| `run_pipeline_on_images(rgb_paths, thermal_paths, output_dir, dataset_name)` | Run full 6-step pipeline |

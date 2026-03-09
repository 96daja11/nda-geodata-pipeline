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
| Images | 235 training images |
| Status | **Ready** – dataset already extracted at `data/datasets/pv-panel/` |

Real DJI thermal JPEGs of solar-panel arrays are fed through the pipeline.
Temperature data is derived from image luminance (pixel 0 → 15 °C, pixel 255 → 85 °C).
The YOLO-based detector runs in mock mode; thermal anomaly extraction runs on real data.

Output: `data/outputs/pv_panel/report/rapport_pv_panel.pdf`

---

### 2. `taseg_demo.py` – TASeg Thermal Aerial Building Segmentation
| Property | Value |
|---|---|
| Source | TASeg project – `manual_set.zip` |
| Sensor | Thermal building imagery |
| Status | ZIP downloaded – **needs extraction** |

Extract:
```bash
cd data/datasets/taseg && unzip manual_set.zip
```

Output: `data/outputs/taseg/report/rapport_taseg.pdf`

---

### 3. `hit_uav_demo.py` – HIT-UAV High-altitude Infrared Thermal
| Property | Value |
|---|---|
| Source | HIT-UAV dataset – `HIT-UAV.zip` |
| Sensor | Infrared thermal camera, multiple altitudes |
| Status | ZIP downloaded – **needs extraction** |

Extract:
```bash
cd data/datasets/hit-uav && unzip HIT-UAV.zip
```

Output: `data/outputs/hit_uav/report/rapport_hit_uav.pdf`

---

### 4. `uavid3d_demo.py` – UAVID3D Urban 3D Scene Understanding
| Property | Value |
|---|---|
| Source | UAVID3D – `Blume_004.zip` + `Olympic_004.zip` |
| Sensor | High-resolution oblique RGB |
| Status | ZIPs downloaded – **needs extraction** |

Extract:
```bash
cd data/datasets/uavid3d
unzip Blume_004.zip
unzip Olympic_004.zip
```

Output: `data/outputs/uavid3d/report/rapport_uavid3d.pdf`

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
| `mock_gps_coords(n, center_lat, center_lon)` | Generate Gothenburg-centred GPS points |
| `run_pipeline_on_images(rgb_paths, thermal_paths, output_dir, dataset_name)` | Run full 6-step pipeline |

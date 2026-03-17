# Geodata Analysis Pipeline

End-to-end analysis pipeline for drone survey data exported from WebODM.
Produces terrain analysis, anomaly detection, volume estimates, point cloud
statistics and a PDF report — all from a single command.

## Quick start

```bash
# Run the full pipeline on a dataset directory
./run_pipeline.sh /path/to/dataset

# Examples
./run_pipeline.sh /mnt/storage4tb/smarttek-demo/webodm_sampledata
./run_pipeline.sh /mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia
```

The dataset directory must contain `data/raw/{dtm.tif,dsm.tif,odm_orthophoto.tif,georeferenced_model.laz}`.
Outputs are written to `data/raw/../outputs/figures/` and `outputs/reports/`.

## Dataset configuration — `config.json`

Place a `config.json` file in the dataset root directory to override pipeline
parameters. If no `config.json` is present, sensible defaults are used.

```json
{
  "dataset_name": "My Dataset",
  "dataset_slug": "my-dataset",
  "crs": "EPSG:32633",
  "anomaly": {
    "radii_m": [2.5, 5.0, 10.0],
    "min_area_m2": 2.0,
    "threshold_depression_m": -0.25,
    "threshold_elevation_m": 0.30
  },
  "tree_analysis": {
    "enabled": true,
    "sigma": 3.0,
    "veg_threshold_m": 1.5,
    "min_distance_px": 100,
    "tree_height_min_m": 3.0
  },
  "report": {
    "title": "My Dataset — Geodataanalys",
    "footer_dataset": "My Dataset 2024",
    "pdf_name": "geodata_analys_rapport_my_dataset.pdf"
  }
}
```

### Config fields

| Key | Description | Default |
|-----|-------------|---------|
| `anomaly.radii_m` | Multi-scale analysis radii in metres | `[2.5, 5.0, 10.0]` |
| `anomaly.min_area_m2` | Minimum anomaly region area to report | `2.0` |
| `anomaly.threshold_depression_m` | Negative deviation threshold | `-0.25` |
| `anomaly.threshold_elevation_m` | Positive deviation threshold | `0.30` |
| `tree_analysis.enabled` | Run step 5b tree analysis | `true` |
| `tree_analysis.sigma` | Gaussian smoothing sigma (pixels) | `3.0` |
| `tree_analysis.veg_threshold_m` | Min CHM height for vegetation | `1.5` |
| `tree_analysis.min_distance_px` | Min pixel distance between tree tops | `100` |
| `tree_analysis.tree_height_min_m` | Min tree top height | `3.0` |
| `report.pdf_name` | Output PDF filename | `geodata_analys_rapport.pdf` |
| `report.title` | Dataset title in report | `"Geodataanalys"` |
| `report.footer_dataset` | Footer text in report | `"Geodataanalys 2022"` |

## Pipeline steps

| Step | Script | Description |
|------|--------|-------------|
| 0 | `00_verify_inputs.py` | Validate all input files, CRS consistency, nodata |
| 1 | `01_load_and_inspect.py` | DSM/DTM/CHM overview, elevation profile |
| 2 | `02_terrain_analysis.py` | Slope, aspect, hillshade, slope classification |
| 3 | `03_anomaly_detection.py` | Multi-scale local deviation, depression/elevation detection |
| 4 | `04_volume_analysis.py` | Volume estimation via 3 reference plane methods |
| 5 | `05_pointcloud_analysis.py` | LAZ density map, height map, classification stats |
| 5b | `05b_tree_analysis.py` | Individual tree detection via CHM + watershed (optional) |
| 6 | `06_report_figures.py` | Full summary figure (4×3 grid) |
| 7 | `07_generate_pdf_report.py` | HTML + PDF report via WeasyPrint |

Each script accepts `--project-dir <path>` to point at the dataset root.

## Running a single step

```bash
source /path/to/.venv/bin/activate

python3 scripts/03_anomaly_detection.py \
  --project-dir /mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia
```

## WebODM integration

`webodm_poll.sh` polls a running WebODM task, downloads the outputs when
complete, and then invokes `run_pipeline.sh` automatically:

```bash
./webodm_poll.sh /path/to/dataset \
  --task-id  <uuid>   \
  --url      http://localhost:8010 \
  --user     admin \
  --password admin123
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU acceleration (CuPy) is optional. Each script falls back to NumPy/CPU
if CuPy is not available or no CUDA GPU is present.

## Included datasets

| Directory | config.json | Tree analysis |
|-----------|------------|---------------|
| `webodm_sampledata/` | Güterweg Ritzing — open terrain, tight thresholds | disabled |
| `webodm_sampledata_wietrznia/` | Wietrznia — forested, relaxed thresholds | enabled |

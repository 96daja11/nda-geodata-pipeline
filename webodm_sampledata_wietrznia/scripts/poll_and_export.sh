#!/usr/bin/env bash
# poll_and_export.sh — Wietrznia dataset pipeline runner
# Data is already exported from WebODM. This script runs the full analysis pipeline directly.
set -euo pipefail

TASK_ID="WIETRZNIA_MANUAL"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV="/mnt/storage4tb/smarttek-demo/webodm_sampledata/.venv/bin/activate"

echo "============================================================"
echo " Geodata Analysis Pipeline — Wietrznia"
echo " Dataset: ${TASK_ID}"
echo "============================================================"

# Verify data exists
if [ ! -f "$BASE_DIR/data/raw/dsm.tif" ] || [ ! -f "$BASE_DIR/data/raw/dtm.tif" ]; then
  echo "ERROR: Rasterdata saknas i $BASE_DIR/data/raw/"
  echo "       Kontrollera att dsm.tif, dtm.tif, odm_orthophoto.tif och"
  echo "       georeferenced_model.laz finns pa plats."
  exit 1
fi

echo ""
echo "Data hittad i $BASE_DIR/data/raw/"
ls -lh "$BASE_DIR/data/raw/"

# Activate virtual environment
source "$VENV"

cd "$BASE_DIR"

echo ""
echo "[1/9] Validerar ingångsdata..."
python3 scripts/00_verify_inputs.py

echo ""
echo "[2/9] Laddar och inspekterar höjdmodell..."
python3 scripts/01_load_and_inspect.py

echo ""
echo "[3/9] Terränganalys..."
python3 scripts/02_terrain_analysis.py

echo ""
echo "[4/9] Anomalidetektering..."
python3 scripts/03_anomaly_detection.py

echo ""
echo "[5/9] Volymanalys..."
python3 scripts/04_volume_analysis.py

echo ""
echo "[6/9] Punktmolnsanalys..."
python3 scripts/05_pointcloud_analysis.py

echo ""
echo "[7/9] Trädanalys (CHM, watershed, individuella träd)..."
python3 scripts/05b_tree_analysis.py

echo ""
echo "[8/9] Rapportfigurer (sammanfattningsgrid)..."
python3 scripts/06_report_figures.py

echo ""
echo "[9/9] Genererar PDF-rapport..."
python3 scripts/07_generate_pdf_report.py

echo ""
echo "============================================================"
echo " KOMPLETT ANALYS KLAR"
echo " Figurer:   $BASE_DIR/outputs/figures/"
echo " Rapporter: $BASE_DIR/outputs/reports/"
echo "============================================================"
ls -la "$BASE_DIR/outputs/figures/" 2>/dev/null || echo "(inga figurer ännu)"

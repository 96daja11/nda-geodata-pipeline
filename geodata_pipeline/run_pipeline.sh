#!/usr/bin/env bash
# run_pipeline.sh — Run the full geodata analysis pipeline on a dataset
# Usage: ./run_pipeline.sh <project-dir> [--venv <venv-path>]
#
# Example:
#   ./run_pipeline.sh /mnt/storage4tb/smarttek-demo/webodm_sampledata
#   ./run_pipeline.sh /mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${1:?Usage: $0 <project-dir>}"
PROJECT_DIR="$(realpath "$PROJECT_DIR")"

# Find venv: --venv arg, or PROJECT_DIR/.venv, or SCRIPT_DIR/../webodm_sampledata/.venv
VENV_ARG="${2:-}"
if [[ "$VENV_ARG" == "--venv" ]]; then
  VENV="$(realpath "${3:?--venv requires a path}")/bin/activate"
elif [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  VENV="$PROJECT_DIR/.venv/bin/activate"
else
  VENV="$(realpath "$SCRIPT_DIR/../webodm_sampledata/.venv")/bin/activate"
fi

echo "============================================================"
echo " Geodata Analysis Pipeline"
echo " Dataset:  $PROJECT_DIR"
echo " Scripts:  $SCRIPT_DIR/scripts"
echo " Venv:     $VENV"
echo "============================================================"

# Verify data exists
if [[ ! -f "$PROJECT_DIR/data/raw/dtm.tif" ]]; then
  echo "ERROR: $PROJECT_DIR/data/raw/dtm.tif not found."
  echo "       Export data from WebODM before running the pipeline."
  exit 1
fi

source "$VENV"

echo ""
echo "[1/8] Validating inputs..."
python3 "$SCRIPT_DIR/scripts/00_verify_inputs.py"       --project-dir "$PROJECT_DIR"

echo ""
echo "[2/8] Load and inspect..."
python3 "$SCRIPT_DIR/scripts/01_load_and_inspect.py"    --project-dir "$PROJECT_DIR"

echo ""
echo "[3/8] Terrain analysis..."
python3 "$SCRIPT_DIR/scripts/02_terrain_analysis.py"    --project-dir "$PROJECT_DIR"

echo ""
echo "[4/8] Anomaly detection..."
python3 "$SCRIPT_DIR/scripts/03_anomaly_detection.py"   --project-dir "$PROJECT_DIR"

echo ""
echo "[5/8] Volume analysis..."
python3 "$SCRIPT_DIR/scripts/04_volume_analysis.py"     --project-dir "$PROJECT_DIR"

echo ""
echo "[6/8] Point cloud analysis..."
python3 "$SCRIPT_DIR/scripts/05_pointcloud_analysis.py" --project-dir "$PROJECT_DIR"

echo ""
echo "[7/8] Tree analysis (optional, skips if disabled in config.json)..."
python3 "$SCRIPT_DIR/scripts/05b_tree_analysis.py"      --project-dir "$PROJECT_DIR" || true

echo ""
echo "[7/8] Report figures..."
python3 "$SCRIPT_DIR/scripts/06_report_figures.py"      --project-dir "$PROJECT_DIR"

echo ""
echo "[8/8] Generate PDF report..."
python3 "$SCRIPT_DIR/scripts/07_generate_pdf_report.py" --project-dir "$PROJECT_DIR"

echo ""
echo "============================================================"
echo " PIPELINE COMPLETE"
echo " Figures: $PROJECT_DIR/outputs/figures/"
echo " Reports: $PROJECT_DIR/outputs/reports/"
echo "============================================================"
ls -lh "$PROJECT_DIR/outputs/figures/" 2>/dev/null | tail -10

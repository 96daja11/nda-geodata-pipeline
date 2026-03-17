#!/usr/bin/env bash
# Kör hela analyskedjan i ordning
# Projekt: Oldsmar Trail — WebODM Task 974e55a0
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

echo "════════════════════════════════════════════════════════════"
echo "  GEODATAANALYS — OLDSMAR TRAIL"
echo "  WebODM Task: 974e55a0-05ce-41bd-b398-81fe8001af6b"
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$SCRIPT_DIR/outputs/figures" "$SCRIPT_DIR/outputs/reports"

run_step() {
    local step="$1"
    local script="$2"
    echo "┌─────────────────────────────────────────────────────────"
    echo "│ $step"
    echo "└─────────────────────────────────────────────────────────"
    "$PYTHON" "$SCRIPT_DIR/scripts/$script"
    echo ""
}

run_step "Steg 0 — Validera ingångsdata"       "00_verify_inputs.py"
run_step "Steg 1 — Ladda och visualisera"       "01_load_and_inspect.py"
run_step "Steg 2 — Terränganalys"               "02_terrain_analysis.py"
run_step "Steg 3 — Anomalidetektion"            "03_anomaly_detection.py"
run_step "Steg 4 — Volymanalys"                 "04_volume_analysis.py"
run_step "Steg 5 — Punktmolnsanalys"            "05_pointcloud_analysis.py"
run_step "Steg 6 — Rapportfigurer"              "06_report_figures.py"

echo "════════════════════════════════════════════════════════════"
echo "  ✅ PIPELINE KLAR"
echo "  Figurer: $SCRIPT_DIR/outputs/figures/"
echo "  Rapporter: $SCRIPT_DIR/outputs/reports/"
echo "════════════════════════════════════════════════════════════"
ls -lh "$SCRIPT_DIR/outputs/figures/"

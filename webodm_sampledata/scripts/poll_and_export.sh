#!/usr/bin/env bash
# poll_and_export.sh — Polls WebODM until task is complete, then downloads DTM and re-runs analysis
set -euo pipefail

WEBODM_URL="http://localhost:8010"
PROJECT_ID="2"
TASK_ID="693fb877-7297-45d8-a608-a662512101a2"
WEBODM_USER="admin"
WEBODM_PASS="admin123"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_RAW="$BASE_DIR/data/raw"
VENV="$BASE_DIR/.venv/bin/activate"

echo "============================================================"
echo " WebODM Export + Analys Pipeline"
echo " Task: $TASK_ID"
echo "============================================================"

# Get JWT token
get_token() {
  curl -s -X POST "$WEBODM_URL/api/token-auth/" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$WEBODM_USER\",\"password\":\"$WEBODM_PASS\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))"
}

# Poll until complete or failed
echo ""
echo "Polling WebODM task status..."
while true; do
  TOKEN=$(get_token)
  RESP=$(curl -s "$WEBODM_URL/api/projects/$PROJECT_ID/tasks/$TASK_ID/?format=json" \
    -H "Authorization: JWT $TOKEN")

  STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))")
  PROGRESS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('running_progress',0)*100:.1f}\")")
  ASSETS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(' '.join(d.get('available_assets',[])))")

  TIMESTAMP=$(date '+%H:%M:%S')

  if [ "$STATUS" = "40" ]; then
    echo "[$TIMESTAMP] ✅ Task COMPLETED!"
    break
  elif [ "$STATUS" = "30" ]; then
    echo "[$TIMESTAMP] ❌ Task FAILED. Kontrollera WebODM-loggen."
    exit 1
  else
    echo "[$TIMESTAMP] ⏳ Status: RUNNING — ${PROGRESS}%"
  fi

  sleep 30
done

# Download DTM if available
echo ""
echo "Laddar ner assets..."
TOKEN=$(get_token)

# Check if dtm.tif is in available assets
if echo "$ASSETS" | grep -q "dtm.tif"; then
  echo "  Laddar ner dtm.tif..."
  curl -s -L "$WEBODM_URL/api/projects/$PROJECT_ID/tasks/$TASK_ID/download/dtm.tif" \
    -H "Authorization: JWT $TOKEN" \
    -o "$DATA_RAW/dtm.tif"
  echo "  ✅ dtm.tif sparad till $DATA_RAW/dtm.tif"
else
  echo "  ⚠️  dtm.tif finns inte i task-assets. Använder befintlig DTM."
fi

# Also download fresh DSM and orthophoto for completeness
echo "  Laddar ner dsm.tif..."
curl -s -L "$WEBODM_URL/api/projects/$PROJECT_ID/tasks/$TASK_ID/download/dsm.tif" \
  -H "Authorization: JWT $TOKEN" \
  -o "$DATA_RAW/dsm.tif"
echo "  ✅ dsm.tif uppdaterad"

# Re-run the full analysis pipeline
echo ""
echo "============================================================"
echo " Kör analyskedjan om med ny data..."
echo "============================================================"

source "$VENV"

cd "$BASE_DIR"

echo ""
echo "[1/6] Validerar ingångsdata..."
python3 scripts/00_verify_inputs.py

echo ""
echo "[2/6] Laddar och inspekterar..."
python3 scripts/01_load_and_inspect.py

echo ""
echo "[3/6] Terränganalys..."
python3 scripts/02_terrain_analysis.py

echo ""
echo "[4/6] Anomalidetektering..."
python3 scripts/03_anomaly_detection.py

echo ""
echo "[5/6] Volymanalys..."
python3 scripts/04_volume_analysis.py

echo ""
echo "[6/6] Punktmolnsanalys..."
python3 scripts/05_pointcloud_analysis.py

echo ""
echo "[7/7] Rapportfigurer..."
python3 scripts/06_report_figures.py

echo ""
echo "[8/8] Genererar PDF-rapport..."
python3 scripts/07_generate_pdf_report.py

echo ""
echo "============================================================"
echo " ✅ KOMPLETT ANALYS KLAR"
echo " Figurer: $BASE_DIR/outputs/figures/"
echo " Rapporter: $BASE_DIR/outputs/reports/"
echo "============================================================"
ls -la "$BASE_DIR/outputs/figures/"

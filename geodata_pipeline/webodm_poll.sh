#!/usr/bin/env bash
# webodm_poll.sh — Poll WebODM task until complete, download assets, then run pipeline
# Usage: ./webodm_poll.sh <project-dir> [options]
#
# Options:
#   --task-id     <id>     WebODM task UUID (required)
#   --project-id  <id>     WebODM project ID (default: 1)
#   --url         <url>    WebODM URL (default: http://localhost:8010)
#   --user        <user>   WebODM username (default: admin)
#   --password    <pass>   WebODM password (default: admin123)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${1:?Usage: $0 <project-dir> --task-id <id>}"
PROJECT_DIR="$(realpath "$PROJECT_DIR")"
shift

WEBODM_URL="http://localhost:8010"
PROJECT_ID="1"
TASK_ID=""
WEBODM_USER="admin"
WEBODM_PASS="admin123"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id)    TASK_ID="$2";     shift 2 ;;
    --project-id) PROJECT_ID="$2";  shift 2 ;;
    --url)        WEBODM_URL="$2";  shift 2 ;;
    --user)       WEBODM_USER="$2"; shift 2 ;;
    --password)   WEBODM_PASS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

[[ -z "$TASK_ID" ]] && { echo "ERROR: --task-id is required"; exit 1; }

DATA_RAW="$PROJECT_DIR/data/raw"
mkdir -p "$DATA_RAW"

echo "============================================================"
echo " WebODM Poll + Export"
echo " URL:      $WEBODM_URL"
echo " Task:     $TASK_ID"
echo " Dataset:  $PROJECT_DIR"
echo "============================================================"

get_token() {
  curl -s -X POST "$WEBODM_URL/api/token-auth/" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$WEBODM_USER\",\"password\":\"$WEBODM_PASS\"}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))"
}

echo ""
echo "Polling task status..."
while true; do
  TOKEN=$(get_token)
  RESP=$(curl -s "$WEBODM_URL/api/projects/$PROJECT_ID/tasks/$TASK_ID/?format=json" \
    -H "Authorization: JWT $TOKEN")
  STATUS=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))")
  PROGRESS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('running_progress',0)*100:.1f}\")")
  ASSETS=$(echo "$RESP"   | python3 -c "import sys,json; d=json.load(sys.stdin); print(' '.join(d.get('available_assets',[])))")

  if [[ "$STATUS" == "40" ]]; then
    echo "[$(date '+%H:%M:%S')] Task COMPLETED"
    break
  elif [[ "$STATUS" == "30" ]]; then
    echo "[$(date '+%H:%M:%S')] Task FAILED — check WebODM logs"
    exit 1
  else
    echo "[$(date '+%H:%M:%S')] Running — ${PROGRESS}%"
  fi
  sleep 30
done

TOKEN=$(get_token)

echo ""
echo "Downloading assets..."
for ASSET in dtm.tif dsm.tif odm_orthophoto.tif georeferenced_model.laz; do
  if echo "$ASSETS" | grep -q "$ASSET"; then
    echo "  Downloading $ASSET..."
    curl -s -L "$WEBODM_URL/api/projects/$PROJECT_ID/tasks/$TASK_ID/download/$ASSET" \
      -H "Authorization: JWT $TOKEN" \
      -o "$DATA_RAW/$ASSET"
    echo "  Saved: $DATA_RAW/$ASSET"
  fi
done

echo ""
echo "Running analysis pipeline..."
"$SCRIPT_DIR/run_pipeline.sh" "$PROJECT_DIR"

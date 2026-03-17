#!/usr/bin/env python3
"""
North Drone Analytics — Potree Server (port 9001)
Serves static Potree files and provides a tree analysis API endpoint.
"""
import json
import os
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SERVE_DIR       = Path(__file__).parent.resolve()
PROJECT_WIETRZNIA = Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata_wietrznia")
SCRIPT_PATH     = Path("/mnt/storage4tb/smarttek-demo/geodata_pipeline/scripts/05b_tree_analysis.py")
VENV_PYTHON     = Path("/mnt/storage4tb/smarttek-demo/webodm_sampledata/.venv/bin/python3")
RESULT_PATH     = PROJECT_WIETRZNIA / "outputs/reports/05b_tree_analysis.json"

# ── Job state ──────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_job  = {"running": False, "result": None, "error": None, "params": None, "started_at": None}


class NDAHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    # ── Routing ────────────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self._cors(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/tree_tune":
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n)) if n else {}
                self._tree_tune(body)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
        else:
            self.send_response(404); self.end_headers()

    def do_GET(self):
        if self.path == "/api/tree_status":
            self._tree_status()
        elif self.path == "/api/tree_results":
            self._tree_results()
        else:
            super().do_GET()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _cors(self, code):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self._cors(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    # ── API handlers ───────────────────────────────────────────────────────────
    def _tree_tune(self, body):
        global _job
        with _lock:
            if _job["running"]:
                self._json({"error": "En analys pågår redan — vänta tills den är klar."}, 409)
                return

        sigma       = float(body.get("sigma",       3.0))
        min_dist_m  = float(body.get("min_dist_m",  5.0))
        veg_th      = float(body.get("veg_th",      1.5))
        height_min  = float(body.get("height_min",  3.0))
        pixel_size  = 0.05   # CHM resolution (5 cm/px)
        min_dist_px = max(1, round(min_dist_m / pixel_size))

        # Update config.json with new parameters
        cfg_path = PROJECT_WIETRZNIA / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["tree_analysis"].update({
            "sigma":           sigma,
            "veg_threshold_m": veg_th,
            "min_distance_px": min_dist_px,
            "tree_height_min_m": height_min,
        })
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        params = {
            "sigma":        sigma,
            "min_dist_m":   min_dist_m,
            "min_dist_px":  min_dist_px,
            "veg_th":       veg_th,
            "height_min":   height_min,
        }
        with _lock:
            _job = {"running": True, "result": None, "error": None,
                    "params": params, "started_at": time.time()}

        def _run():
            global _job
            python = str(VENV_PYTHON) if VENV_PYTHON.exists() else "python3"
            try:
                proc = subprocess.run(
                    [python, str(SCRIPT_PATH), "--project-dir", str(PROJECT_WIETRZNIA)],
                    capture_output=True, text=True, timeout=480,
                    cwd=str(PROJECT_WIETRZNIA),
                )
                if proc.returncode == 0 and RESULT_PATH.exists():
                    with open(RESULT_PATH) as f:
                        result = json.load(f)
                    with _lock:
                        _job.update(running=False, result=result)
                else:
                    tail = (proc.stderr or proc.stdout or "Okänt fel")[-3000:]
                    with _lock:
                        _job.update(running=False, error=tail)
            except subprocess.TimeoutExpired:
                with _lock:
                    _job.update(running=False, error="Timeout — analysen tog >8 min")
            except Exception as exc:
                with _lock:
                    _job.update(running=False, error=str(exc))

        threading.Thread(target=_run, daemon=True).start()
        self._json({"started": True, "params": params,
                    "message": "Analys startad. Polla /api/tree_status för status."})

    def _tree_status(self):
        with _lock:
            snap = dict(_job)
        # Trim large result to summary fields only
        if snap.get("result"):
            r = snap["result"]
            snap["result"] = {
                "tree_count":          r.get("tree_count"),
                "mean_height_m":       r.get("mean_height_m"),
                "mean_crown_area_m2":  r.get("mean_crown_area_m2"),
                "total_canopy_area_m2": r.get("total_canopy_area_m2"),
            }
        if snap.get("started_at"):
            snap["elapsed_s"] = round(time.time() - snap["started_at"], 1)
        self._json(snap)

    def _tree_results(self):
        if RESULT_PATH.exists():
            with open(RESULT_PATH) as f:
                self._json(json.load(f))
        else:
            self._json({"error": "Inga resultat tillgängliga — kör analysen först"}, 404)

    def log_message(self, fmt, *args):
        print(f"  {args[0]}  {args[1]}  {self.path}")


if __name__ == "__main__":
    os.chdir(str(SERVE_DIR))
    print(f"\n{'='*55}")
    print(f"  North Drone Analytics — Potree Server")
    print(f"  http://0.0.0.0:9001/")
    print(f"  Static: {SERVE_DIR}")
    print(f"  API:    /api/tree_tune  /api/tree_status  /api/tree_results")
    print(f"{'='*55}\n")
    HTTPServer(("0.0.0.0", 9001), NDAHandler).serve_forever()

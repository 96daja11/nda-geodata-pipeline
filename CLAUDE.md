# CLAUDE.md – smarttek-demo

## Vad det här repot är

En end-to-end demonstrationspipeline för North Drone Analytics:s drönbaserade fastighetsinspektion.
Pipelinens ingång är råbilder (RGB + termisk) från en DJI Mavic 3 Thermal.
Pipelinens utdata är en strukturerad inspektionsrapport i PDF-format med georefererade fynd,
temperatursignaturer och prioriterade åtgärdsrekommendationer.

Syftet med det här repot är att:
1. Kunna köras lokalt med `docker compose up` på exempeldata
2. Visa hela kedjan i en teknisk demo för potentiella kunder
3. Fungera som bas för produktionssystemet längre fram

## Stack och beroenden

**Språk:** Python 3.11+
**Paketering:** `uv` (föredras framför pip/poetry), med `pyproject.toml`
**Containerisering:** Docker + Docker Compose
**Fotogrammetri:** OpenDroneMap (ODM) via WebODM REST API, körs i egen container
**Termisk analys:** `flirimageextractor` + `flirpy` för att extrahera temperaturmatriser
**GIS/raster:** `rasterio`, `geopandas`, `shapely`
**AI-detektering:** `ultralytics` (YOLOv8/YOLO11), modellvikter i `models/`
**Rapportgenerering:** `WeasyPrint` (HTML→PDF) med Jinja2-mallar
**API:** `FastAPI` + `uvicorn`
**Testning:** `pytest`

## Repostruktur

```
smarttek-demo/
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── README.md
├── CLAUDE.md                   ← den här filen
│
├── data/
│   ├── sample/
│   │   ├── rgb/                # Exempelbilder i JPEG/DNG
│   │   └── thermal/            # Termiska bilder, FLIR R-JPEG-format
│   └── outputs/                # Genereras automatiskt, gitignoreras
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest/                 # Steg 1: validering och metadata
│   │   ├── __init__.py
│   │   ├── validator.py        # Kontrollerar filformat, GPS-metadata
│   │   └── models.py           # Pydantic-modeller för indata
│   ├── photogrammetry/         # Steg 2: ODM-integration
│   │   ├── __init__.py
│   │   └── odm_client.py       # WebODM API-klient, skapar task, pollar status
│   ├── thermal/                # Steg 3: termisk extraktion
│   │   ├── __init__.py
│   │   └── extractor.py        # Läser R-JPEG, returnerar temperaturmatris + GPS
│   ├── detection/              # Steg 4: AI-skadedetektering
│   │   ├── __init__.py
│   │   └── detector.py         # YOLOv8-inferens på RGB + termisk
│   ├── analysis/               # Steg 5: GIS-analys och prioritering
│   │   ├── __init__.py
│   │   └── analyzer.py         # Georefererar fynd, klassificerar allvarlighetsgrad
│   └── report/                 # Steg 6: rapportgenerering
│       ├── __init__.py
│       ├── generator.py        # Sammanställer rapport från analysresultat
│       └── templates/
│           └── report.html     # Jinja2-mall för inspektionsrapporten
│
├── models/
│   └── crack_detection/
│       └── README.md           # Instruktioner för att ladda ner vikter
│
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI-app
│   └── routes/
│       ├── __init__.py
│       └── pipeline.py         # POST /run – kör pipeline, returnerar rapport-URL
│
├── frontend/
│   ├── index.html              # Demo-UI: filuppladdning, pipeline-status, länk till rapport
│   └── assets/
│       └── logo.png            # North Drone Analytics-logotyp (används i rapport)
│
└── tests/
    ├── conftest.py
    ├── test_ingest.py
    ├── test_thermal.py
    ├── test_detection.py
    └── sample_run.py           # Kör hela pipeline end-to-end på exempeldata
```

## Hur pipeline:n fungerar

Varje steg i pipeline är ett separat Python-paket med en tydlig in- och uttyp (Pydantic-modell).
Stegen är designade att kunna köras oberoende för testning, men körs sekventiellt i demo-läge.

```
IngestResult → PhotogrammetryResult → ThermalResult → DetectionResult → AnalysisResult → Report (PDF)
```

### Steg 1 – Ingest (`pipeline/ingest/`)
- Tar emot en katalog med bilder
- Validerar filformat (JPEG, DNG, R-JPEG)
- Extraherar GPS-koordinater och kamerametadata via exiftool
- Returnerar `IngestResult` med fillistor och metadata

### Steg 2 – Fotogrammetri (`pipeline/photogrammetry/`)
- Skickar RGB-bilder till WebODM REST API
- Pollar status tills task är klar
- Laddar ner ortofoto (GeoTIFF) och punktmoln (LAZ)
- I demo-läge: om WebODM inte är tillgänglig, använd förberäknat exempelortofoto

### Steg 3 – Termisk extraktion (`pipeline/thermal/`)
- Läser termiska R-JPEG-filer med `flirimageextractor`
- Returnerar temperaturmatris (numpy array, grader Celsius per pixel) + GPS per bild
- Identifierar termiska avvikelser: pixlar >X°C avvikelse från omgivande medelvärde
- Parametern X är konfigurerbar via `.env`

### Steg 4 – Detektering (`pipeline/detection/`)
- Kör YOLOv8-inferens på RGB-bilder (sprickor, skador, vattenansamlingar)
- Kör temperaturtröskelanalys på termiska matriser (köldbryggor, fukt, läckor)
- Returnerar lista av `Finding` med typ, konfidensgrad, boundingbox och bildkoordinater

### Steg 5 – Analys (`pipeline/analysis/`)
- Georefererar fynd från bildkoordinater till GPS-koordinater
- Klassificerar allvarlighetsgrad: KRITISK / HÖG / MEDIUM / LÅG
- Grupperar fynd per takyta eller fasad
- Genererar GeoJSON med alla georefererade fynd

### Steg 6 – Rapport (`pipeline/report/`)
- Renderar Jinja2-mall med alla fynd, kartor och temperaturgrafik
- Genererar PDF via WeasyPrint
- Rapport innehåller: sammanfattning, karta med fynd, bildbilagor, prioriterad åtgärdslista

**Krav på rapportens utseende och känsla:**
Rapporten är det enda kunden faktiskt ser. Den ska kännas som ett professionellt dokument
från ett seriöst techbolag – inte ett automatgenererat skript-output. Följ dessa principer:

- **Visuell design:** Skandinavisk, avskalad precision. Mörkt omslag med North Drone Analytics-identitet,
  tydlig typografi (DM Sans för rubriker, monospace för tekniska värden).
- **Framsida:** Uppdragsreferens, datum, fastighetsadress, beställare och en
  sammanfattning av fynd-statistik (antal per allvarlighetsgrad).
- **Karta:** Leaflet-baserad interaktiv karta (i HTML-versionen) eller statisk karta
  (i PDF-versionen) med georefererade fynd som färgkodade markörer per allvarlighetsgrad.
- **Termisk bildsektion:** Sidvid visning av RGB-bild och termisk bild för varje
  kritiskt/högt fynd. Temperaturskala synlig. Fynd inringade med ID-etikett.
- **Detekteringstabell:** Alla fynd med ID, typ, allvarlighetsgrad (badge), källa
  (Termisk / RGB-AI / Kombinerad), konfidensgrad (visuell progress-bar) och yta i m².
- **Prioriterad åtgärdsplan:** Rankad lista med urgency (inom X veckor/månader)
  och estimerad kostnad per åtgärd.
- **Metodiksektion:** Enkel pipeline-visualisering som visar de 6 stegen och vilka
  verktyg som användes – ger kunden förtroende för analysens grund.
- **Sidfot:** North Drone Analytics, tagline "Data från luften. Insikt på marken.",
  rapport-ID, genereringsdatum, bekräftelse att all data behandlats inom Sverige.

Färgkodning av allvarlighetsgrad ska vara konsekvent genom hela rapporten:
- KRITISK: mörkröd (#9b1d1d)
- HÖG: bränd orange (#b5451b)
- MEDEL: amber (#c87d2f)
- LÅG: grön (#2d6a4f)

## Demo-UI (`frontend/index.html`)

Demo-UI:t är det första en kund ser när de kör demon lokalt. Det ska vara visuellt
imponerande och kommunicera vad North Drone Analytics gör på under 5 sekunder.

Sidan är en enkel single-page HTML-fil (ingen build-process, ingen React) som:
1. Visar North Drone Analytics-identitet och tagline tydligt
2. Har ett enkelt formulär för att starta en pipeline-körning (välj exempeldata eller ladda upp egna bilder)
3. Visar pipeline-progress i realtid via polling mot API:et (steg 1–6 med status)
4. Presenterar en länk till den färdiga rapporten när pipeline:n är klar

Design: samma estetik som rapporten – mörkt tema, skandinavisk precision, tydlig typografi.
Använd Google Fonts (DM Sans + DM Mono).
Leaflet för kartvisningar.
Ingen extern CSS-framework – ren CSS med custom properties.

## Dataformat

### Indata (termisk bild)
DJI Mavic 3 Thermal sparar termisk data inbäddad i JPEG-metadata (R-JPEG-format).
`flirimageextractor` hanterar detta direkt. GPS-koordinater finns i EXIF.

### Utdata per steg
Varje steg sparar sitt resultat som JSON i `data/outputs/{run_id}/`:
```
data/outputs/{run_id}/
├── ingest.json
├── photogrammetry/
│   ├── orthophoto.tif
│   └── pointcloud.laz
├── thermal/
│   └── findings.json
├── detection/
│   └── findings.json
├── analysis/
│   ├── findings.geojson
│   └── summary.json
└── report/
    └── rapport_{run_id}.pdf
```

## Konfiguration

Alla parametrar via `.env` (kopiera `.env.example`):

```env
# WebODM
WEBODM_URL=http://localhost:8000
WEBODM_USERNAME=admin
WEBODM_PASSWORD=admin

# Termisk analys
THERMAL_ANOMALY_THRESHOLD_C=3.0   # Grader avvikelse för att flagga anomali
THERMAL_MIN_AREA_PX=50             # Minsta anomaliyta i pixlar

# Detektering
DETECTION_MODEL_PATH=models/crack_detection/best.pt
DETECTION_CONFIDENCE=0.4

# Rapport
REPORT_COMPANY_NAME=North Drone Analytics
REPORT_LOGO_PATH=frontend/assets/logo.png
```

## Att köra demo:n

```bash
# 1. Klona och konfigurera
git clone https://github.com/smarttek-ab/smarttek-demo
cd smarttek-demo
cp .env.example .env

# 2. Starta stacken (ODM + API + frontend)
docker compose up -d

# 3. Kör pipeline på exempeldata
python tests/sample_run.py

# 4. Rapporten finns i
open data/outputs/sample/report/rapport_sample.pdf
```

## Docker Compose-tjänster

```yaml
services:
  webodm:        # OpenDroneMap – fotogrammetri
  api:           # FastAPI – pipeline-API
  frontend:      # Nginx – servar demo-UI
```

WebODM kräver minst 8 GB RAM för att köra på riktiga dataset.
I demo-läge med förberäknat exempelortofoto klarar sig API:et utan WebODM.

## Demo-läge utan riktiga bilder

När Claude Code eller en ny utvecklare sätter upp repot för första gången
ska allt fungera utan tillgång till DJI-råbilder. Repot ska innehålla:

- `data/sample/rgb/` – minst 3 syntetiska eller publikt licensierade takbilder (JPEG)
- `data/sample/thermal/` – minst 2 syntetiska termiska bilder med injicerade anomalier
- `data/sample/precomputed/orthophoto.tif` – ett enkelt syntetiskt ortofoto
- `models/crack_detection/` – ett lättviktigt förtränat YOLO-nano-modell från Ultralytics
  publika modeller, eller en dummy-modell som returnerar mockade fynd

Pipeline-stegen ska ha ett `--demo` flag som byter mot syntetiska indata
när riktiga bilder saknas.

## Licenser och compliance

- **OpenDroneMap (AGPLv3):** Vi kör ODM som intern server, inte distribuerar det.
  Analyslogiken i `pipeline/` är separat och proprietär.
- **YOLOv8 (AGPL-3.0):** Samma princip. Vid kommersiell skalning: utvärdera
  Ultralytics Enterprise-licens (~400 USD/år).
- **flirimageextractor / flirpy (MIT):** Fri för kommersiell användning.
- **rasterio / geopandas / shapely (BSD/MIT):** Fri för kommersiell användning.
- **WeasyPrint (BSD):** Fri för kommersiell användning.
- **FastAPI (MIT):** Fri för kommersiell användning.

## Vad som inte ingår i det här repot

- Drönflygningsplanering (hanteras i DJI Pilot 2)
- Träning av YOLOv8-modellen (separat process, se `models/crack_detection/README.md`)
- Kundportal och autentisering (kommande)
- Integration med fastighetsregister (kommande)

## Nästa steg efter demo

1. Fine-tune YOLOv8 på egna fastighetsbilder från Halland
2. Bygg kundportal (Next.js, samma stack som smarttek-website)
3. Lägg till stöd för flera fastigheter i ett uppdrag (batch-körning)
4. Integration med kommunens fastighetsregister (API eller CSV-import)

## Tekniska beslut att hålla fast vid

- **All data stannar i Sverige** – kör aldrig bilddata mot externa API:er
- **Modulär pipeline** – varje steg är utbytbart utan att påverka resten
- **Docker-first** – allt ska köra i containers, inga globala systeminstallationer
- **Pydantic för alla datamodeller** – tydliga kontrakt mellan pipeline-stegen
- **Demo-läge alltid fungerande** – `python tests/sample_run.py` ska alltid gå igenom
- **Inga notebooks i produktion** – Jupyter är OK för experiment, pipeline-koden ska vara moduler

# BubbleHead Web UI

This document focuses on the browser interface served by `ui.py`. The backend is FastAPI and the frontend is a static single-page app in `frontend/index.html`.

## Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Ollama:
   ```bash
   ollama serve
   ```
3. Pull the configured models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral:latest
   ```
4. Launch the app:
   ```bash
   python ui.py
   ```

Open `http://localhost:7860`.

## What The UI Does

- Uploads one document at a time to `POST /api/ingest`
- Sends research questions to `POST /api/query`
- Polls `GET /api/status` until the pipeline has finished loading
- Polls `GET /api/collection` to show chunk totals and collection state

## User Flow

1. Wait for the pipeline to finish warming up.
2. Attach a supported file: PDF, DOCX, PPTX, TXT, HTML, or CSV.
3. Confirm ingestion succeeded and chunks were stored.
4. Ask questions against the populated collection.

## Notes

- Ingested chunk data is persisted under `data/chroma/`.
- The frontend stores session history and document names in browser `localStorage`.
- If the page loads before the backend is ready, the UI will retry automatically.

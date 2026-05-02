@echo off
REM BubbleHead Research Analysis Interface — Windows Startup Script

echo.
echo  BubbleHead Research Analysis Interface
echo  =======================================
echo.

REM ── Check Ollama ──────────────────────────────────────────────────────────
echo  Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo  [WARN]  Ollama is not running.
    echo          Start it in a separate terminal: ollama serve
    echo.
) else (
    echo  [OK]    Ollama is running.
)

REM ── Check embedding model ─────────────────────────────────────────────────
echo  Checking required models...
ollama list 2>nul | findstr "nomic-embed-text" >nul
if errorlevel 1 (
    echo  [WARN]  nomic-embed-text not found. Run: ollama pull nomic-embed-text
) else (
    echo  [OK]    nomic-embed-text found.
)

REM ── Check LLM ─────────────────────────────────────────────────────────────
ollama list 2>nul | findstr "mistral:latest" >nul
if errorlevel 1 (
    echo  [WARN]  mistral:latest not found. Run: ollama pull mistral:latest
) else (
    echo  [OK]    mistral:latest found.
)

echo.
echo  Starting server on http://localhost:7860
echo  Press Ctrl+C to stop.
echo.

REM ── Launch FastAPI backend ────────────────────────────────────────────────
python ui.py

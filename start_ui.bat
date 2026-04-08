@echo off
REM BubbleHead RAG UI Startup Script for Windows

echo 🫧 Starting BubbleHead RAG Testing UI...
echo.

REM Check if Ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Warning: Ollama doesn't appear to be running
    echo    Please start Ollama in another terminal: ollama serve
    echo.
)

echo 📦 Checking required models...
ollama list | findstr "nomic-embed-text" >nul
if errorlevel 1 (
    echo    ⚠️  nomic-embed-text not found
    echo    Run: ollama pull nomic-embed-text
) else (
    echo    ✅ nomic-embed-text found
)

ollama list | findstr "mistral:7b" >nul
if errorlevel 1 (
    echo    ⚠️  mistral:7b not found
    echo    Run: ollama pull mistral:7b
) else (
    echo    ✅ mistral:7b found
)

echo.
echo 🚀 Launching UI on http://localhost:7860
echo.

REM Launch the UI
python ui.py

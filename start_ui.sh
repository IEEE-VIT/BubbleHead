#!/bin/bash

# BubbleHead Research Analysis Interface startup script

echo
echo "BubbleHead Research Analysis Interface"
echo "======================================"
echo

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[WARN] Ollama is not running."
    echo "       Start it in another terminal: ollama serve"
    echo
else
    echo "[OK]   Ollama is running."
fi

echo "Checking required models..."
if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    echo "[OK]   nomic-embed-text found."
else
    echo "[WARN] nomic-embed-text not found. Run: ollama pull nomic-embed-text"
fi

if ollama list 2>/dev/null | grep -q "mistral:latest"; then
    echo "[OK]   mistral:latest found."
else
    echo "[WARN] mistral:latest not found. Run: ollama pull mistral:latest"
fi

echo
echo "Starting server on http://localhost:7860"
echo "Press Ctrl+C to stop."
echo

python ui.py

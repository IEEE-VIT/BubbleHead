#!/bin/bash

# BubbleHead RAG UI Startup Script

echo "🫧 Starting BubbleHead RAG Testing UI..."
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama doesn't appear to be running"
    echo "   Please start Ollama in another terminal: ollama serve"
    echo ""
fi

# Check if models are available
echo "📦 Checking required models..."
if ollama list | grep -q "nomic-embed-text"; then
    echo "   ✅ nomic-embed-text found"
else
    echo "   ⚠️  nomic-embed-text not found"
    echo "   Run: ollama pull nomic-embed-text"
fi

if ollama list | grep -q "mistral:7b"; then
    echo "   ✅ mistral:7b found"
else
    echo "   ⚠️  mistral:7b not found"
    echo "   Run: ollama pull mistral:7b"
fi

echo ""
echo "🚀 Launching UI on http://localhost:7860"
echo ""

# Launch the UI
python ui.py

#!/bin/bash
set -e

echo "================================================================"
echo "        MongoDB Assistance Docker Container Starting"
echo "================================================================"
echo ""
echo "API URL: ${API_URL}"
echo "Python Version: $(python --version)"
echo ""
echo "================================================================"
echo ""

# Start Ollama service in background
echo "🚀 Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "✅ Ollama service is ready!"
        break
    fi
    sleep 1
done

# Run the complete setup
echo ""
echo "🔧 Running complete setup..."
python /app/complete_setup.py

# If setup successful, launch the app
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "        Launching MongoDB Assistance Application"
    echo "================================================================"
    echo ""
    exec streamlit run /app/app_dynamic.py --server.port=8501 --server.address=0.0.0.0
else
    echo ""
    echo "❌ Setup failed. Please check the logs above."
    exit 1
fi

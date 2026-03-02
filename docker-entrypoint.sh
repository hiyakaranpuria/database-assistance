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

# Ask user if they want to install Ollama
echo "🤖 Do you want to install Ollama for AI-powered queries?"
echo "   (Ollama enables natural language MongoDB queries)"
echo ""
read -p "Install Ollama? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install Ollama if not already installed
    if ! command -v ollama &> /dev/null; then
        echo "📥 Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "✅ Ollama installed!"
    else
        echo "✅ Ollama already installed"
    fi

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

    # Run the complete setup if it exists
    if [ -f "/app/complete_setup.py" ]; then
        echo ""
        echo "🔧 Running complete setup..."
        python /app/complete_setup.py
    fi
else
    echo "⏭️  Skipping Ollama installation"
    echo "   You can still use the app without AI features"
    echo ""
fi

# Launch the app
echo ""
echo "================================================================"
echo "        Launching MongoDB Assistance Application"
echo "================================================================"
echo ""

# Try to run app_dynamic.py, fallback to MONGODB_AI_CHAT.py
if [ -f "/app/app_dynamic.py" ]; then
    exec streamlit run /app/app_dynamic.py --server.port=8501 --server.address=0.0.0.0
elif [ -f "/app/MONGODB_AI_CHAT.py" ]; then
    exec streamlit run /app/MONGODB_AI_CHAT.py --server.port=8501 --server.address=0.0.0.0
else
    echo "❌ No Streamlit app found!"
    exit 1
fi

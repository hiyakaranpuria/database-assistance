FROM python:3.11-slim

# Install system dependencies (curl for Ollama, procps for ps, git if needed)
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
# We copy this first to cache pip installs
COPY requirements.txt .

# CRITICAL: Install CPU-only PyTorch first to prevent ~4GB of NVIDIA CUDA drivers from downloading
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir requests python-dotenv pymongo

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501
# Expose Ollama port for local health checks/queries if needed
EXPOSE 11434

# Disable Streamlit telemetry and file watcher warnings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Interactive entrypoint script
ENTRYPOINT ["python", "entrypoint.py"]

# ================================================================
# MongoDB Assistance Docker Image
# ================================================================
# AI-Powered MongoDB Assistant with natural language queries
# Includes: Identity verification, OTP, Ollama AI, Streamlit UI
# ================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="karanpuriahiya@gmail.com"
LABEL description="AI-Powered MongoDB Assistant with natural language queries"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    API_URL=https://mongodb-assistant-backend.onrender.com

# Install system dependencies (including Ollama requirements)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Note: Ollama will be installed at runtime via entrypoint script

# Create app directory
WORKDIR /app

# Copy only required files for Docker
COPY requirements-docker.txt ./requirements.txt

# Copy all Python files (this will copy everything that exists)
COPY *.py .

# Copy Modelfile if it exists
COPY Modelfile* . 2>/dev/null || true

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/logs /app/.streamlit

# Create Streamlit config
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#10b981"\n\
backgroundColor = "#0f172a"\n\
secondaryBackgroundColor = "#1e293b"\n\
textColor = "#f1f5f9"\n' > /app/.streamlit/config.toml

# Copy and make entrypoint executable
COPY docker-entrypoint.sh .
RUN chmod +x /app/docker-entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

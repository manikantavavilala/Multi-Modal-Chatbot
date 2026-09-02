# =====================================================
# Dockerfile — Campus Orientation Assistant
# =====================================================
# Build:   docker build -t campus-assistant .
# Run:     docker run -p 8501:8501 campus-assistant
# Access:  http://localhost:8501
# =====================================================

FROM python:3.10-slim

# System dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY knowledge_base/ ./knowledge_base/
COPY dataset/ ./dataset/
COPY saved_models/ ./saved_models/
COPY streamlit_app/ ./streamlit_app/
COPY src/ ./src/
COPY outputs/ ./outputs/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]

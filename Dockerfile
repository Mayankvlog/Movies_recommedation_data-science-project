# Dockerfile for Streamlit Movie Recommendation app
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_CLIENT_LOGGER_LEVEL=info

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Streamlit configuration
RUN echo "[client]\nlogLevel = 'info'\n[logger]\nlevel = 'info'" > ~/.streamlit/config.toml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Default command to run the Streamlit app (apps.py)
CMD ["streamlit", "run", "apps.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=info"]

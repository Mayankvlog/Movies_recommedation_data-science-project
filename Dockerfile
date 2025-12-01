# Dockerfile for Streamlit Movie Recommendation app
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (if you need more, add them here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit specific configuration
EXPOSE 8501

# Default command to run the Streamlit app (apps.py)
CMD ["streamlit", "run", "apps.py", "--server.port=8501", "--server.address=0.0.0.0"]

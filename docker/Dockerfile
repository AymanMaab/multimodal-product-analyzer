# Multi-stage Dockerfile for Multimodal Product Analyzer

# Stage 1: Base image with Python and dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Stage 2: API Service
FROM base as api

COPY . /app

# Expose API port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Streamlit Dashboard
FROM base as dashboard

COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Training Environment
FROM base as training

# Install additional training dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel

COPY . /app

# Default command
CMD ["/bin/bash"]
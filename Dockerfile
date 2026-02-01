# E-Commerce Bundle Recommendation Service - Docker Image
# Python 3.11 slim base for smaller image size (~600-800MB total)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for data processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

# Copy pre-downloaded Kaggle dataset into image
COPY data/data.csv ./data/data.csv

# Create directories for models (will be mounted as volume)
RUN mkdir -p /app/models

# Copy startup script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 5000

# Health check using existing /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Docker Deployment Guide

Containerized deployment of the E-Commerce Bundle Recommendation API using Docker and Docker Compose.

## Quick Start

```bash
# 1. Build the Docker image
docker-compose build

# 2. Start the API server
docker-compose up

# 3. Access the Web UI
open http://localhost:5000
```

The API will be available at `http://localhost:5000` with all endpoints accessible.

## Prerequisites

### Required Before Running

1. **Pre-trained Model**: You must have a trained model file
   ```bash
   # Train the model outside Docker (one-time setup)
   python main.py --train
   ```
   This creates `models/recommendation_engine.pkl` which will be mounted into the container.

2. **Docker & Docker Compose**: Install from [docker.com](https://www.docker.com/get-started)

3. **Dataset**: The raw `data/data.csv` file (included in the Docker image)

### Directory Structure

Ensure your project has this structure before building:
```
ecom-recommend-service/
├── Dockerfile
├── docker-compose.yml
├── docker-entrypoint.sh
├── data/
│   └── data.csv              # Required: Raw Kaggle dataset
├── models/
│   └── recommendation_engine.pkl  # Required: Pre-trained model
└── src/                      # Application code
```

## Docker Architecture

### Included in Image
- Python 3.11 slim base
- All Python dependencies from `requirements.txt`
- Application code (`src/`, `config/`, `main.py`)
- Raw dataset (`data/data.csv`) - 25MB
- Non-root user (`appuser`) for security

### Mounted as Volumes
- `models/` - Pre-trained model files (read-only)
- `data/` - Data directory for cache persistence (optional)

### Not Included
- Virtual environments (handled by container isolation)
- Tests, notebooks, documentation
- Git files, IDE configurations
- Processed data files (can be generated at runtime)

## Building the Image

### Standard Build

```bash
docker-compose build
```

### Manual Build (without docker-compose)

```bash
docker build -t ecom-recommend-api:latest .
```

**Expected build time:** 2-3 minutes  
**Expected image size:** 600-800 MB

## Running the Container

### Using Docker Compose (Recommended)

```bash
# Start in foreground (see logs)
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down
```

### Using Docker Run

```bash
docker run -d \
  --name ecom-recommend-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  -e PORT=5000 \
  ecom-recommend-api:latest
```

## Environment Variables

Configure the API behavior using environment variables in `docker-compose.yml`:

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | API server port |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode |

### Path Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PATH` | `/app/data` | Path to data directory |
| `MODELS_PATH` | `/app/models` | Path to model files |

### ML Configuration (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_SUPPORT` | `0.02` | Minimum support for bundles |
| `MIN_CONFIDENCE` | `0.5` | Minimum confidence threshold |
| `MAX_BUNDLE_SIZE` | `5` | Maximum items per bundle |

### Example: Custom Port

```yaml
# docker-compose.yml
services:
  api:
    ports:
      - "8080:8080"  # Map to different host port
    environment:
      - PORT=8080
```

## Volume Mounts

### Models Volume (Required)

The pre-trained model **must** be mounted as a read-only volume:

```yaml
volumes:
  - ./models:/app/models:ro
```

**What happens if missing:** Container startup will fail with clear error message.

### Data Volume (Optional)

Mount the data directory to persist caches and processed files:

```yaml
volumes:
  - ./data:/app/data
```

**Benefits:**
- Persists LLM caches (`*_cache.json`)
- Persists processed data files
- Faster restarts (no reprocessing needed)

## Health Checks

The container includes a built-in health check using the `/health` endpoint:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

Check container health status:
```bash
docker-compose ps
# or
docker inspect --format='{{.State.Health.Status}}' ecom-recommend-api
```

## Accessing the API

### Web UI
Open your browser:
```
http://localhost:5000
```

### REST API Endpoints

```bash
# Health check
curl http://localhost:5000/health

# List recommenders
curl http://localhost:5000/api/v1/recommenders

# Get bundle recommendations
curl "http://localhost:5000/api/v1/bundles?product_description=candle&threshold=0.3&top_n=5"

# Get cross-sell suggestions
curl "http://localhost:5000/api/v1/cross-sell?product_description=candle&top_n=5"

# Get statistics
curl http://localhost:5000/api/v1/stats
```

## Troubleshooting

### Model Not Found Error

**Error:** "Pre-trained model not found at /app/models/recommendation_engine.pkl"

**Solution:**
1. Train the model outside Docker:
   ```bash
   python main.py --train
   ```
2. Verify the model file exists:
   ```bash
   ls -lh models/recommendation_engine.pkl
   ```
3. Ensure volume mount is correct in `docker-compose.yml`

### Port Already in Use

**Error:** "Bind for 0.0.0.0:5000 failed: port is already allocated"

**Solution:** Change the host port mapping:
```yaml
ports:
  - "8080:5000"  # Use port 8080 instead
```

### Container Exits Immediately

Check logs for errors:
```bash
docker-compose logs api
```

Common causes:
- Missing model file
- Port conflict
- Invalid configuration

### Permission Denied Errors

The container runs as non-root user `appuser`. Ensure mounted volumes have correct permissions:

```bash
# Fix permissions on host
chmod -R 755 models/
chmod -R 755 data/
```

### Build Fails - Dataset Missing

**Error:** "COPY failed: file not found: data/data.csv"

**Solution:** Ensure `data/data.csv` exists before building:
```bash
# Download dataset
python main.py --download

# Verify file exists
ls -lh data/data.csv
```

## Production Deployment

### Recommendations for Production

1. **Use specific image tags** (not `latest`):
   ```yaml
   image: ecom-recommend-api:1.0.0
   ```

2. **Use production WSGI server** instead of Flask dev server:
   ```dockerfile
   # Add to Dockerfile
   RUN pip install gunicorn
   
   # Update entrypoint
   CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "src.api:app"]
   ```

3. **Add nginx reverse proxy** for static files:
   ```yaml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
   ```

4. **Use secrets for sensitive data**:
   ```yaml
   secrets:
     - kaggle_key
   environment:
     - KAGGLE_KEY_FILE=/run/secrets/kaggle_key
   ```

5. **Enable container restart policy**:
   ```yaml
   restart: unless-stopped
   ```

6. **Set resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
       reservations:
         cpus: '1'
         memory: 1G
   ```

### Kubernetes Deployment

For Kubernetes, use this manifest as a starting point:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ecom-recommend-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ecom-recommend-api
  template:
    metadata:
      labels:
        app: ecom-recommend-api
    spec:
      containers:
      - name: api
        image: ecom-recommend-api:1.0.0
        ports:
        - containerPort: 5000
        env:
        - name: PORT
          value: "5000"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 40
          periodSeconds: 30
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: ecom-models-pvc
```

## Maintenance

### Updating the Model

1. Train new model on host:
   ```bash
   python main.py --train
   ```

2. Restart container to load new model:
   ```bash
   docker-compose restart
   ```

### Viewing Logs

```bash
# Follow logs in real-time
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api

# Since specific time
docker-compose logs --since=30m api
```

### Cleaning Up

```bash
# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi ecom-recommend-api:latest

# Clean up Docker system
docker system prune -a
```

## Advanced Configuration

### Development with Live Code Reload

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./src:/app/src  # Mount source code
      - ./models:/app/models:ro
    environment:
      - FLASK_DEBUG=true
      - PORT=5000
    command: python main.py --api --verbose
```

Run with:
```bash
docker-compose -f docker-compose.dev.yml up
```

### Multi-stage Build (Smaller Image)

Optimize Dockerfile for smaller production images:

```dockerfile
# Stage 1: Builder
FROM python:3.11 as builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "main.py", "--api"]
```

## Support

For Docker-specific issues:
1. Check logs: `docker-compose logs -f`
2. Verify mounts: `docker inspect ecom-recommend-api`
3. Test health: `curl http://localhost:5000/health`
4. Check container status: `docker-compose ps`

For application issues, see main [README.md](README.md) documentation.

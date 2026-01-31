#!/bin/bash
set -e

echo "=========================================="
echo "E-Commerce Recommendation API - Starting"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validate required model file
MODEL_FILE="/app/models/recommendation_engine.pkl"
echo -n "Checking for pre-trained model... "

if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}FAILED${NC}"
    echo ""
    echo -e "${RED}ERROR: Pre-trained model not found!${NC}"
    echo ""
    echo "The API requires a pre-trained model at:"
    echo "  $MODEL_FILE"
    echo ""
    echo "Please ensure you have:"
    echo "  1. Trained the model outside the container:"
    echo "     python main.py --train"
    echo ""
    echo "  2. Mounted the models directory as a volume:"
    echo "     docker-compose up"
    echo "     OR"
    echo "     docker run -v ./models:/app/models ecom-recommend-api"
    echo ""
    exit 1
fi
echo -e "${GREEN}OK${NC}"

# Validate data.csv exists
DATA_FILE="/app/data/data.csv"
echo -n "Checking for dataset... "

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}WARNING${NC}"
    echo "Dataset not found at $DATA_FILE"
    echo "API may not function correctly without the dataset."
else
    echo -e "${GREEN}OK${NC}"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Port: ${PORT:-5000}"
echo "  Data Path: ${DATA_PATH:-/app/data}"
echo "  Models Path: ${MODELS_PATH:-/app/models}"
echo "  Debug Mode: ${FLASK_DEBUG:-false}"
echo ""

# Display model info
echo "Model Information:"
MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
echo "  Size: $MODEL_SIZE"
echo "  Path: $MODEL_FILE"
echo ""

echo "=========================================="
echo "Starting Flask API Server..."
echo "=========================================="
echo ""

# Execute the main command
exec python main.py --api

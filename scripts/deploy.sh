#!/bin/bash
# Deployment script for IBKR Trading Bot
# Run on Contabo VPS

set -e

echo "=========================================="
echo "IBKR Trading Bot Deployment"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Copy .env.example to .env and configure it first."
    exit 1
fi

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not installed!"
    echo "Install Docker first: curl -fsSL https://get.docker.com | sh"
    exit 1
fi

# Check Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose not installed!"
    exit 1
fi

# Create required directories
mkdir -p data logs

# Pull latest images
echo ""
echo "Pulling latest images..."
docker compose pull

# Build the trading bot
echo ""
echo "Building trading bot..."
docker compose build

# Start services
echo ""
echo "Starting services..."
docker compose up -d

# Show status
echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
docker compose ps
echo ""
echo "View logs with: docker compose logs -f"
echo "Stop with: docker compose down"

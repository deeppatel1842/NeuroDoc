#!/bin/bash

# NeuroDoc Docker Deployment Script

echo "Starting NeuroDoc Deployment..."

# Build and start the container
echo "Building Docker image..."
docker-compose build

echo "Starting NeuroDoc container..."
docker-compose up -d

echo "Waiting for service to be ready..."
sleep 10

# Check if service is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "NeuroDoc is running successfully!"
    echo "Local URL: http://localhost:8000"
    echo "Chat Interface: http://localhost:8000/static/index.html"
    echo ""
    echo "To check logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "Service failed to start. Check logs with: docker-compose logs"
fi

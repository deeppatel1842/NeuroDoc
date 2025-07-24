@echo off

REM NeuroDoc Docker Deployment Script for Windows

echo 🧠 Starting NeuroDoc Deployment...

REM Build and start the container
echo 📦 Building Docker image...
docker-compose build

echo 🚀 Starting NeuroDoc container...
docker-compose up -d

echo ⏳ Waiting for service to be ready...
timeout /t 10 /nobreak > nul

REM Check if service is running
curl -f http://localhost:8000/health > nul 2>&1
if %errorlevel% == 0 (
    echo ✅ NeuroDoc is running successfully!
    echo 🌐 Local URL: http://localhost:8000
    echo 📱 Chat Interface: http://localhost:8000/static/index.html
    echo.
    echo 📊 To check logs: docker-compose logs -f
    echo 🛑 To stop: docker-compose down
) else (
    echo ❌ Service failed to start. Check logs with: docker-compose logs
)

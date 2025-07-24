@echo off
echo Starting NeuroDoc (Non-Docker Mode)...

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting NeuroDoc server...
echo Local URL: http://localhost:8000
echo Chat Interface: http://localhost:8000/static/index.html
echo.
echo Press Ctrl+C to stop the server
echo.

python -m src.api.main

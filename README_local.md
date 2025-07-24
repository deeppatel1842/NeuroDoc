# NeuroDoc - AI-Powered Document Analysis

A sophisticated RAG (Retrieval-Augmented Generation) system for intelligent document analysis and Q&A.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available

### Local Deployment

1. **Clone and navigate to the project:**
   ```bash
   cd NeuroDoc
   ```

2. **Run the deployment script:**
   
   **Windows:**
   ```cmd
   deploy.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Access the application:**
   - Chat Interface: http://localhost:8000/static/index.html
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Manual Docker Commands

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## 🌐 Getting a Public URL

### Option 1: Using ngrok (Recommended for testing)

1. **Install ngrok:** https://ngrok.com/download
2. **Start your Docker container locally**
3. **Expose port 8000:**
   ```bash
   ngrok http 8000
   ```
4. **Use the provided HTTPS URL** (e.g., `https://abc123.ngrok.io`)

### Option 2: Deploy to Cloud Platforms

#### Railway
1. Connect your GitHub repository to Railway
2. Deploy with these environment variables:
   ```
   PORT=8000
   HOST=0.0.0.0
   ```

#### Render
1. Connect repository to Render
2. Use Docker deployment
3. Set port to 8000

#### DigitalOcean App Platform
1. Create new app from Docker Hub or GitHub
2. Configure port 8000
3. Deploy

#### Google Cloud Run
1. Build and push to Container Registry:
   ```bash
   docker build -t gcr.io/YOUR-PROJECT/neurodoc .
   docker push gcr.io/YOUR-PROJECT/neurodoc
   ```
2. Deploy to Cloud Run with port 8000

## 📱 Features

- **Dual-Mode Analysis:** Pre-loaded AI research papers + user document uploads
- **Intelligent Responses:** Enhanced LLM integration with comprehensive analysis
- **Clean Interface:** Modern chat UI with document management
- **Session Management:** Persistent conversations and document tracking
- **Real-time Processing:** Fast document indexing and retrieval

## 🔧 Configuration

The application works out of the box with local LLM models. For production:

1. **Environment Variables:**
   ```bash
   HOST=0.0.0.0
   PORT=8000
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

2. **Volume Mounts:**
   - `./data:/app/data` - Knowledge base and sessions
   - `./static:/app/static` - Web interface

## 🏥 Health Monitoring

- **Basic Health:** `GET /health`
- **Detailed Health:** `GET /health/advanced`

## 📊 API Endpoints

- `POST /session` - Create new session
- `POST /upload` - Upload documents
- `POST /query` - Ask questions
- `GET /static/*` - Web interface

## 🛠️ Development

For local development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python src/api/main.py
```

## 📝 License

This project is configured for both local use and cloud deployment.

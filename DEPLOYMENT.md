# NeuroDoc Docker & Cloud Deployment Guide

## 🔧 Prerequisites Setup

### 1. Docker Desktop (for local development)
- **Windows/Mac:** Download Docker Desktop from https://docker.com/products/docker-desktop
- **Start Docker Desktop** before running deployment scripts

### 2. Cloud Deployment Options

## 🌐 Get Public URL - Multiple Options

### Option 1: ngrok (Easiest for testing)

1. **Install ngrok:** https://ngrok.com/download
2. **Start locally:**
   ```bash
   # Start your app locally first
   python src/api/main.py
   
   # In another terminal
   ngrok http 8000
   ```
3. **Get public URL:** Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

### Option 2: Railway (Recommended)

1. **Visit:** https://railway.app
2. **Connect GitHub:** Link your repository
3. **Deploy:** Railway auto-detects Docker
4. **Environment Variables:**
   ```
   PORT=8000
   HOST=0.0.0.0
   PYTHONPATH=/app
   ```
5. **Get URL:** Railway provides public domain

### Option 3: Render

1. **Visit:** https://render.com
2. **Create Web Service** from Docker
3. **Connect repository**
4. **Configure:**
   - **Runtime:** Docker
   - **Port:** 8000
   - **Health Check:** `/health`

### Option 4: Google Cloud Run

1. **Install gcloud CLI**
2. **Deploy:**
   ```bash
   # Build and deploy
   gcloud run deploy neurodoc \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8000
   ```

## 🐳 Local Docker Setup

### If Docker Desktop is running:

```bash
# Build the image
docker-compose build

# Start the service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### If you prefer running without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python src/api/main.py
```

Then access at: http://localhost:8000/static/index.html

## 🚀 Quick Cloud Deploy Commands

### Railway CLI
```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

### Render (Git Deploy)
1. Push to GitHub
2. Connect repository on Render
3. Select "Docker" runtime
4. Deploy automatically

## 📱 Access Your App

Once deployed, your NeuroDoc will be available at:
- **Chat Interface:** `https://your-domain.com/static/index.html`
- **API Docs:** `https://your-domain.com/docs`
- **Health Check:** `https://your-domain.com/health`

## 🔧 Troubleshooting

### Docker Issues:
- Ensure Docker Desktop is running
- Check: `docker --version`
- Restart Docker Desktop if needed

### Cloud Deploy Issues:
- Check build logs in platform dashboard
- Verify port 8000 is configured
- Ensure health check endpoint works

### Memory Issues:
- Increase container memory to 2GB+
- Use cloud platforms with adequate resources

## 🎯 Recommended for Beginners:

1. **For Testing:** Use ngrok with local Python server
2. **For Production:** Use Railway (easiest) or Render

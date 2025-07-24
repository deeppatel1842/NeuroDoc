# NeuroDoc

An intelligent document analysis system powered by AI that provides comprehensive answers from research documents with accurate citations and conversation memory.

## Features

- **Dual-Mode Retrieval System**: Pre-loaded AI research papers + user document uploads
- **Intelligent Document Processing**: PDF text extraction with OCR fallback
- **Advanced LLM Integration**: Local Ollama integration for comprehensive responses
- **Session Management**: Conversation history and document tracking
- **Vector Search**: FAISS-based similarity search with dense embeddings
- **Modern Chat Interface**: Responsive HTML interface with file upload
- **Docker Ready**: Complete containerization for easy deployment

## Quick Start

### Prerequisites

- Python 3.8+
- Ollama (for LLM inference)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deeppatel1842/NeuroDoc.git
   cd NeuroDoc
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama2
   ```

5. **Build knowledge base** (optional - for pre-loaded documents)
   ```bash
   python build_knowledge_base.py
   ```

6. **Run the application**
   ```bash
   python src/api/main.py
   ```

7. **Access the interface**
   Open http://localhost:8000 in your browser

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Manual Docker Build

```bash
docker build -t neurodoc .
docker run -p 8000:8000 neurodoc
```

## Project Structure

```
NeuroDoc/
├── src/
│   ├── api/              # FastAPI backend endpoints
│   ├── document_processing/  # PDF processing and text extraction
│   ├── embeddings/       # Vector embeddings generation
│   ├── llm/             # LLM integration and response generation
│   ├── memory/          # Session and conversation management
│   ├── retrieval/       # Hybrid retrieval system
│   └── utils/           # Utility functions and optimization
├── static/              # Frontend HTML interface
├── data/               # Document storage and processing
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
└── docker-compose.yml # Multi-service deployment
```

## API Endpoints

- `POST /upload` - Upload and process documents
- `POST /query` - Submit questions and get AI responses
- `GET /sessions/{session_id}/history` - Retrieve conversation history
- `GET /health` - Health check endpoint

## Configuration

Key settings in `src/config.py`:

- **LLM Model**: Default Ollama model selection
- **Embedding Model**: Sentence transformer model
- **Vector Store**: FAISS index configuration
- **Session Settings**: Timeout and memory limits

## Features in Detail

### Dual-Mode Operation

- **DEFAULT MODE**: Uses pre-loaded AI research papers for general questions
- **USER MODE**: Automatically switches to analyze only uploaded documents when available

### Document Processing

- Multi-format PDF support with text extraction
- OCR processing for scanned documents
- Intelligent text chunking and preprocessing
- Metadata extraction and storage

### LLM Integration

- Local Ollama integration for privacy
- Enhanced prompting for comprehensive responses
- Context-aware answer generation
- Citation management and source tracking

### Performance Optimization

- Sub-200ms retrieval latency
- Efficient vector similarity search
- Background processing for large documents
- Caching for frequently accessed content

## Deployment Options

### Local Development
```bash
python src/api/main.py
```

### Production Deployment

1. **Railway/Render**: Direct GitHub deployment
2. **AWS/GCP/Azure**: Container deployment
3. **Docker**: Containerized deployment
4. **Heroku**: Platform-as-a-Service deployment

See `DEPLOYMENT.md` for detailed deployment instructions.

## Usage Examples

### Basic Question
```
Q: "What are the latest advances in transformer architectures?"
A: Based on the research papers, recent advances include...
```

### Document Upload
1. Upload PDF documents via the web interface
2. Ask questions specific to your documents
3. Get answers with accurate citations

### API Usage
```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)

# Query document
query = {"question": "What are the main findings?", "session_id": "session123"}
response = requests.post('http://localhost:8000/query', json=query)
```

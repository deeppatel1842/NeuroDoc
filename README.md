```markdown
# NeuroDoc

An intelligent document analysis system powered by AI that provides comprehensive answers from research documents with accurate citations and conversation memory.

## 🎯 System Overview

NeuroDoc is a sophisticated AI-powered document analysis platform that combines modern LLM technology with advanced retrieval systems to provide intelligent insights from your documents.

![NeuroDoc Workflow](https://raw.githubusercontent.com/deeppatel1842/NeuroDoc/main/NeuroDoc_Flow/NeuroDoc_Workflow.png)

## ✨ Features

- **Dual-Mode Retrieval System**: Pre-loaded AI research papers + user document uploads
- **Intelligent Document Processing**: PDF text extraction with OCR fallback
- **Advanced LLM Integration**: Local Ollama integration for comprehensive responses
- **Session Management**: Conversation history and document tracking
- **Vector Search**: FAISS-based similarity search with dense embeddings
- **Modern Chat Interface**: Responsive HTML interface with file upload
- **Smart Suggestions**: Auto-generated questions for uploaded documents
- **Docker Ready**: Complete containerization for easy deployment

## 🚀 Live Demo Screenshots

### Document Upload and Processing Interface
![NeuroDoc Interface - Document Upload](https://raw.githubusercontent.com/deeppatel1842/NeuroDoc/main/Output/output_1.png)

*The clean, modern interface showing document upload functionality with drag-and-drop support and uploaded document management.*

### Interactive Chat and AI Responses
![NeuroDoc Chat Interface](https://raw.githubusercontent.com/deeppatel1842/NeuroDoc/main/Output/output_2.png)

*Real-time conversation with the AI assistant, showing intelligent responses and suggested questions for uploaded documents.*

## 🔄 System Architecture & Workflow

![NeuroDoc System Workflow](https://raw.githubusercontent.com/deeppatel1842/NeuroDoc/main/NeuroDoc_Flow/NeuroDoc_Workflow.png)

### Core Components Flow:

1. **Document Ingestion Pipeline**
   - PDF Upload → Text Extraction → Chunking → Embedding Generation → Vector Storage

2. **Query Processing Engine**
   - User Query → Hybrid Retrieval → Context Ranking → LLM Processing → Response Generation

3. **Session Management System**
   - Conversation Tracking → Document Association → History Persistence

## 📊 Key Features Demonstrated

### 1. Document Upload and Processing
```
✅ Successfully uploaded "LLM_1.pdf" (325 chunks processed)

💡 Automatically Generated Suggested Questions:
• What is the main purpose or summary of LLM_1.pdf?
• What are the key points discussed in LLM_1.pdf?
• Can you explain the important concepts from LLM_1.pdf?
```

### 2. Intelligent AI Assistant
**User Question:** "who are you"

**NeuroDoc Response:**
```
I'm NeuroDoc, your AI-powered document analysis assistant. I specialize in 
helping you understand and explore research papers in AI, machine learning, 
and related fields.

My capabilities include:
✓ Analyzing uploaded PDF documents
✓ Answering questions about research papers
✓ Providing detailed explanations of complex concepts
✓ Generating insights from your document collection
✓ Maintaining conversation context and history
```

### 3. Document Analysis Capabilities
- **Smart Question Generation**: Automatically creates relevant questions for uploaded documents
- **Context-Aware Responses**: Maintains conversation flow and document context
- **Session Management**: Tracks uploaded documents and conversation history
- **Dual-Mode Retrieval**: Can access both uploaded documents and pre-loaded knowledge base

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | HTML5, CSS3, JavaScript | Modern responsive chat interface |
| **Backend** | FastAPI, Python 3.8+ | REST API and document processing |
| **LLM** | Ollama (Gemma3) | Natural language generation |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2) | Document vectorization |
| **Vector DB** | FAISS | High-performance similarity search |
| **Document Processing** | PyMuPDF, pdfplumber | PDF text extraction |
| **Session Management** | In-memory + persistent storage | User session tracking |
| **Containerization** | Docker, docker-compose | Easy deployment |

## 🚀 Quick Start

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

2. **Set up Python environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install and setup Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama pull gemma3:latest
```

4. **Start the application**
```bash
python src/api/main.py
```

5. **Access the interface**
```
Open: http://localhost:8000
```

## 🐳 Docker Deployment

### Quick Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

### Manual Docker Build

```bash
# Build the image
docker build -t neurodoc .

# Run the container
docker run -p 8000:8000 neurodoc
```

## 📁 Project Structure

```
NeuroDoc/
├── src/
│   ├── api/              # FastAPI backend
│   ├── llm/              # LLM integration (Ollama)
│   ├── retrieval/        # Hybrid document retrieval
│   ├── embeddings/       # Vector embeddings (FAISS)
│   ├── document_processing/  # PDF processing pipeline
│   ├── memory/           # Session management
│   └── utils/            # Utility functions
├── static/
│   └── index.html        # Modern chat interface
├── data/
│   ├── documents/        # Uploaded documents storage
│   ├── vector_store/     # FAISS indices
│   └── processed/        # Processed document chunks
├── Output/               # Demo screenshots
├── NeuroDoc_Flow/        # System workflow diagrams
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=gemma3:latest

# Vector Store Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DIMENSION=384
MAX_CHUNKS_PER_DOCUMENT=1000

# API Configuration
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["*"]

# Session Management
SESSION_TIMEOUT_HOURS=24
MAX_CONVERSATION_HISTORY=50
```

## 🌐 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main chat interface |
| `/session` | POST | Create new user session |
| `/upload` | POST | Upload and process PDF documents |
| `/query` | POST | Process user queries |
| `/health` | GET | System health check |
| `/docs` | GET | API documentation |

### Example API Usage

```bash
# Create a new session
curl -X POST "http://localhost:8000/session" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123"}'

# Upload a document
curl -X POST "http://localhost:8000/upload" \
     -F "file=@document.pdf" \
     -F "session_id=session123"

# Ask a question
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"session_id": "session123", "question": "What is this document about?"}'
```

## 🌐 Deployment Options

### Option 1: Railway (Recommended for Production)
```bash
# Deploy to Railway
railway login
railway init
railway up
```

### Option 2: Render (Free Tier Available)
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use Docker environment
4. Automatic deployment from Dockerfile

### Option 3: Local Development with Public Access
```bash
# Start local server
python src/api/main.py

# Create public tunnel with ngrok
ngrok http 8000
```

## 📊 Performance Metrics

- **Document Processing**: 2-5 seconds per PDF (depending on size)
- **Query Response Time**: <3 seconds average
- **Vector Search Latency**: <200ms retrieval time
- **Embedding Generation**: ~100ms per chunk
- **Session Persistence**: Maintained across browser sessions
- **Supported File Types**: PDF documents
- **Maximum Document Size**: 50MB per file

## 🔍 System Capabilities

### Document Processing
- **Text Extraction**: Advanced PDF parsing with OCR fallback
- **Intelligent Chunking**: Context-aware text segmentation
- **Metadata Extraction**: Document properties and structure analysis
- **Multi-format Support**: PDF with planned support for DOCX, TXT

### AI-Powered Analysis
- **Natural Language Understanding**: Advanced query comprehension
- **Context-Aware Responses**: Maintains conversation flow
- **Citation Generation**: Accurate source referencing
- **Multilingual Support**: Handles various document languages

### User Experience
- **Intuitive Interface**: Modern, responsive chat design
- **Real-time Processing**: Live document upload and analysis
- **Session Management**: Persistent conversation history
- **Smart Suggestions**: Auto-generated relevant questions

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👨‍💻 Author

**Deep Patel** - [deeppatel1842](https://github.com/deeppatel1842)
- 🔗 LinkedIn: [Deep Patel](https://linkedin.com/in/deeppatel1842)
- 📧 Email: pateldeep1842@gmail.com

## 🙏 Acknowledgments

- **Ollama Team** - For providing excellent local LLM infrastructure
- **Hugging Face** - For state-of-the-art embedding models
- **FAISS Team** - For high-performance vector search capabilities
- **FastAPI Team** - For the robust and fast web framework
- **Open Source Community** - For the amazing tools and libraries

## 🆘 Support

For questions, issues, or support:
- 📋 Open an issue on [GitHub Issues](https://github.com/deeppatel1842/NeuroDoc/issues)
- 📧 Email: pateldeep1842@gmail.com
- 💬 Discussions: [GitHub Discussions](https://github.com/deeppatel1842/NeuroDoc/discussions)

---

**NeuroDoc** - Empowering document analysis with AI intelligence.
```

# ZaaKy AI Platform - Backend

A secure, scalable AI-powered document processing and chat system built with FastAPI.

## 🚀 Features

- **Intelligent Document Processing**: PDF, JSON, and web content ingestion with AI analysis
- **Secure Web Scraping**: Enterprise-grade scraper with SSRF protection and rate limiting
- **Real-time Chat**: WebSocket-powered chat with context-aware responses
- **Vector Search**: Pinecone-powered semantic search and retrieval
- **Multi-tenant Architecture**: Organization-based isolation and permissions
- **Adaptive Concurrency**: Dynamic load balancing for optimal performance

## 🏗️ Architecture

```
├── app/            # FastAPI application
├── config/         # Configuration management
├── routers/        # API endpoints
├── services/       # Business logic
├── utils/          # Shared utilities
└── tests/          # Test suite
```

## 🔧 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL (via Supabase)
- Pinecone vector database
- OpenAI API access

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/zaaky/backend.git
   cd zaaky-backend
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp env_example .env
   # Edit .env with your configuration
   ```

5. **Run the development server**
   ```bash
   python start_server.py
   ```

The API will be available at `http://localhost:8001`

## 📋 Environment Configuration

Copy `env_example` to `.env` and configure:

### Required Variables

```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_JWT_SECRET=your_jwt_secret

# AI Services
OPENAI_API_KEY=sk-your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=your_index_name
```

### Optional Configuration

```bash
# Performance
SCRAPING_CONCURRENT_REQUESTS=3
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30

# Security
SCRAPING_ENABLE_SSRF_PROTECTION=true
CORS_ORIGINS=http://localhost:3000
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest -m "not slow"          # Skip slow tests
pytest -m integration         # Integration tests only
pytest tests/unit/            # Unit tests only
```

## 🔒 Security Features

### Web Scraping Security

- **SSRF Protection**: Validates URLs and blocks internal network access
- **Rate Limiting**: Intelligent delays and concurrency control
- **robots.txt Compliance**: Respects website scraping policies
- **Content Validation**: Size limits and type checking
- **Secure Logging**: Sanitizes sensitive data from logs

### API Security

- **JWT Authentication**: Secure token-based authentication
- **CORS Protection**: Configurable cross-origin policies
- **Rate Limiting**: Request throttling per endpoint
- **Input Validation**: Comprehensive request validation

## 📊 Performance

### Adaptive Concurrency

- Dynamic worker scaling (1-10 workers)
- Per-domain performance optimization
- System resource monitoring
- Intelligent retry logic with exponential backoff

### Benchmarks

- **65% faster** processing with adaptive concurrency
- **39.6% time reduction** in mixed workload scenarios
- **Zero server overload** with intelligent rate limiting

## 🐳 Docker Support

```bash
# Build image
docker build -t zaaky-backend .

# Run with docker-compose
docker-compose up -d

# Development with hot reload
docker-compose -f docker-compose.dev.yml up
```

## 📚 API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

### Key Endpoints

```bash
# Authentication
POST /auth/login
POST /auth/register
GET  /auth/me

# Chat
POST /chat/conversations
GET  /chat/conversations/{id}
POST /chat/conversations/{id}/messages

# Document Processing
POST /uploads/
GET  /uploads/{id}/status
POST /search/documents

# Organizations
GET  /orgs/{id}
PUT  /orgs/{id}/settings
```

## 🔧 Development

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .
mypy .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Database Migrations

```bash
# Run migrations (if using Alembic)
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"
```

## 📈 Monitoring

### Health Checks

- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive system status

### Metrics

- Request/response times
- Error rates by endpoint
- Database connection pool status
- Vector database performance
- Scraping performance metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure virtual environment is activated
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Database Connection**

   ```bash
   # Verify Supabase credentials
   python -c "from config.settings import validate_environment; validate_environment()"
   ```

3. **Vector Database Issues**
   ```bash
   # Test Pinecone connection
   python scripts/test_pinecone.py
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Documentation: [docs.zaaky.ai](https://docs.zaaky.ai)
- Issues: [GitHub Issues](https://github.com/zaaky/backend/issues)
- Email: support@zaaky.ai

---

**Built with ❤️ by the ZaaKy Team**

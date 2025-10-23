# Environment Variables Template

Copy this content to `.env` in the backend directory and fill in your values.

```env
# ================================
# ZaaKy AI Platform Configuration
# ================================

# Application Settings
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
APP_NAME=ZaaKy AI Platform
APP_VERSION=2.1.0
API_BASE_URL=http://localhost:8001

# Database (Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
SUPABASE_JWT_SECRET=your-jwt-secret-here
SUPABASE_PROJECT_ID=your-project-id

# AI Services
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX=your-pinecone-index-name

# AI Configuration
DEFAULT_AI_MODEL=gpt-4
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
EMBEDDING_MODEL=text-embedding-3-small

# Redis Caching (REQUIRED for new caching feature)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=20
REDIS_CONNECTION_TIMEOUT=10

# Caching Configuration
ENABLE_CACHING=true
CACHE_DEFAULT_TTL=3600
CACHE_MEMORY_SIZE_MB=100

# Security
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
ALLOWED_HOSTS=localhost,yourdomain.com
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
ENABLE_AUTH_VALIDATION=true

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CONNECTION_POOL_SIZE=20
ENABLE_COMPRESSION=true
WORKER_INTERVAL_SECONDS=30

# Web Scraping
SCRAPING_TIMEOUT=30
SCRAPING_MAX_CONTENT_SIZE=52428800
SCRAPING_MIN_DELAY=1.0
SCRAPING_MAX_DELAY=3.0
SCRAPING_MAX_RETRIES=3
SCRAPING_CONCURRENT_REQUESTS=3
SCRAPING_RESPECT_ROBOTS=true
SCRAPING_ENABLE_SSRF_PROTECTION=true
SCRAPING_MAX_PAGES=100
SCRAPING_MAX_DEPTH=5

# Optional: Feature Flags
ENABLE_ANALYTICS=true
```

## Required Variables

The following variables are **REQUIRED** for the application to function:

### Critical (Application won't start without these)

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWT_SECRET`
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX`

### Redis (Required for caching feature)

- `REDIS_URL` - Default: `redis://localhost:6379`
- `REDIS_PASSWORD` - Leave empty if no password

### Optional (Have sensible defaults)

- All other variables have defaults and are optional

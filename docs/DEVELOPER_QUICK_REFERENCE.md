# Developer Quick Reference

## 🚀 Quick Start

### Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Start server
python start_server.py

# Or with uvicorn directly
uvicorn app.main:app --reload --port 8001
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_security_validators.py -v

# With coverage
pytest --cov=app --cov-report=html

# Fast tests only (skip slow)
pytest -m "not slow"
```

## 🔒 Security Best Practices

### Always Validate User Input

```python
from app.utils.validators import (
    validate_url,
    validate_upload_id,
    validate_metadata_filter,
    ValidationError
)

# Validate URLs
try:
    safe_url = validate_url(user_url)
except ValidationError as e:
    raise HTTPException(400, detail=str(e))

# Validate IDs
safe_id = validate_upload_id(upload_id)

# Validate filters
safe_filter = validate_metadata_filter({"upload_id": upload_id})
```

### Use Singleton Clients

```python
# ❌ DON'T create new clients
from supabase import create_client
client = create_client(url, key)

# ✅ DO use singleton
from app.services.storage.supabase_client import get_supabase_client
client = get_supabase_client()
```

## 📊 Pagination

### Always Paginate List Endpoints

```python
from app.utils.pagination import validate_pagination_params, create_pagination_meta

@router.get("/items")
async def list_items(page: int = 1, page_size: int = 20):
    # Validate
    params = validate_pagination_params(page, page_size)

    # Query with pagination
    result = supabase.table("items").select("*", count="exact").range(
        params.offset, params.offset + params.page_size - 1
    ).execute()

    # Return with metadata
    return {
        "items": result.data,
        "pagination": create_pagination_meta(page, page_size, result.count)
    }
```

## 🔍 Query Optimization

### Monitor Slow Queries

```python
from app.utils.query_optimizer import monitor_query

@monitor_query("get_user_data")
async def get_user_data(user_id: str):
    return supabase.table("users").select("*").eq("id", user_id).execute()
```

### Select Only Needed Fields

```python
# ❌ DON'T fetch all fields
result = supabase.table("uploads").select("*").execute()

# ✅ DO select specific fields
result = supabase.table("uploads").select(
    "id,status,type,created_at"
).execute()
```

### Use Indexed Columns for Filtering

```python
# ✅ Filter by indexed columns first
result = supabase.table("uploads").select("*").eq(
    "org_id", org_id  # Indexed
).eq(
    "status", "completed"  # Indexed
).order(
    "created_at", desc=True  # Indexed
).execute()
```

## 🎯 Error Handling

### Use Proper Exception Handling

```python
from fastapi import HTTPException
from app.utils.validators import ValidationError

@router.post("/items")
async def create_item(data: dict):
    try:
        # Validate
        validated = validate_data(data)

        # Process
        result = process_item(validated)

        return {"success": True, "data": result}

    except ValidationError as e:
        # Client error (400)
        raise HTTPException(400, detail=str(e))

    except Exception as e:
        # Server error (500)
        logger.error(f"Failed to create item: {e}")
        raise HTTPException(500, detail="Internal server error")
```

## 📝 Code Style

### Function Documentation

```python
async def process_upload(
    upload_id: str,
    org_id: str,
    file_type: str
) -> dict:
    """
    Process an upload and extract content

    Args:
        upload_id: Unique upload identifier
        org_id: Organization identifier
        file_type: Type of file (pdf, json, url)

    Returns:
        Dictionary with processing results

    Raises:
        ValidationError: If input is invalid
        HTTPException: If processing fails
    """
    # Implementation
```

### Use Type Hints

```python
from typing import List, Dict, Optional

async def get_items(
    org_id: str,
    limit: int = 20,
    filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    # Implementation
```

## 🧪 Testing

### Unit Test Template

```python
import pytest
from app.utils.validators import validate_url, ValidationError

class TestURLValidation:
    def test_valid_url(self):
        """Valid URLs should pass"""
        url = validate_url("https://example.com")
        assert url == "https://example.com"

    def test_invalid_url(self):
        """Invalid URLs should raise error"""
        with pytest.raises(ValidationError):
            validate_url("http://localhost")
```

### Integration Test Template

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_list_uploads():
    response = client.get(
        "/api/uploads?page=1&page_size=20",
        headers={"Authorization": "Bearer test-token"}
    )
    assert response.status_code == 200
    assert "uploads" in response.json()
    assert "pagination" in response.json()
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_key
SUPABASE_JWT_SECRET=your_secret

# AI Services
OPENAI_API_KEY=sk-your_key
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index

# Connection Pooling
SUPABASE_POOL_SIZE=20
PINECONE_POOL_SIZE=10

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
```

## 📊 Monitoring

### Check System Health

```bash
# Basic health
curl http://localhost:8001/health

# Detailed health
curl http://localhost:8001/health/detailed

# Connection pools
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8001/api/monitoring/connection-pools

# System alerts
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8001/api/monitoring/alerts
```

### View Query Stats

```python
from app.utils.query_optimizer import query_monitor

stats = query_monitor.get_stats()
print(f"Average query time: {stats['avg_duration_ms']}ms")
print(f"Slow queries: {stats['slow_queries']}")
```

## 🐛 Debugging

### Enable Debug Logging

```bash
# In .env
LOG_LEVEL=DEBUG
DEBUG=true
```

### Common Issues

#### Connection Pool Exhausted

```bash
# Increase pool size
SUPABASE_POOL_SIZE=50
PINECONE_POOL_SIZE=20
```

#### Slow Queries

```python
# Add monitoring
@monitor_query("slow_operation")
async def slow_operation():
    # Check query_monitor.get_stats()
```

#### Memory Issues

```python
# Use pagination
# Limit result sets
# Clear large variables after use
del large_data
```

## 🚀 Deployment

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Connection pool sizes set
- [ ] Monitoring endpoints accessible
- [ ] Error tracking configured
- [ ] Database indexes created
- [ ] Rate limits configured

### Post-Deployment Monitoring

- [ ] Check `/health` endpoint
- [ ] Monitor connection pool utilization
- [ ] Check for slow queries
- [ ] Verify error rates
- [ ] Monitor memory usage

## 📚 Additional Resources

- [Full Documentation](./README.md)
- [Security Guide](./SECURITY_IMPROVEMENTS.md)
- [Pagination Guide](./PAGINATION_GUIDE.md)
- [Connection Pooling](./CONNECTION_POOLING.md)
- [API Documentation](http://localhost:8001/docs)

## 🆘 Getting Help

- Check logs: `tail -f logs/app.log`
- Review error monitoring dashboard
- Check system health: `GET /api/monitoring/system-health`
- Review test failures: `pytest -v`

---

**Quick Tips:**

- Always validate user input
- Use pagination for lists
- Monitor query performance
- Use singleton clients
- Write tests for new features
- Document your code
- Check security implications

**Remember:** Security, performance, and maintainability are not optional!

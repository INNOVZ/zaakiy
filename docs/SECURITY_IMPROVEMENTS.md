# Security Improvements - Input Validation

## Overview

This document outlines the comprehensive security improvements made to strengthen input validation and prevent injection attacks, SSRF vulnerabilities, and other security issues.

## 🔒 Enhanced URL Validation

### SSRF Protection

The URL validator now includes comprehensive SSRF (Server-Side Request Forgery) protection:

#### DNS Resolution Validation

- Resolves hostnames to IP addresses before allowing requests
- Blocks private IP ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
- Blocks loopback addresses (127.0.0.1, ::1)
- Blocks link-local addresses (169.254.x.x)
- Blocks reserved IP ranges

#### Cloud Metadata Protection

- Blocks AWS/Azure/GCP metadata endpoints (169.254.169.254)
- Blocks IPv6 metadata endpoints (fd00:ec2::254)

#### Hostname Validation

- Blocks internal TLDs (.local, .internal, .private, .corp, .home, .lan)
- Blocks direct IP address access to private networks
- Validates both IPv4 and IPv6 addresses

#### Port Restrictions

- Blocks dangerous ports (SSH:22, Telnet:23, SMTP:25, MySQL:3306, PostgreSQL:5432, Redis:6379, MongoDB:27017, Elasticsearch:9200)

#### Pattern Detection

- Blocks encoded attack patterns (%00, %2e%2e, %2f%2f)
- Blocks dangerous schemes (file://, ftp://, javascript:, data://)
- Blocks authentication in URLs (user:pass@)
- Blocks path traversal attempts (..)

### Example Usage

```python
from app.utils.validators import validate_url, ValidationError

# Valid URL
url = validate_url("https://example.com/api")

# Blocked - private IP
try:
    validate_url("http://192.168.1.1")
except ValidationError as e:
    print(f"Blocked: {e}")

# Allowed with flag
url = validate_url("http://localhost:8000", allow_localhost=True)
```

## 🛡️ Injection Prevention

### Upload ID Validation

Prevents SQL injection and path traversal in upload IDs:

- Only allows alphanumeric, hyphens, and underscores
- Length limits (1-100 characters)
- Blocks special characters and SQL keywords

```python
from app.utils.validators import validate_upload_id

# Valid
upload_id = validate_upload_id("550e8400-e29b-41d4-a716-446655440000")

# Blocked - SQL injection attempt
validate_upload_id("123'; DROP TABLE uploads; --")  # Raises ValidationError
```

### Namespace Validation

Prevents path traversal and injection in Pinecone namespaces:

- Only allows alphanumeric, hyphens, underscores, and dots
- Blocks path traversal (..)
- Blocks leading/trailing dots
- Length limits (1-200 characters)

```python
from app.utils.validators import validate_namespace

# Valid
namespace = validate_namespace("org-123")

# Blocked - path traversal
validate_namespace("org-123/../admin")  # Raises ValidationError
```

### Organization ID Validation

Prevents injection in organization identifiers:

- Only allows alphanumeric, hyphens, and underscores
- Length limits (1-100 characters)
- Blocks SQL injection patterns

### Metadata Filter Validation

Comprehensive validation for Pinecone metadata filters:

#### Whitelist Approach

Only allows specific, safe keys:

- `upload_id`, `org_id`, `source`, `type`, `chunk_index`
- `has_products`, `created_at`, `updated_at`

#### Value Validation

- String values: checked for injection patterns ($, {}, .., /, \, null bytes)
- Integer values: range validation (0-1,000,000)
- Boolean values: type checking
- Length limits (max 500 characters for strings)

#### Structure Validation

- Maximum 10 keys per filter
- Non-empty filters required
- Type checking for all values

```python
from app.utils.validators import validate_metadata_filter

# Valid
filter_dict = {"upload_id": "123", "org_id": "org-456"}
validated = validate_metadata_filter(filter_dict)

# Blocked - injection attempt
validate_metadata_filter({"upload_id": "123$ne"})  # Raises ValidationError

# Blocked - disallowed key
validate_metadata_filter({"malicious_key": "value"})  # Raises ValidationError
```

## 📊 JSON Safety Validation

Prevents DoS attacks through malicious JSON structures:

### Depth Limiting

- Maximum nesting depth (default: 10 levels)
- Prevents stack overflow attacks

### Size Limiting

- Dictionary: max 1,000 keys
- List: max 10,000 items
- String: max 100,000 characters

### Type Validation

- Only allows safe JSON types (string, number, boolean, null, dict, list)
- Blocks custom objects
- Enforces string keys in dictionaries

```python
from app.utils.validators import validate_json_safe

# Valid
data = {"key": "value", "nested": {"level": 2}}
validate_json_safe(data)

# Blocked - too deep
deeply_nested = {"l1": {"l2": {"l3": {...}}}}  # 15 levels
validate_json_safe(deeply_nested, max_depth=10)  # Raises ValidationError
```

## 🔧 Integration with Vector Operations

The vector operations module now uses these validators:

```python
# Before (vulnerable)
index.delete(filter={"upload_id": user_input}, namespace=namespace)

# After (secure)
from app.utils.validators import validate_metadata_filter, validate_namespace

validated_filter = validate_metadata_filter({"upload_id": user_input})
validated_namespace = validate_namespace(namespace)
index.delete(filter=validated_filter, namespace=validated_namespace)
```

## 🧪 Testing

Comprehensive test suite in `tests/test_security_validators.py`:

- 50+ test cases covering all validators
- Tests for common attack patterns
- Tests for edge cases and boundary conditions

Run tests:

```bash
pytest tests/test_security_validators.py -v
```

## 📈 Security Impact

### Before

- ❌ SSRF attacks possible via URL manipulation
- ❌ SQL injection possible via upload IDs
- ❌ Path traversal possible via namespaces
- ❌ NoSQL injection possible via metadata filters
- ❌ DoS possible via malicious JSON structures

### After

- ✅ Comprehensive SSRF protection with DNS resolution
- ✅ SQL injection prevented via input sanitization
- ✅ Path traversal blocked at validation layer
- ✅ NoSQL injection prevented via whitelist approach
- ✅ DoS attacks mitigated via size/depth limits

## 🚀 Best Practices

### Always Validate User Input

```python
# Bad
def process_upload(upload_id: str):
    delete_vectors(upload_id)

# Good
def process_upload(upload_id: str):
    validated_id = validate_upload_id(upload_id)
    delete_vectors(validated_id)
```

### Use Whitelist Approach

```python
# Bad - blacklist (can be bypassed)
if "$" not in value and "{" not in value:
    use_value(value)

# Good - whitelist (only allow known-safe)
allowed_keys = {"upload_id", "org_id", "type"}
if key in allowed_keys:
    use_value(value)
```

### Validate at Multiple Layers

```python
# 1. Pydantic model validation
class Request(BaseModel):
    upload_id: str

    @validator('upload_id')
    def validate_id(cls, v):
        return validate_upload_id(v)

# 2. Business logic validation
def process(request: Request):
    # Additional validation if needed
    pass

# 3. Database layer validation
# Use parameterized queries, never string concatenation
```

## 📝 Migration Guide

### Updating Existing Code

1. **Import validators**

```python
from app.utils.validators import (
    validate_url,
    validate_upload_id,
    validate_namespace,
    validate_metadata_filter,
    ValidationError
)
```

2. **Add validation before operations**

```python
# Before
def delete_upload(upload_id: str, namespace: str):
    index.delete(filter={"upload_id": upload_id}, namespace=namespace)

# After
def delete_upload(upload_id: str, namespace: str):
    try:
        validated_id = validate_upload_id(upload_id)
        validated_ns = validate_namespace(namespace)
        validated_filter = validate_metadata_filter({"upload_id": validated_id})
        index.delete(filter=validated_filter, namespace=validated_ns)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
```

3. **Handle validation errors**

```python
try:
    validated_url = validate_url(user_url)
except ValidationError as e:
    return {"error": f"Invalid URL: {str(e)}"}
```

## 🔍 Monitoring

### Log Validation Failures

```python
import logging

logger = logging.getLogger(__name__)

try:
    validate_url(suspicious_url)
except ValidationError as e:
    logger.warning(
        "Blocked suspicious URL",
        extra={
            "url_domain": urlparse(suspicious_url).netloc,
            "error": str(e),
            "user_id": user_id
        }
    )
```

### Track Attack Patterns

Monitor logs for:

- Repeated validation failures from same user/IP
- Specific attack patterns (SQL injection, SSRF attempts)
- Unusual input patterns

## 🎯 Future Improvements

1. **Rate Limiting on Validation Failures**

   - Implement rate limiting for users with repeated validation failures
   - Temporary blocks for suspicious activity

2. **Machine Learning Detection**

   - Train models to detect novel attack patterns
   - Adaptive validation rules

3. **Centralized Security Dashboard**

   - Real-time monitoring of validation failures
   - Attack pattern visualization
   - Automated alerting

4. **Additional Validators**
   - Email validation with disposable email detection
   - Phone number validation
   - Credit card validation (if needed)

## 📚 References

- [OWASP SSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
- [OWASP Input Validation Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)
- [CWE-918: Server-Side Request Forgery (SSRF)](https://cwe.mitre.org/data/definitions/918.html)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)

## 📞 Support

For questions or issues related to security validation:

- Create an issue in the repository
- Contact the security team
- Review test cases for examples

---

**Last Updated**: 2025-01-08
**Version**: 2.0
**Status**: Production Ready ✅

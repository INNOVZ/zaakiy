# Debugging Guide: Forbidden Phrase Still Appearing

## Issues Fixed

### 1. ✅ Hardcoded Fallback Messages in `prompt_sanitizer.py`
**Problem**: `sanitize_fallback()` was returning hardcoded responses with forbidden phrase:
- `"I'm sorry, I don't have information about that."`

**Fix**: Changed to constructive message:
- `"I'd be happy to help you with that! Could you provide more details or rephrase your question so I can assist you better?"`

### 2. ✅ Fallback Responses in `response_generation_service.py`
**Problem**: `generate_fallback_response()` was using chatbot's `fallback_message` from database without cleaning it.

**Fix**: Added forbidden phrase removal to clean fallback messages before returning.

## Why It Might Still Not Work

### 1. **Server Not Restarted** ⚠️
**Most Common Issue**: The server is still running old code.

**Solution**:
```bash
# Stop the server completely
# Then restart it
cd backend
python start_server.py
# Or however you start your server
```

### 2. **Cached Responses** ⚠️
**Problem**: Old cached responses might contain forbidden phrases.

**Solution**:
```bash
# Clear Redis cache (if using Redis)
redis-cli FLUSHALL

# Or clear the cache service
# Check your cache implementation
```

### 3. **Database Fallback Messages** ⚠️
**Problem**: The chatbot's `fallback_message` in the database might contain the forbidden phrase.

**Solution**:
1. Check the database for chatbots with bad fallback messages:
   ```sql
   SELECT id, name, fallback_message FROM chatbots
   WHERE fallback_message LIKE '%don''t have information%';
   ```

2. Update them:
   ```sql
   UPDATE chatbots
   SET fallback_message = 'I''d be happy to help you with that! Could you provide more details or rephrase your question so I can assist you better?'
   WHERE fallback_message LIKE '%don''t have information%';
   ```

### 4. **LLM Still Generating It** ⚠️
**Problem**: Even with prompt engineering, the LLM might still generate the forbidden phrase.

**Solution**: The post-processing should catch it, but verify:
- Check logs for `🚨 CRITICAL: Detected forbidden phrase`
- Verify `_remove_forbidden_phrases` is being called
- Check if responses are going through `_format_response`

## Debugging Steps

### Step 1: Verify Code is Loaded
```bash
# Check if the new code is in the file
grep -n "safe_fallback" backend/app/services/chat/prompt_sanitizer.py

# Should show line 234 with the new safe fallback message
```

### Step 2: Check Server Logs
```bash
# Watch logs in real-time
tail -f backend/app/logs/zaaky_$(date +%Y%m%d).log | grep -i "forbidden"

# Look for:
# - 🔍 _remove_forbidden_phrases called
# - 🚨 CRITICAL: Detected forbidden phrase
# - 🚨 REWRITTEN: Forbidden phrase detected and replaced
```

### Step 3: Test the Detection
```python
# Run this in Python to test detection
from app.services.chat.response_generation_service import ResponseGenerationService

service = ResponseGenerationService(
    org_id='test',
    openai_client=None,
    context_config=None,
    chatbot_config={'name': 'Test', 'tone': 'friendly', 'model': 'gpt-3.5-turbo', 'temperature': 0.2, 'max_tokens': 300},
)

test_response = "I don't have information about an office in Spain."
cleaned = service._remove_forbidden_phrases(
    test_response,
    {'demo_links': [], 'contact_info': {}},
    'Do you have office in Spain?'
)

print('Original:', test_response)
print('Cleaned:', cleaned)
print('Contains forbidden phrase:', 'don\'t have information about' in cleaned.lower())
```

### Step 4: Clear Cache
```bash
# If using Redis
redis-cli FLUSHALL

# Or restart Redis
redis-cli SHUTDOWN
# Then restart Redis
```

### Step 5: Check Database
```sql
-- Check for bad fallback messages
SELECT id, name, fallback_message
FROM chatbots
WHERE fallback_message LIKE '%don''t have information%'
   OR fallback_message LIKE '%don''t have that information%';

-- Update them
UPDATE chatbots
SET fallback_message = 'I''d be happy to help you with that! Could you provide more details or rephrase your question so I can assist you better?'
WHERE fallback_message LIKE '%don''t have information%'
   OR fallback_message LIKE '%don''t have that information%';
```

## Verification Checklist

- [ ] Server has been restarted with new code
- [ ] Cache has been cleared
- [ ] Database fallback messages have been updated
- [ ] Logs show `_remove_forbidden_phrases` being called
- [ ] Test query triggers detection in logs
- [ ] Response doesn't contain forbidden phrases

## Expected Behavior After Fix

1. **New Responses**: Should not contain forbidden phrases
2. **Cached Responses**: Should be cleaned when retrieved
3. **Fallback Responses**: Should use safe fallback messages
4. **Database Fallback**: Should be cleaned when used

## If Still Not Working

1. **Check Response Source**: Verify which code path is generating the response
   - Is it from `generate_enhanced_response`?
   - Is it from `generate_fallback_response`?
   - Is it from `error_handler.create_fallback_response`?

2. **Check Logs**: Look for which function is generating the response
   - Search for the exact response text in logs
   - Check timestamps to see when it was generated

3. **Test Directly**: Use the test script to verify detection works
   - If tests pass but production doesn't, it's likely a cache/server issue

4. **Check Multiple Chatbots**: Different chatbots might have different fallback messages
   - Update all chatbots in the database

## Files Modified

1. `backend/app/services/chat/prompt_sanitizer.py`
   - Changed `sanitize_fallback()` to use safe fallback message

2. `backend/app/services/chat/response_generation_service.py`
   - Added forbidden phrase removal to `generate_fallback_response()`

## Next Steps

1. **Restart the server** (CRITICAL)
2. **Clear the cache** (if using caching)
3. **Update database fallback messages** (if they contain forbidden phrases)
4. **Test with a fresh query** (not cached)
5. **Check logs** to verify detection is working

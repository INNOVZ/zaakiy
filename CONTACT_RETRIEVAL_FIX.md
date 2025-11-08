# 🔧 Contact Information Retrieval Fix

## Problem Summary

The chat system was not correctly retrieving and displaying:
1. **Phone numbers** - Exists in Pinecone but not fetched/displayed
2. **Location/Address** - Exists in Pinecone but inconsistently retrieved
3. **Demo/Booking Links** - Different links being returned instead of the exact link from Pinecone

## Root Causes Identified

1. **Chunk Extraction Issues**:
   - Chunks stored as JSON-like strings in Pinecone metadata weren't being properly parsed
   - Contact information embedded in serialized JSON wasn't extracted

2. **No Contact Information Prioritization**:
   - Contact queries didn't prioritize chunks containing contact info
   - Context building didn't ensure contact info was included

3. **Missing Contact Extraction**:
   - No specialized extraction for phone numbers, emails, addresses, and demo links
   - Demo links weren't identified and preserved separately

4. **Context Truncation**:
   - Important contact information was being truncated due to context length limits
   - Contact chunks weren't prioritized in context building

## Solution Implemented

### 1. Created Contact Extractor Module (`contact_extractor.py`)

**Features**:
- Extracts phone numbers using multiple international formats (UAE, US, etc.)
- Extracts email addresses
- Extracts addresses using pattern matching
- Identifies and extracts demo/booking links
- Parses JSON-like strings from chunks
- Scores chunks based on contact information content

**Key Functions**:
- `extract_contact_info()` - Main extraction function
- `score_chunk_for_contact_query()` - Scores chunks for contact relevance
- `_try_parse_json()` - Attempts to parse JSON from text chunks
- `_identify_demo_links()` - Identifies demo/booking links

### 2. Enhanced Document Retrieval Service

**Changes**:
- **Contact Query Detection**: Detects contact-related queries (phone, email, contact, demo, booking, etc.)
- **Increased Retrieval**: Returns up to 15 documents for contact queries (instead of 5)
- **Contact Scoring**: Re-scores documents based on contact information content
- **Contact Boosting**: Boosts scores of documents containing contact info
- **Better Chunk Extraction**: Tries multiple metadata fields (chunk, text, content)
- **Enhanced Logging**: Logs contact information found in retrieved documents

### 3. Enhanced Response Generation Service

**Changes**:
- **Contact Info Extraction**: Extracts contact info from all retrieved documents
- **Contact Prioritization**: Prioritizes chunks with contact info in context building
- **Structured Contact Section**: Creates a structured "CONTACT INFORMATION" section in context
- **Demo Link Preservation**: Ensures demo links from context are included in responses
- **Enhanced System Prompt**: Added explicit instructions for demo/booking links
- **Response Validation**: Validates that contact info in response matches context
- **Auto-link Insertion**: Automatically adds demo links if missing from response

### 4. Improved Context Building

**Changes**:
- **Contact Chunks First**: Places chunks with contact info at the beginning of context
- **Contact Info Section**: Adds structured contact info section to context
- **Prevents Truncation**: Ensures contact info is always included even if context is long
- **Better Logging**: Logs what contact information was extracted

## Key Features

### Contact Information Extraction

✅ **Phone Numbers**: Supports multiple formats
- International: `+971 52 867 8679`
- Local: `052 867 8679`
- Various spacing formats

✅ **Email Addresses**: Standard email pattern matching

✅ **Addresses**: Pattern-based extraction for:
- Building names
- Street addresses
- City/Country (Dubai, UAE, etc.)

✅ **Demo/Booking Links**: Identifies links from:
- SurveySparrow
- Calendly
- Custom booking platforms
- Links containing "demo", "booking", "schedule" keywords

### Contact Query Handling

✅ **Automatic Detection**: Detects contact-related queries
✅ **Enhanced Retrieval**: Retrieves more documents for contact queries
✅ **Score Boosting**: Boosts documents with contact info
✅ **Prioritized Context**: Places contact info at the beginning of context

### Demo Link Handling

✅ **Link Extraction**: Extracts demo/booking links from chunks
✅ **Link Preservation**: Ensures exact links from Pinecone are used
✅ **Link Validation**: Prevents using different links than what's in context
✅ **Auto-Insertion**: Adds demo links to response if missing

## Testing

### Test Results

✅ **Phone Extraction**: Successfully extracts `+971 52 867 8679`
✅ **Demo Link Extraction**: Successfully extracts `https://innovz.surveysparrow.com/s/Zaakiy-onboarding/tt-NwNkd`
✅ **Address Extraction**: Successfully extracts addresses
✅ **Contact Info Detection**: Correctly identifies chunks with contact info

### Test File

Run the test:
```bash
python3 test_contact_extractor_standalone.py
```

## Files Modified

1. **`app/services/chat/contact_extractor.py`** (NEW)
   - Contact information extraction module

2. **`app/services/chat/document_retrieval_service.py`**
   - Enhanced contact query detection
   - Contact scoring and boosting
   - Better chunk extraction

3. **`app/services/chat/response_generation_service.py`**
   - Contact info extraction from documents
   - Contact prioritization in context
   - Demo link handling
   - Enhanced system prompt

## Deployment

### Before Deployment

1. **Clear Cache**: Clear retrieval cache to ensure fresh results
   ```bash
   # Clear Redis cache if using Redis
   redis-cli FLUSHDB
   ```

2. **Force Rebuild**: Use force rebuild to ensure new code is deployed
   ```bash
   ./deploy.sh force
   ```

### After Deployment

1. **Test Contact Queries**:
   - "How can I contact you?"
   - "What's your phone number?"
   - "How can I book a demo?"

2. **Verify**:
   - Phone numbers are displayed
   - Addresses are displayed
   - Demo links match Pinecone data
   - No hallucinations or incorrect links

## Monitoring

### Logs to Watch

1. **Contact Query Detection**:
   ```
   🔍 Contact query detected - returning X documents (boosted by contact info)
   ```

2. **Contact Information Found**:
   ```
   ✅ Extracted contact info: phones=X, emails=X, demo_links=X, addresses=X
   ```

3. **Missing Contact Info** (should not appear):
   ```
   ⚠️ Contact query detected but NO contact info found in any retrieved documents!
   ```

### Metrics

- Contact info extraction rate
- Demo link accuracy
- Phone number accuracy
- Response quality for contact queries

## Expected Improvements

1. ✅ **Phone Numbers**: Always retrieved and displayed for contact queries
2. ✅ **Addresses**: Consistently retrieved and displayed
3. ✅ **Demo Links**: Exact links from Pinecone are used (no incorrect links)
4. ✅ **Response Quality**: Better responses for contact-related queries
5. ✅ **Retrieval Accuracy**: More relevant documents retrieved for contact queries

## Troubleshooting

### Issue: Still not retrieving phone numbers

**Check**:
1. Verify phone numbers exist in Pinecone metadata
2. Check logs for contact query detection
3. Verify chunk extraction is working
4. Check if contact info is being extracted from chunks

### Issue: Wrong demo link

**Check**:
1. Verify correct link exists in Pinecone
2. Check if demo link extraction is working
3. Verify context includes demo links
4. Check system prompt instructions

### Issue: Contact info not in context

**Check**:
1. Verify documents are being retrieved (check logs)
2. Check contact scoring and boosting
3. Verify chunk extraction from metadata
4. Check context length limits

## Future Enhancements

1. **Address Formatting**: Improve address extraction regex
2. **Link Validation**: Validate demo links before using
3. **Contact Info Caching**: Cache extracted contact info
4. **Multi-language Support**: Support contact info in multiple languages
5. **Contact Info Deduplication**: Better handling of duplicate contact info

## Summary

This fix addresses the root causes of contact information retrieval issues by:
1. ✅ Creating specialized contact extraction
2. ✅ Prioritizing contact info in retrieval and context
3. ✅ Ensuring demo links are preserved and used correctly
4. ✅ Adding comprehensive logging for debugging
5. ✅ Improving system prompts for better AI responses

The system now reliably retrieves and displays phone numbers, addresses, and demo links from Pinecone data.

# Robust Scraping System Architecture

## Overview

This document describes the comprehensive scraping system that handles all types of content: URLs (including modern JS frameworks and e-commerce), PDFs, and JSON files.

## System Architecture

### 1. Unified Scraper (`unified_scraper.py`)

The `UnifiedScraper` class provides a single entry point for all URL scraping with intelligent strategy selection and fallbacks.

**Features:**

- Automatic strategy selection based on URL type
- Multiple retry attempts with exponential backoff
- Comprehensive error handling
- Structured product data extraction for e-commerce sites

**Scraping Strategies (in order):**

1. **E-commerce Scraper** (for Shopify, WooCommerce, BigCommerce, etc.)

   - Extracts structured product data
   - Handles product cards, collections, and single product pages
   - Extracts: title, price, description, SKU, availability, images, product links
   - Timeout: 90 seconds (configurable)

2. **Playwright Scraper** (for React, Next.js, Vue, Angular, etc.)

   - Renders JavaScript before extraction
   - Waits for content to load
   - Handles lazy-loaded content
   - Timeout: 30 seconds (configurable)

3. **Traditional Scraper** (fallback for static HTML)
   - Fast HTTP-based scraping
   - No JavaScript rendering
   - Good for traditional websites

### 2. E-commerce Scraper (`ecommerce_scraper.py`)

Specialized scraper for e-commerce platforms with enhanced product extraction.

**Capabilities:**

- **Product Card Extraction**: Finds products in collection/listing pages
- **Single Product Extraction**: Detailed product information
- **Product Link Discovery**: Extracts all product URLs from pages
- **Structured Output**: Formats data for easy indexing

**Supported Platforms:**

- Shopify (including .myshopify.com domains)
- WooCommerce
- BigCommerce
- Magento
- Generic e-commerce sites

**Product Data Extracted:**

- Title (multiple fallback strategies)
- Price (with currency detection)
- Description
- SKU
- Availability/Stock status
- Product images
- Product links (absolute URLs)
- Brand
- Category
- Specifications

### 3. Playwright Scraper (`playwright_scraper.py`)

Handles modern JavaScript-rendered websites.

**Features:**

- Full JavaScript execution
- Waits for network idle
- Handles lazy-loaded content
- Removes UI noise (nav, footer, modals)
- Extracts contact information

**Optimizations:**

- Special handling for Shopify sites (longer wait times)
- Waits for content selectors
- Handles slow-loading sites

### 4. PDF Scraping (`ingestion_worker.py`)

Robust PDF text extraction with memory management.

**Features:**

- Streaming download (prevents memory issues)
- Size limits (100MB max)
- Page limits (1000 pages max)
- Progress logging for large PDFs
- Handles scanned/image-based PDFs gracefully

**Process:**

1. Download PDF in chunks
2. Validate PDF format
3. Extract text page by page
4. Clean and validate extracted text

### 5. JSON Scraping (`ingestion_worker.py`)

Flexible JSON data extraction.

**Features:**

- Handles various JSON structures
- Size limits (50MB max)
- Extracts from common fields: `content`, `text`, `body`
- Converts arrays to text
- Handles large arrays (limits to 10,000 items)

## Usage

### Basic URL Scraping

```python
from app.services.scraping.unified_scraper import scrape_url_unified

text = await scrape_url_unified("https://example.com")
```

### URL Scraping with Products

```python
from app.services.scraping.unified_scraper import scrape_url_with_products

result = await scrape_url_with_products("https://shop.example.com")
# result contains: text, products, product_urls
```

### Using UnifiedScraper Directly

```python
from app.services.scraping.unified_scraper import UnifiedScraper

scraper = UnifiedScraper(
    ecommerce_timeout=90000,
    playwright_timeout=30000,
    max_retries=3
)

result = await scraper.scrape("https://example.com", extract_products=True)
```

## Error Handling

The system includes comprehensive error handling:

1. **Retry Logic**: Up to 3 attempts with exponential backoff
2. **Fallback Chains**: If one scraper fails, automatically tries the next
3. **Detailed Logging**: All failures are logged with context
4. **Graceful Degradation**: Returns partial results when possible

## Performance Optimizations

1. **Caching**: Scraped content can be cached (disabled for e-commerce by default)
2. **Streaming**: Large files are processed in chunks
3. **Memory Management**: Explicit cleanup to prevent leaks
4. **Timeout Management**: Configurable timeouts per scraper type

## E-commerce Product Extraction

For e-commerce sites, the system extracts:

### Product Cards (Collection Pages)

- Title (with multiple fallback strategies)
- Price (with data attributes support)
- Description
- Product link (absolute URL)
- Image URL
- SKU
- Availability

### Single Product Pages

- All product card fields, plus:
- Full description
- Multiple images
- Specifications/attributes
- Brand
- Category
- Reviews summary

### Product Link Discovery

- Extracts from product cards
- Finds product URLs in page links
- Cleans and normalizes URLs
- Removes duplicate URLs

## Chunking and Filtering

After scraping, content is:

1. **Chunked**: Split into 800-character chunks with 200-character overlap
2. **Filtered**: Removes UI noise (login buttons, cart elements, etc.)
3. **E-commerce Lenient Filtering**: Uses progressive fallback (60 → 30 → 20 → 10 chars)
4. **Last Resort**: Keeps any chunk > 5 chars that isn't pure UI noise

## Supported Website Types

✅ **Modern JavaScript Frameworks**

- React
- Next.js
- Vue.js
- Angular
- Svelte
- Any JavaScript-rendered site

✅ **E-commerce Platforms**

- Shopify
- WooCommerce
- BigCommerce
- Magento
- Squarespace Commerce
- Wix Stores
- Generic e-commerce

✅ **Traditional Websites**

- Static HTML
- Server-rendered pages
- PHP sites
- Any standard HTML

✅ **File Types**

- PDF documents
- JSON data files

## Configuration

Key configuration options:

- `ecommerce_timeout`: Timeout for e-commerce scrapers (default: 90s)
- `playwright_timeout`: Timeout for Playwright (default: 30s)
- `max_retries`: Maximum retry attempts (default: 3)
- `extract_products`: Whether to extract structured product data (default: True)

## Logging

The system provides detailed logging at each step:

- Strategy selection
- Retry attempts
- Success/failure with metrics
- Product extraction counts
- Error details

## Future Enhancements

Potential improvements:

1. **Parallel Scraping**: Scrape multiple product pages simultaneously
2. **Image Extraction**: Extract and store product images
3. **Review Extraction**: Extract customer reviews
4. **Price History**: Track price changes
5. **Inventory Monitoring**: Track stock levels

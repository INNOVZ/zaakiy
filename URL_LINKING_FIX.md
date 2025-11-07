# 🔗 URL Linking Fix - Book Demo & Website Links

## Problem
When users asked "how can I book a demo?" or the AI mentioned "our website", the links were not showing up as clickable. The text "book a demo" or "our website" appeared as plain text without the actual URL from context.

## Solution

### 1. Enhanced System Prompt
Added explicit instructions for URL handling:
- ✅ **ALWAYS format URLs as markdown links**: `[link text](URL)`
- ✅ **When saying "book a demo"**, you MUST include the actual URL from context as: `[book a demo](actual-url-from-context)`
- ✅ **When saying "our website"**, you MUST include the actual URL from context as: `[our website](actual-url-from-context)`
- ✅ **NEVER say "book a demo" or "visit our website" without including the actual clickable URL**
- ✅ Extract URLs from context and use them

### 2. URL Extraction (`_extract_urls_from_context`)
- Extracts all URLs from context text
- Handles various formats: `http://`, `https://`, `www.`, and domain-only URLs
- Automatically adds `https://` prefix when missing
- Filters and normalizes URLs
- Returns list of valid, unique URLs

### 3. URL Formatting (`_format_urls_as_links`)
- Converts raw URLs in responses to markdown links
- Detects URLs already in markdown format to avoid double-linking
- Formats URLs as: `[domain](full-url)`
- Example: `https://example.com` → `[example.com](https://example.com)`

### 4. Phrase-to-URL Linking (`_link_common_phrases_to_urls`)
- Automatically links common phrases to URLs from context
- **Demo/Booking URLs**: Links phrases like:
  - "book a demo" → `[book a demo](demo-url)`
  - "schedule a demo" → `[schedule a demo](demo-url)`
  - "book your demo" → `[book your demo](demo-url)`
- **Website URLs**: Links phrases like:
  - "our website" → `[our website](website-url)`
  - "visit our website" → `[visit our website](website-url)`
  - "go to our website" → `[go to our website](website-url)`
  - "website" → `[website](website-url)`

### 5. Smart URL Detection
- **Demo URLs**: Identified by keywords: `demo`, `book`, `schedule`, `appointment`, `booking`, `calendly`, `cal.com`, `meet`, `zoom`
- **Website URLs**: All other URLs (general website URLs)
- Prioritizes demo URLs for booking phrases
- Falls back to demo URL for website phrases if no website URL exists

### 6. Debugging & Logging
- Logs when URLs are found in context
- Logs when demo/booking URLs are detected
- Logs when phrases are linked to URLs
- Helps troubleshoot linking issues

## How It Works

1. **Context Analysis**: When processing a query:
   - Extracts all URLs from the context
   - Identifies demo/booking URLs vs. website URLs
   - Logs findings for debugging

2. **AI Response Generation**: The system prompt instructs the AI to:
   - Include URLs when mentioning "book a demo" or "our website"
   - Format URLs as markdown links

3. **Post-Processing**: After AI generates response:
   - Links common phrases to URLs from context (if AI didn't include them)
   - Formats any raw URLs as markdown links
   - Ensures all URLs are clickable

## Examples

### Before Fix:
**User**: "how can I book a demo?"
**AI Response**: "To book a demo, you can visit our website and schedule a demo with our team." ❌ (no clickable links)

### After Fix:
**User**: "how can I book a demo?"
**AI Response**: "To [book a demo](https://example.com/book-demo), you can visit [our website](https://example.com) and schedule a demo with our team." ✅ (clickable links)

Or if context has a demo URL:
**AI Response**: "You can [book a demo](https://calendly.com/company/demo) to schedule a meeting with our team." ✅

## URL Detection Keywords

### Demo/Booking URLs:
- `demo`
- `book`
- `schedule`
- `appointment`
- `booking`
- `calendly`
- `cal.com`
- `meet`
- `zoom`

### Website URLs:
- Any URL that doesn't match demo/booking keywords

## Supported URL Formats

The system can extract URLs in various formats:
- `https://example.com/book-demo`
- `http://example.com`
- `www.example.com` (automatically converts to `https://www.example.com`)
- `example.com` (automatically converts to `https://example.com`)
- URLs with paths: `https://example.com/path/to/page`
- URLs with query parameters: `https://example.com?param=value`

## Testing

To test the fix:
1. **Ask**: "how can I book a demo?"
   - Should see clickable link: `[book a demo](url)`

2. **Ask**: "what's your website?"
   - Should see clickable link: `[our website](url)` or `[website](url)`

3. **Check logs** for:
   - `🔗 Found X URLs in context`
   - `📅 Found demo/booking URLs`
   - `✅ Linked 'book a demo' to <url>`
   - `✅ Linked 'our website' to <url>`

## Deployment

After deploying this fix:
1. URLs from context will be automatically extracted
2. Common phrases will be linked to appropriate URLs
3. All URLs will be formatted as clickable markdown links
4. Users will see clickable links for "book a demo" and "our website"

## Notes

- The system prioritizes demo URLs for booking-related phrases
- If no demo URL exists, it uses the website URL for booking phrases
- If no URLs are found in context, phrases remain as plain text (no broken links)
- The system avoids double-linking (won't link if already in markdown format)

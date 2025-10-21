# Streaming Response Implementation Guide

## Overview

This guide outlines how to implement streaming responses for the chat service. Streaming responses provide perceived performance improvements by sending tokens as they're generated, rather than waiting for the complete response.

## Benefits

- **Perceived Performance:** Users see responses immediately (first token in ~200-500ms)
- **Better UX:** Appears more conversational and responsive
- **Reduced Wait Time:** Total time unchanged, but feels much faster
- **Real-time Feedback:** Users know the system is working

## Implementation Steps

### 1. Backend Changes

#### Update Response Generation Service

```python
# backend/app/services/chat/response_generation_service.py

async def generate_streaming_response(
    self,
    message: str,
    conversation_history: List[Dict[str, Any]],
    retrieved_documents: List[Dict[str, Any]],
):
    """Generate streaming response with OpenAI"""
    try:
        # Build context and prompts (same as before)
        context_data = self._build_context(retrieved_documents)
        system_prompt = self._create_system_prompt(context_data)
        messages = self._build_conversation_messages(
            system_prompt, conversation_history, message
        )

        # Call OpenAI with streaming enabled
        model = self.chatbot_config.get("model", "gpt-3.5-turbo")
        temperature = self.chatbot_config.get("temperature", 0.7)
        max_tokens = self.chatbot_config.get("max_tokens", 500)

        response_stream = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,  # Enable streaming!
        )

        # Yield tokens as they arrive
        full_response = ""
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token

                # Yield the token to the client
                yield {
                    "type": "token",
                    "content": token,
                    "done": False
                }

        # Send final metadata
        yield {
            "type": "complete",
            "content": full_response,
            "done": True,
            "sources": context_data.get("sources", []),
            "context_quality": context_data.get("context_quality", {})
        }

    except Exception as e:
        logger.error("Streaming response generation failed: %s", e)
        yield {
            "type": "error",
            "error": str(e),
            "done": True
        }
```

#### Update Public Chat Router

```python
# backend/app/routers/public_chat.py

from fastapi.responses import StreamingResponse
import json

@router.post("/chat/stream")
@rate_limit(**get_rate_limit_config("public_chat"))
async def public_chat_stream(request: PublicChatRequest):
    """Public chat endpoint with streaming responses"""
    try:
        # Get chatbot config (cached)
        chatbot = await get_cached_chatbot_config(request.chatbot_id)

        # Initialize chat service
        chat_service = ChatService(
            org_id=chatbot["org_id"],
            chatbot_config=chatbot,
            entity_id=chatbot["org_id"],
            entity_type="organization",
        )

        # Define streaming generator
        async def generate_stream():
            try:
                # Get conversation and history
                conversation = await chat_service.conversation_manager.get_or_create_conversation(
                    session_id=request.session_id or "anonymous"
                )

                # Add user message
                await chat_service.conversation_manager.add_message(
                    conversation_id=conversation["id"],
                    role="user",
                    content=request.message
                )

                # Get history
                history = await chat_service.conversation_manager.get_conversation_history(
                    conversation_id=conversation["id"],
                    limit=10
                )

                # Enhance query
                enhanced_queries = await chat_service.response_generator.enhance_query(
                    request.message, history
                )

                # Retrieve documents
                documents = await chat_service.document_retrieval.retrieve_documents(
                    queries=enhanced_queries
                )

                # Stream response
                full_response = ""
                async for chunk in chat_service.response_generator.generate_streaming_response(
                    message=request.message,
                    conversation_history=history,
                    retrieved_documents=documents,
                ):
                    # Send SSE formatted data
                    yield f"data: {json.dumps(chunk)}\n\n"

                    if chunk.get("type") == "complete":
                        full_response = chunk.get("content", "")

                        # Save assistant message
                        await chat_service.conversation_manager.add_message(
                            conversation_id=conversation["id"],
                            role="assistant",
                            content=full_response,
                            metadata={
                                "sources": chunk.get("sources", []),
                                "context_quality": chunk.get("context_quality", {})
                            }
                        )

            except Exception as e:
                logger.error("Streaming error: %s", e)
                error_chunk = {
                    "type": "error",
                    "error": str(e),
                    "done": True
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        # Return streaming response with SSE
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Public chat streaming failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat service unavailable") from e
```

### 2. Frontend Changes

#### Update Chat Widget to Support Streaming

```javascript
// frontend/public/chat-widget.js

sendMessage: async function(message) {
    if (!message.trim()) return;

    var input = document.querySelector('.zaaky-input');
    var messagesContainer = document.querySelector('.zaaky-messages');

    if (!input || !messagesContainer) return;

    // Add user message
    this.addMessage(message, 'user');
    input.value = '';

    // Add placeholder for bot response
    var botMessageDiv = document.createElement('div');
    botMessageDiv.className = 'zaaky-message bot';
    botMessageDiv.style.cssText = `
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        background: #f8f9fa;
        margin-right: 40px;
    `;
    botMessageDiv.textContent = '...';
    messagesContainer.appendChild(botMessageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        // Use EventSource for Server-Sent Events (SSE)
        const url = new URL(this.config.apiUrl + '/chat/stream');

        // For POST with SSE, we need to use fetch with stream
        const response = await fetch(this.config.apiUrl + '/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                chatbot_id: this.config.chatbotId,
                session_id: this.getSessionId()
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            // Decode the chunk
            const chunk = decoder.decode(value, { stream: true });

            // Parse SSE data
            const lines = chunk.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));

                    if (data.type === 'token') {
                        // Append token to response
                        fullResponse += data.content;
                        botMessageDiv.textContent = fullResponse;
                        messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    }
                    else if (data.type === 'complete') {
                        // Response complete
                        fullResponse = data.content;
                        botMessageDiv.textContent = fullResponse;
                        break;
                    }
                    else if (data.type === 'error') {
                        throw new Error(data.error);
                    }
                }
            }
        }

    } catch (error) {
        console.error('ZaaKy Streaming Error:', error);
        botMessageDiv.textContent = 'Sorry, I encountered an error. Please try again.';
    }
},
```

### 3. Nginx Configuration

#### Update nginx.conf for SSE support

```nginx
# backend/nginx.conf

location /api/public/chat/stream {
    proxy_pass http://backend:8001;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # SSE specific settings
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;

    # Prevent buffering
    chunked_transfer_encoding on;
    tcp_nopush off;
    tcp_nodelay on;
}
```

### 4. Testing

#### Test with curl

```bash
# Test streaming endpoint
curl -N -X POST http://localhost:8001/api/public/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "chatbot_id": "your-chatbot-id",
    "session_id": "test-session"
  }'

# Expected output:
# data: {"type":"token","content":"Hello","done":false}
# data: {"type":"token","content":"!","done":false}
# data: {"type":"token","content":" I","done":false}
# ...
# data: {"type":"complete","content":"Hello! I'm doing well...","done":true}
```

#### Test in browser

```javascript
// Open browser console and test
const eventSource = new EventSource("/api/public/chat/stream?...");
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log("Received:", data);
};
```

## Performance Comparison

### Without Streaming (Current)

```
User sends message → Wait 2-3 seconds → Complete response appears
└─ Perceived wait time: 2-3 seconds
```

### With Streaming

```
User sends message → 200ms → First word appears → More words stream in → Complete
└─ Perceived wait time: 200-500ms (first token)
└─ Total time: Still 2-3 seconds, but feels instant!
```

## Considerations

### Pros

- ✅ Much better perceived performance
- ✅ More engaging user experience
- ✅ Real-time feedback
- ✅ Works well with long responses

### Cons

- ❌ More complex to implement
- ❌ Harder to debug
- ❌ Requires SSE support in frontend
- ❌ May not work with all proxies/load balancers
- ❌ Token consumption tracking needs adjustment

## Alternative: Chunked Responses

If full streaming is too complex, consider chunked responses:

```python
# Send response in chunks
async def generate_chunked_response():
    # Generate full response
    full_response = await generate_response()

    # Split into chunks (by sentence or word count)
    chunks = split_into_chunks(full_response, chunk_size=50)

    # Send chunks with small delays
    for i, chunk in enumerate(chunks):
        yield {
            "chunk": chunk,
            "index": i,
            "done": i == len(chunks) - 1
        }
        await asyncio.sleep(0.1)  # Small delay between chunks
```

This provides some of the streaming benefits with less complexity.

## Migration Path

1. **Phase 1:** Implement streaming endpoint alongside existing endpoint
2. **Phase 2:** Test with subset of users
3. **Phase 3:** Monitor performance and user feedback
4. **Phase 4:** Gradually roll out to all users
5. **Phase 5:** Deprecate old endpoint after successful migration

## Monitoring

Add streaming-specific metrics:

```python
# Track time to first token
performance_monitor.track_operation("time_to_first_token")

# Track tokens per second
performance_monitor.track_operation("tokens_per_second")

# Track streaming errors
performance_monitor.track_operation("streaming_errors")
```

## Conclusion

Streaming responses provide the best perceived performance improvement but require significant implementation effort. The current optimizations (caching, parallelization, query enhancement optimization) provide real performance improvements without the complexity of streaming.

**Recommendation:** Implement streaming if:

- Users frequently report slow responses
- Responses are typically long (>100 tokens)
- You have development resources available
- Your infrastructure supports SSE

Otherwise, the current optimizations should provide substantial improvements (40-50% faster) with much less implementation complexity.

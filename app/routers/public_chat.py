import html
import os

from fastapi import APIRouter, HTTPException

from ..models import PublicChatRequest, PublicChatResponse
from ..services.chat.chat_service import ChatService
from ..services.shared import cache_service
from ..services.storage.supabase_client import get_supabase_client
from ..utils.logging_config import get_logger
from ..utils.rate_limiter import get_rate_limit_config, rate_limit

# Constants
CHATBOT_NOT_FOUND_MESSAGE = "Chatbot not found"
CHATBOT_NOT_FOUND_OR_INACTIVE_MESSAGE = "Chatbot not found or inactive"
CHATBOT_CACHE_TTL = 300  # Cache chatbot config for 5 minutes

# Get logger
logger = get_logger(__name__)

# Get centralized Supabase client
supabase = get_supabase_client()

router = APIRouter()


async def get_cached_chatbot_config(chatbot_id: str):
    """Get chatbot configuration with caching for better performance"""
    cache_key = f"chatbot_config:{chatbot_id}"

    # Try cache first
    try:
        cached_config = await cache_service.get(cache_key)
        if cached_config:
            logger.debug(f"Chatbot config cache HIT for {chatbot_id}")
            return cached_config
    except Exception as e:
        logger.warning(f"Cache retrieval failed: {e}")

    # Cache miss - fetch from database
    logger.debug(f"Chatbot config cache MISS for {chatbot_id}")
    response = (
        supabase.table("chatbots")
        .select("*")
        .eq("id", chatbot_id)
        .eq("chain_status", "active")
        .execute()
    )

    if not response.data or len(response.data) == 0:
        raise HTTPException(
            status_code=404, detail=CHATBOT_NOT_FOUND_OR_INACTIVE_MESSAGE
        )

    chatbot = response.data[0]

    # Cache for future requests
    try:
        await cache_service.set(cache_key, chatbot, CHATBOT_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Cache set failed: {e}")

    return chatbot


@router.post("/chat", response_model=PublicChatResponse)
@rate_limit(**get_rate_limit_config("public_chat"))
async def public_chat(request: PublicChatRequest):
    """Public chat endpoint for embedded chatbots - Optimized with caching"""
    try:
        logger.info(
            "Public chat request received",
            extra={
                "chatbot_id": request.chatbot_id,
                "message_length": len(request.message),
                "session_id": request.session_id,
            },
        )

        # Get chatbot configuration with caching
        chatbot = await get_cached_chatbot_config(request.chatbot_id)

        logger.info(
            "Chatbot config retrieved",
            extra={
                "chatbot_name": chatbot["name"],
                "org_id": chatbot["org_id"],
            },
        )

        # Initialize chat service with required parameters
        try:
            chat_service = ChatService(
                org_id=chatbot["org_id"],
                chatbot_config=chatbot,
                entity_id=chatbot["org_id"],  # Use organization ID for public chat
                entity_type="organization",  # Public chat consumes tokens from organization's subscription
            )
            logger.info("ChatService initialized successfully")
        except Exception as init_error:
            logger.error(
                "Failed to initialize ChatService",
                extra={"error": str(init_error)},
                exc_info=True,
            )
            raise

        # Generate response using the existing chat method
        try:
            result = await chat_service.chat(
                message=request.message,
                session_id=request.session_id or "anonymous",
                chatbot_id=request.chatbot_id,
            )
            logger.info("Chat response generated successfully")
        except Exception as chat_error:
            logger.error(
                "Failed to generate chat response",
                extra={"error": str(chat_error)},
                exc_info=True,
            )
            raise

        return {
            "response": result["response"],
            "product_links": result.get("product_links", []),
            "chatbot": {
                "name": chatbot["name"],
                "avatar_url": chatbot.get("avatar_url"),
                "color_hex": chatbot["color_hex"],
            },
            "session_id": request.session_id or "anonymous",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Public chat request failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "chatbot_id": request.chatbot_id,
            },
            exc_info=True,
        )
        # Return more detailed error for debugging
        error_detail = f"Chat service unavailable: {type(e).__name__}"
        if hasattr(e, "args") and e.args:
            error_detail += f" - {str(e.args[0])[:100]}"
        raise HTTPException(status_code=500, detail=error_detail) from e


@router.get("/chatbot/{chatbot_id}/config")
async def get_public_chatbot_config(chatbot_id: str):
    """Get public chatbot configuration for embedding"""
    try:
        # Use proper Supabase client
        response = (
            supabase.table("chatbots")
            .select("*")
            .eq("id", chatbot_id)
            .eq("chain_status", "active")
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

        chatbot = response.data[0]

        # Return only public-safe information
        return {
            "id": chatbot["id"],
            "name": chatbot["name"],
            "avatar_url": chatbot.get("avatar_url"),
            "color_hex": chatbot["color_hex"],
            "description": chatbot.get("description", ""),
            "greeting_message": chatbot.get(
                "greeting_message", "Hello! How can I help you today?"
            ),
            "status": "active",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get chatbot configuration",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get chatbot configuration"
        ) from e


@router.get("/chatbot/{chatbot_id}/widget")
async def get_chatbot_widget_code(chatbot_id: str):
    """Get embeddable widget code for a chatbot - FIXED XSS vulnerability"""
    try:
        # Get chatbot config
        response = (
            supabase.table("chatbots")
            .select("*")
            .eq("id", chatbot_id)
            .eq("chain_status", "active")
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

        chatbot = response.data[0]

        # Get environment variables
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8001")

        # SECURITY FIX: Escape all user-provided data
        chatbot_id_escaped = html.escape(chatbot_id)
        chatbot_name_escaped = html.escape(str(chatbot["name"]))
        chatbot_color_escaped = html.escape(str(chatbot["color_hex"]))
        greeting_msg_escaped = html.escape(
            str(chatbot.get("greeting_message", "Hello! How can I help you today!"))
        )
        api_base_url_escaped = html.escape(api_base_url)

        # Generate widget HTML/JS code - Fixed XSS vulnerability
        widget_code = f"""
<!-- ZaaKy AI Chatbot Widget -->
<div id="zaaky-chatbot-{chatbot_id_escaped}"></div>
<script>
  (function() {{
    var chatbotConfig = {{
      chatbotId: '{chatbot_id_escaped}',
      apiUrl: '{api_base_url_escaped}/api/public',
      name: '{chatbot_name_escaped}',
      color: '{chatbot_color_escaped}',
      greeting: '{greeting_msg_escaped}'
    }};

    var script = document.createElement('script');
    script.src = chatbotConfig.apiUrl + '/chatbot/' + encodeURIComponent(chatbotConfig.chatbotId) + '/widget.js';
    script.onload = function() {{
      if (window.ZaakyWidget) {{
        ZaakyWidget.init(chatbotConfig);
      }}
    }};
    document.head.appendChild(script);
  }})();
</script>
"""

        return {
            "chatbot_id": chatbot_id,
            "widget_code": widget_code,
            "integration_url": f"{api_base_url}/api/public/chatbot/{chatbot_id}/widget.js",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to generate widget code",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to generate widget code"
        ) from e


@router.get("/chatbot/{chatbot_id}/widget.js")
async def get_chatbot_widget_js(chatbot_id: str):
    """Serve the JavaScript widget file - Enhanced security"""
    try:
        # Get chatbot config for defaults
        response = (
            supabase.table("chatbots")
            .select("*")
            .eq("id", chatbot_id)
            .eq("chain_status", "active")
            .execute()
        )

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=CHATBOT_NOT_FOUND_MESSAGE)

        # Enhanced JavaScript widget code with improved security
        js_code = """
// ZaaKy AI Chatbot Widget v2.1 - Security Enhanced
(function() {
  'use strict';

  window.ZaakyWidget = {
    config: null,
    isInitialized: false,

    init: function(config) {
      if (this.isInitialized) {
        console.warn('ZaaKy Widget already initialized');
        return;
      }

      this.config = this.sanitizeConfig(config);
      this.createWidget();
      this.bindEvents();
      this.isInitialized = true;

      console.log('ZaaKy Widget v2.1 initialized for chatbot:', this.config.chatbotId);
    },

    sanitizeConfig: function(config) {
      return {
        chatbotId: this.escapeHtml(config.chatbotId || ''),
        apiUrl: this.escapeHtml(config.apiUrl || ''),
        name: this.escapeHtml(config.name || 'AI Assistant'),
        color: this.sanitizeColor(config.color || '#6a8fff'),
        greeting: this.escapeHtml(config.greeting || 'Hello! How can I help you today?')
      };
    },

    escapeHtml: function(str) {
      if (!str) return '';
      var div = document.createElement('div');
      div.textContent = str;
      return div.innerHTML;
    },

    sanitizeColor: function(color) {
      // Basic color validation
      if (typeof color !== 'string') return '#6a8fff';
      // Allow hex colors and basic color names
      if (/^#[0-9A-Fa-f]{6}$/.test(color) || /^#[0-9A-Fa-f]{3}$/.test(color)) {
        return color;
      }
      return '#6a8fff'; // Default fallback
    },

    createWidget: function() {
      var container = document.getElementById('zaaky-chatbot-' + this.config.chatbotId);
      if (!container) {
        console.error('ZaaKy Widget container not found');
        return;
      }

      // Create widget UI with properly escaped content
      var widgetHTML = this.buildWidgetHTML();
      container.innerHTML = widgetHTML;
    },

    buildWidgetHTML: function() {
      return `
        <div class="zaaky-widget" style="
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 350px;
          max-height: 500px;
          border-radius: 12px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.15);
          background: white;
          border: 1px solid #e1e5e9;
          z-index: 10000;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
          <div class="zaaky-header" style="
            background: ${this.config.color};
            color: white;
            padding: 16px;
            border-radius: 12px 12px 0 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
          ">
            <h3 style="margin: 0; font-size: 16px;">${this.config.name}</h3>
            <button class="zaaky-close" style="
              background: none;
              border: none;
              color: white;
              font-size: 18px;
              cursor: pointer;
            ">×</button>
          </div>
          <div class="zaaky-messages" style="
            height: 300px;
            overflow-y: auto;
            padding: 16px;
          ">
            <div class="zaaky-message bot" style="
              background: #f8f9fa;
              padding: 12px;
              border-radius: 8px;
              margin-bottom: 12px;
            ">${this.config.greeting}</div>
          </div>
          <div class="zaaky-input-area" style="
            padding: 16px;
            border-top: 1px solid #e1e5e9;
          ">
            <div style="display: flex; gap: 8px;">
              <input type="text" class="zaaky-input" placeholder="Type your message..." style="
                flex: 1;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 6px;
                outline: none;
              ">
              <button class="zaaky-send" style="
                background: ${this.config.color};
                color: white;
                border: none;
                padding: 12px 16px;
                border-radius: 6px;
                cursor: pointer;
              ">Send</button>
            </div>
          </div>
        </div>
      `;
    },

    bindEvents: function() {
      var widget = document.querySelector('.zaaky-widget');
      if (!widget) return;

      var closeBtn = widget.querySelector('.zaaky-close');
      var sendBtn = widget.querySelector('.zaaky-send');
      var input = widget.querySelector('.zaaky-input');

      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          widget.style.display = 'none';
        });
      }

      if (sendBtn && input) {
        var sendMessage = () => this.sendMessage(input.value);
        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
          if (e.key === 'Enter') sendMessage();
        });
      }
    },

    sendMessage: function(message) {
      if (!message.trim()) return;

      var input = document.querySelector('.zaaky-input');
      var messagesContainer = document.querySelector('.zaaky-messages');

      if (!input || !messagesContainer) return;

      // Add user message
      this.addMessage(message, 'user');
      input.value = '';

      // Send to API with proper error handling
      fetch(this.config.apiUrl + '/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          chatbot_id: this.config.chatbotId,
          session_id: this.getSessionId()
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        if (data.response) {
          this.addMessage(data.response, 'bot');
        }
      })
      .catch(error => {
        console.error('ZaaKy API Error:', error);
        this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
      });
    },

    addMessage: function(text, sender) {
      var messagesContainer = document.querySelector('.zaaky-messages');
      if (!messagesContainer) return;

      var messageDiv = document.createElement('div');
      messageDiv.className = 'zaaky-message ' + sender;
      messageDiv.style.cssText = `
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        ${sender === 'user' ?
          'background: ' + this.config.color + '; color: white; margin-left: 40px;' :
          'background: #f8f9fa; margin-right: 40px;'
        }
      `;
      messageDiv.textContent = text; // Use textContent to prevent XSS

      messagesContainer.appendChild(messageDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    },

    getSessionId: function() {
      var sessionId = localStorage.getItem('zaaky-session-' + this.config.chatbotId);
      if (!sessionId) {
        sessionId = 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('zaaky-session-' + this.config.chatbotId, sessionId);
      }
      return sessionId;
    }
  };
})();
"""

        return {"content": js_code, "content_type": "application/javascript"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to serve widget JavaScript",
            extra={"error": str(e), "chatbot_id": chatbot_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to serve widget JavaScript"
        ) from e

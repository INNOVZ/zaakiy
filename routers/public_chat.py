import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ChatService
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client (same as chat.py)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter()


class PublicChatRequest(BaseModel):
    message: str
    chatbot_id: str
    session_id: Optional[str] = None
    user_identifier: Optional[str] = None  # For tracking anonymous users


class PublicChatResponse(BaseModel):
    response: str
    chatbot: dict
    session_id: str


@router.post("/chat", response_model=PublicChatResponse)
async def public_chat(request: PublicChatRequest):
    """Public chat endpoint for embedded chatbots"""
    try:
        # Get chatbot configuration using proper Supabase client
        response = supabase.table("chatbots").select("*").eq(
            "id", request.chatbot_id
        ).eq("chain_status", "active").execute()

        if not response.data or len(response.data) == 0:
            raise HTTPException(
                status_code=404, detail="Chatbot not found or inactive"
            )

        chatbot = response.data[0]

        # Initialize chat service
        chat_service = ChatService()

        # Generate response using organization-specific context
        result = await chat_service.generate_public_response(
            message=request.message,
            org_id=chatbot["org_id"],
            chatbot_config=chatbot,
            session_id=request.session_id
        )

        return {
            "response": result["response"],
            "chatbot": {
                "name": chatbot["name"],
                "avatar_url": chatbot.get("avatar_url"),
                "color_hex": chatbot["color_hex"]
            },
            "session_id": request.session_id or "anonymous"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Public chat failed: {e}")
        raise HTTPException(status_code=500, detail="Chat service unavailable") from e


@router.get("/chatbot/{chatbot_id}/config")
async def get_public_chatbot_config(chatbot_id: str):
    """Get public chatbot configuration for embedding"""
    try:
        # Use proper Supabase client
        response = supabase.table("chatbots").select("*").eq(
            "id", chatbot_id
        ).eq("chain_status", "active").execute()

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        chatbot = response.data[0]

        # Return only public-safe information
        return {
            "id": chatbot["id"],
            "name": chatbot["name"],
            "avatar_url": chatbot.get("avatar_url"),
            "color_hex": chatbot["color_hex"],
            "description": chatbot.get("description", ""),
            "greeting_message": chatbot.get("greeting_message", "Hello! How can I help you today?"),
            "status": "active"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Get chatbot config failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get chatbot configuration"
        ) from e


@router.get("/chatbot/{chatbot_id}/widget")
async def get_chatbot_widget_code(chatbot_id: str):
    """Get embeddable widget code for a chatbot"""
    try:
        # Get chatbot config
        response = supabase.table("chatbots").select("*").eq(
            "id", chatbot_id
        ).eq("chain_status", "active").execute()

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        chatbot = response.data[0]

        # Generate widget HTML/JS code
        widget_code = f"""
<!-- ZaaKy AI Chatbot Widget -->
<div id="zaaky-chatbot-{chatbot_id}"></div>
<script>
  (function() {{
    var chatbotConfig = {{
      chatbotId: '{chatbot_id}',
      apiUrl: '{os.getenv("API_BASE_URL", "http://localhost:8001")}/api/public',
      name: '{chatbot["name"]}',
      color: '{chatbot["color_hex"]}',
      greeting: '{chatbot.get("greeting_message", "Hello! How can I help you today?")}'
    }};
    
    var script = document.createElement('script');
    script.src = chatbotConfig.apiUrl + '/widget.js';
    script.onload = function() {{
      ZaakyWidget.init(chatbotConfig);
    }};
    document.head.appendChild(script);
  }})();
</script>
"""

        return {
            "chatbot_id": chatbot_id,
            "widget_code": widget_code,
            "integration_url": f"{os.getenv('API_BASE_URL', 'http://localhost:8001')}/api/public/chatbot/{chatbot_id}/widget.js"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Get widget code failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate widget code"
        ) from e


@router.get("/chatbot/{chatbot_id}/widget.js")
async def get_chatbot_widget_js(chatbot_id: str):
    """Serve the JavaScript widget file"""
    try:
        # Get chatbot config for defaults
        response = supabase.table("chatbots").select("*").eq(
            "id", chatbot_id
        ).eq("chain_status", "active").execute()

        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Chatbot not found")

        # Return JavaScript widget code
        js_code = """
// ZaaKy AI Chatbot Widget v1.0
(function() {
  'use strict';
  
  window.ZaakyWidget = {
    init: function(config) {
      this.config = config;
      this.createWidget();
      this.bindEvents();
    },
    
    createWidget: function() {
      // Widget creation logic here
      console.log('ZaaKy Widget initialized for chatbot:', this.config.chatbotId);
    },
    
    bindEvents: function() {
      // Event binding logic here
    }
  };
})();
"""

        return {
            "content": js_code,
            "content_type": "application/javascript"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Error] Get widget JS failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to serve widget JavaScript"
        ) from e

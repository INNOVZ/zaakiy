import os
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.chat_service import ChatService
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    chatbot_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    chatbot_config: dict


class CreateChatbotRequest(BaseModel):
    name: str
    description: Optional[str] = None
    color_hex: Optional[str] = "#3B82F6"
    tone: Optional[str] = "helpful"
    behavior: Optional[str] = "Be helpful and informative"
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = "Hello! How can I help you today?"
    fallback_message: Optional[str] = "I'm sorry, I don't have information about that."
    ai_model_config: Optional[dict] = None
    is_active: Optional[bool] = True


class UpdateChatbotRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color_hex: Optional[str] = None
    tone: Optional[str] = None
    behavior: Optional[str] = None
    system_prompt: Optional[str] = None
    greeting_message: Optional[str] = None
    fallback_message: Optional[str] = None
    ai_model_config: Optional[dict] = None
    is_active: Optional[bool] = None


@router.post("/chatbots")
async def create_chatbot(
    request: CreateChatbotRequest,
    user=Depends(verify_jwt_token)
):
    """Create a new chatbot for the organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Generate chatbot ID
        chatbot_id = str(uuid.uuid4())

        # Default model config
        ai_model_config = request.ai_model_config or {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Build system prompt
        system_prompt = request.system_prompt or f"""
You are {request.name}, a {request.tone} AI assistant for this business.

Personality: {request.tone}
Behavior: {request.behavior}

Instructions:
- You are knowledgeable about this business and its offerings
- Answer questions naturally using your knowledge of our products and services
- NEVER mention documents, training data, or uploaded files
- Respond as if this information is simply what you know
- Be conversational and helpful
- Maintain a {request.tone} tone throughout conversations
- If you don't know something specific, say so politely
"""

        # Create chatbot data for database
        chatbot_data = {
            "id": chatbot_id,
            "org_id": org_id,
            "name": request.name,
            "description": request.description,
            "color_hex": request.color_hex,
            "tone": request.tone,
            "behavior": request.behavior,
            "system_prompt": system_prompt,
            "greeting_message": request.greeting_message,
            "fallback_message": request.fallback_message,
            "model_config": ai_model_config,
            "chain_status": "active" if request.is_active else "inactive",
            "avatar_url": None,
            "trained_at": None
        }

        # Insert into Supabase
        response = supabase.table("chatbots").insert(chatbot_data).execute()

        if response.data:
            created_chatbot = response.data[0]
            return {
                "success": True,
                "chatbot": created_chatbot,
                "id": chatbot_id,
                "message": "Chatbot created successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {response.error if hasattr(response, 'error') else 'Unknown error'}"
            )

    except Exception as e:
        print(f"[Error] Create chatbot failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create chatbot: {str(e)}"
        ) from e


@router.put("/chatbots/{chatbot_id}")
async def update_chatbot(
    chatbot_id: str,
    request: UpdateChatbotRequest,
    user=Depends(verify_jwt_token)
):
    """Update an existing chatbot"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Build update data (only include fields that were provided)
        update_data = {}
        for field, value in request.dict(exclude_unset=True).items():
            if value is not None:
                if field == "ai_model_config":
                    update_data["model_config"] = value
                elif field == "is_active":
                    update_data["chain_status"] = "active" if value else "inactive"
                else:
                    update_data[field] = value

        if not update_data:
            raise HTTPException(
                status_code=400, detail="No update data provided")

        # Update chatbot
        response = supabase.table("chatbots").update(update_data).eq(
            "id", chatbot_id).eq("org_id", org_id).execute()

        if response.data:
            return {
                "success": True,
                "chatbot": response.data[0],
                "message": "Chatbot updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except Exception as e:
        print(f"[Error] Update chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update chatbot: {str(e)}"
        ) from e


@router.delete("/chatbots/{chatbot_id}")
async def delete_chatbot(
    chatbot_id: str,
    user=Depends(verify_jwt_token)
):
    """Delete a chatbot"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = supabase.table("chatbots").delete().eq(
            "id", chatbot_id).eq("org_id", org_id).execute()

        if response.data:
            return {
                "success": True,
                "message": "Chatbot deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except Exception as e:
        print(f"[Error] Delete chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete chatbot: {str(e)}"
        ) from e


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token)):
    """List organization's chatbots"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = supabase.table("chatbots").select(
            "*").eq("org_id", org_id).order("created_at", desc=True).execute()

        return {"chatbots": response.data or []}

    except Exception as e:
        print(f"[Error] List chatbots failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch chatbots") from e


@router.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: str, user=Depends(verify_jwt_token)):
    """Get specific chatbot configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = supabase.table("chatbots").select(
            "*").eq("id", chatbot_id).eq("org_id", org_id).execute()

        if response.data and len(response.data) > 0:
            chatbot = response.data[0]
            # Map model_config back to ai_model_config for frontend compatibility
            if "model_config" in chatbot:
                chatbot["ai_model_config"] = chatbot["model_config"]
            return chatbot
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except Exception as e:
        print(f"[Error] Get chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to fetch chatbot") from e


@router.post("/chatbots/{chatbot_id}/activate")
async def activate_chatbot(
    chatbot_id: str,
    user=Depends(verify_jwt_token)
):
    """Activate a chatbot for deployment"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Deactivate all other chatbots first (only one active at a time)
        supabase.table("chatbots").update(
            {"chain_status": "inactive"}).eq("org_id", org_id).execute()

        # Activate the selected chatbot
        response = supabase.table("chatbots").update({"chain_status": "active"}).eq(
            "id", chatbot_id).eq("org_id", org_id).execute()

        if response.data:
            return {
                "success": True,
                "message": "Chatbot activated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except Exception as e:
        print(f"[Error] Activate chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to activate chatbot: {str(e)}"
        ) from e


@router.post("/conversation")
async def chat_conversation(
    request: ChatRequest,
    user=Depends(verify_jwt_token)
):
    """Enhanced chat with document context"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get chatbot config
        chatbot_config = await get_org_chatbot(org_id, request.chatbot_id)

        # Initialize chat service
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)

        # Generate response
        result = await chat_service.generate_response(
            message=request.message,
            conversation_id=request.conversation_id or "default"
        )

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            chatbot_config={
                "name": chatbot_config.get("name", "AI Assistant"),
                "avatar_url": chatbot_config.get("avatar_url"),
                "color_hex": chatbot_config.get("color_hex", "#3B82F6"),
                "tone": chatbot_config.get("tone", "helpful")
            }
        )

    except Exception as e:
        print(f"[Error] Chat conversation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Chat failed: {str(e)}") from e


async def get_org_chatbot(org_id: str, chatbot_id: Optional[str] = None):
    """Get organization's active chatbot with fallback for testing"""
    try:
        if chatbot_id:
            response = supabase.table("chatbots").select(
                "*").eq("id", chatbot_id).eq("org_id", org_id).execute()
        else:
            response = supabase.table("chatbots").select("*").eq("org_id", org_id).eq(
                "chain_status", "active").order("created_at", desc=True).limit(1).execute()

        if response.data and len(response.data) > 0:
            return response.data[0]

        # Return default chatbot for testing
        return {
            "id": "default-chatbot",
            "name": "ZaaKy Assistant",
            "avatar_url": None,
            "color_hex": "#3B82F6",
            "tone": "helpful",
            "behavior": "Be helpful and informative",
            "chain_status": "active",
            "org_id": org_id
        }

    except Exception as e:
        print(f"[ERROR] Error getting chatbot: {str(e)}")
        return {
            "name": "ZaaKy Assistant",
            "color_hex": "#3B82F6",
            "tone": "helpful"
        }

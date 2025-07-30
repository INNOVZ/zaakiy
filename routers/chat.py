import os
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.chat_service import ChatService
from services.supabase_client import client

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    chatbot_id: Optional[str] = None
    conversation_id: Optional[str] = "sandbox"


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    chatbot_config: dict = {}


@router.post("/conversation")
async def chat_conversation(
    request: ChatRequest,
    user=Depends(verify_jwt_token)
):
    """Handle chat conversation with organization's trained chatbot"""
    try:
        # Get user's organization
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get active chatbot for organization
        chatbot = await get_org_chatbot(org_id, request.chatbot_id)

        if not chatbot:
            raise HTTPException(
                status_code=404, detail="No active chatbot found")

        # Initialize chat service with organization context
        chat_service = ChatService(
            org_id=org_id,
            chatbot_config=chatbot
        )

        # Generate AI response
        response_data = await chat_service.generate_response(
            message=request.message,
            conversation_id=request.conversation_id
        )

        return ChatResponse(
            response=response_data["response"],
            sources=response_data.get("sources", []),
            chatbot_config={
                "name": chatbot.get("name", "AI Assistant"),
                "avatar_url": chatbot.get("avatar_url"),
                "color_hex": chatbot.get("color_hex", "#3B82F6"),
                "tone": chatbot.get("tone", "helpful")
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token)):
    """List organization's chatbots"""
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]

    response = await client.get(
        "/chatbots",
        params={
            "select": "*",
            "org_id": f"eq.{org_id}",
            "order": "created_at.desc"
        }
    )

    if response.status_code == 200:
        return {"chatbots": response.json()}
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch chatbots")


@router.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: str, user=Depends(verify_jwt_token)):
    """Get specific chatbot configuration"""
    user_data = await get_user_with_org(user["user_id"])
    org_id = user_data["org_id"]

    response = await client.get(
        "/chatbots",
        params={
            "select": "*",
            "id": f"eq.{chatbot_id}",
            "org_id": f"eq.{org_id}"
        }
    )

    if response.status_code == 200:
        data = response.json()
        if data:
            return data[0]
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch chatbot")


async def get_org_chatbot(org_id: str, chatbot_id: Optional[str] = None):
    """Get organization's active chatbot"""
    if chatbot_id:
        # Get specific chatbot
        response = await client.get(
            "/chatbots",
            params={
                "select": "*",
                "id": f"eq.{chatbot_id}",
                "org_id": f"eq.{org_id}"
            }
        )
    else:
        # Get first active chatbot
        response = await client.get(
            "/chatbots",
            params={
                "select": "*",
                "org_id": f"eq.{org_id}",
                "chain_status": f"eq.active",
                "order": "created_at.desc",
                "limit": "1"
            }
        )

    if response.status_code == 200:
        data = response.json()
        return data[0] if data else None
    return None

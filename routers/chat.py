"""
This module handles chat requests and responses.

Contains functions for creating and managing chat, chat conversation, performing analytics, and configuring context.
"""
import os
import uuid
import traceback
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from services.supabase_auth import verify_jwt_token
from services.user_service import get_user_with_org
from services.chat_service import ChatService
from services.context_analytics import context_analytics
from services.context_config import context_config_manager
from utils.error_handlers import ErrorHandler, handle_errors
from utils.exceptions import ValidationError, DatabaseError

load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

router = APIRouter()


class ChatRequest(BaseModel):
    """
This module handles chat basemodel.

Contains functions for creating and managing chat requests.
"""
    message: str
    chatbot_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
This module handles chat response.

Contains functions for generating and formatting chat responses.
"""
    response: str
    sources: List[str] = []
    chatbot_config: dict
    conversation_id: str
    message_id: Optional[str] = None
    processing_time_ms: int = 0
    context_quality: dict = {}


class CreateChatbotRequest(BaseModel):
    """
This module handles chatbot configuration.

Contains functions for creating and managing chatbots.
"""
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
    """
This module handles chatbot updates.

Contains functions for updating chatbot configurations and settings.
"""
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


class FeedbackRequest(BaseModel):
    """
This module handles user authentication and session management.

Contains functions for login, logout, and token validation.
"""
    message_id: str
    rating: int  # 1 for thumbs up, -1 for thumbs down
    feedback_text: Optional[str] = None


class ContextConfigRequest(BaseModel):
    """
This module handles context configuration.

Contains Class for context config.
"""
    config_updates: dict

# ==========================================
# HELPER FUNCTIONS
# ==========================================


async def get_org_chatbot(org_id: str, chatbot_id: Optional[str] = None):
    """Get organization's active chatbot with enhanced fallback"""
    try:
        if chatbot_id:
            response = supabase.table("chatbots").select("*").eq(
                "id", chatbot_id
            ).eq("org_id", org_id).execute()
        else:
            response = supabase.table("chatbots").select("*").eq(
                "org_id", org_id
            ).eq("chain_status", "active").order(
                "created_at", desc=True
            ).limit(1).execute()

        if response.data and len(response.data) > 0:
            chatbot = response.data[0]
            print(
                f"[SUCCESS] Found chatbot: {chatbot['name']} for org {org_id}")
            return chatbot

        # Enhanced default chatbot
        print(
            f"[INFO] No chatbot found for org {org_id}, using enhanced default")
        return {
            "id": f"default-{org_id}",
            "name": "INNOVZ Assistant",
            "avatar_url": None,
            "color_hex": "#3B82F6",
            "tone": "helpful",
            "behavior": "Be helpful, professional, and informative. Use the knowledge base to provide accurate answers.",
            "chain_status": "active",
            "org_id": org_id,
            "greeting_message": "Hello! I'm your INNOVZ AI Assistant. How can I help you today?",
            "fallback_message": "I apologize, but I don't have enough information to answer that question accurately. Could you please rephrase or provide more context?",
            "system_prompt": "You are INNOVZ Assistant, a helpful AI powered by the organization's knowledge base.",
            "model_config": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }

    except (ConnectionError, TimeoutError) as e:
        print(f"[ERROR] Database connection error for org {org_id}: {str(e)}")
        return {
            "id": f"fallback-{org_id}",
            "name": "INNOVZ Assistant",
            "color_hex": "#3B82F6",
            "tone": "helpful",
            "behavior": "Be helpful and informative",
            "org_id": org_id,
            "greeting_message": "Hello! How can I help you today?",
            "fallback_message": "I'm experiencing some technical difficulties. Please try again."
        }
   
# ==========================================
# CHATBOT MANAGEMENT ENDPOINTS
# ==========================================


@router.post("/chatbots")
@handle_errors(context="create_chatbot")
async def create_chatbot(
    request: CreateChatbotRequest,
    user=Depends(verify_jwt_token)
):
    """Create a new chatbot for the organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

         # Validate request data
        if not request.name or len(request.name.strip()) < 2:
            raise ValidationError(
                "Chatbot name must be at least 2 characters long")


        # Generate chatbot ID
        chatbot_id = str(uuid.uuid4())

        # Default model config
        ai_model_config = request.ai_model_config or {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # Build enhanced system prompt
        system_prompt = request.system_prompt or f"""
You are {request.name}, a {request.tone} AI assistant for INNOVZ.

PERSONALITY AND BEHAVIOR:
- Tone: {request.tone}
- Behavior: {request.behavior}

CORE INSTRUCTIONS:
- Use uploaded documents and knowledge base to answer questions accurately
- Be conversational and maintain a {request.tone} tone
- If you don't have relevant information, acknowledge this honestly
- Provide specific, actionable answers when possible
- Reference sources when appropriate but don't overwhelm users

GREETING: {request.greeting_message}
FALLBACK: {request.fallback_message}
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
            "trained_at": datetime.utcnow().isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        # Insert into Supabase
        response = supabase.table("chatbots").insert(chatbot_data).execute()

        if response.data:
            created_chatbot = response.data[0]
            print(f"[SUCCESS] Chatbot created: {chatbot_id} for org {org_id}")

            # Initialize default context config for this org if it doesn't exist
            try:
                await context_config_manager.get_config(org_id)
            except (KeyError, ValueError, ConnectionError) as config_error:
                print(
                    f"[WARNING] Could not initialize context config: {config_error}")

            return {
                "success": True,
                "chatbot": created_chatbot,
                "id": chatbot_id,
                "message": "Chatbot created successfully"
            }
        else:
            error_msg = getattr(response, 'error', 'Unknown database error')
            print(f"[ERROR] Database insert failed: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Database error: {error_msg}"
            )

    except ValidationError:
        raise  # Re-raise validation errors
    except ConnectionError as e:
        ErrorHandler.log_and_raise(
            DatabaseError,
            "Failed to create chatbot due to database connection issue",
            context="create_chatbot",
            original_exception=e
        )


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
        update_data = {"updated_at": datetime.utcnow().isoformat()}

        for field, value in request.dict(exclude_unset=True).items():
            if value is not None:
                if field == "ai_model_config":
                    update_data["model_config"] = value
                elif field == "is_active":
                    update_data["chain_status"] = "active" if value else "inactive"
                else:
                    update_data[field] = value

        if len(update_data) == 1:  # Only timestamp was added
            raise HTTPException(
                status_code=400, detail="No update data provided")

        # Update chatbot
        response = supabase.table("chatbots").update(update_data).eq(
            "id", chatbot_id
        ).eq("org_id", org_id).execute()

        if response.data:
            print(f"[SUCCESS] Chatbot updated: {chatbot_id}")
            return {
                "success": True,
                "chatbot": response.data[0],
                "message": "Chatbot updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Update chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update chatbot: {str(e)}") from e


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
            "id", chatbot_id
        ).eq("org_id", org_id).execute()

        if response.data:
            print(f"[SUCCESS] Chatbot deleted: {chatbot_id}")
            return {
                "success": True,
                "message": "Chatbot deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Delete chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete chatbot: {str(e)}") from e


@router.get("/chatbots")
async def list_chatbots(user=Depends(verify_jwt_token)):
    """List organization's chatbots"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = supabase.table("chatbots").select("*").eq(
            "org_id", org_id
        ).order("created_at", desc=True).execute()

        chatbots = response.data or []

        # Add compatibility mapping for frontend
        for chatbot in chatbots:
            if "model_config" in chatbot:
                chatbot["ai_model_config"] = chatbot["model_config"]

        print(f"[SUCCESS] Listed {len(chatbots)} chatbots for org {org_id}")
        return {"chatbots": chatbots}

    except Exception as e:
        print(f"[ERROR] List chatbots failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chatbots") from e


@router.get("/chatbots/{chatbot_id}")
async def get_chatbot(chatbot_id: str, user=Depends(verify_jwt_token)):
    """Get specific chatbot configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        response = supabase.table("chatbots").select("*").eq(
            "id", chatbot_id
        ).eq("org_id", org_id).execute()

        if response.data and len(response.data) > 0:
            chatbot = response.data[0]
            # Map model_config back to ai_model_config for frontend compatibility
            if "model_config" in chatbot:
                chatbot["ai_model_config"] = chatbot["model_config"]

            print(f"[SUCCESS] Retrieved chatbot: {chatbot_id}")
            return chatbot
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Get chatbot failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chatbot") from e


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
        supabase.table("chatbots").update({
            "chain_status": "inactive",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("org_id", org_id).execute()

        # Activate the selected chatbot
        response = supabase.table("chatbots").update({
            "chain_status": "active",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", chatbot_id).eq("org_id", org_id).execute()

        if response.data:
            print(f"[SUCCESS] Chatbot activated: {chatbot_id}")
            return {
                "success": True,
                "message": "Chatbot activated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Chatbot not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Activate chatbot failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to activate chatbot: {str(e)}") from e

# ==========================================
# CHAT AND CONVERSATION ENDPOINTS
# ==========================================


@router.post("/conversation")
async def chat_conversation(
    request: ChatRequest,
    user=Depends(verify_jwt_token)
):
    """Main chat endpoint with enhanced context engineering"""
    start_time = datetime.utcnow()

    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        print(
            f"[INFO] Chat request from org {org_id}: {request.message[:50]}...")

        # Get chatbot config
        chatbot_config = await get_org_chatbot(org_id, request.chatbot_id)

        # Generate session ID if not provided
        session_id = request.conversation_id or f"session-{user['user_id']}-{int(datetime.utcnow().timestamp())}"

        # Initialize chat service with full context engineering
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)

        # Generate response
        result = await chat_service.chat(
            message=request.message,
            session_id=session_id,
            chatbot_id=request.chatbot_id or chatbot_config.get("id")
        )

        total_time = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000)

        print(f"[SUCCESS] Chat completed in {total_time}ms for org {org_id}")

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            conversation_id=result["conversation_id"],
            message_id=result.get("message_id"),
            processing_time_ms=result.get("processing_time_ms", total_time),
            context_quality=result.get("context_quality", {}),
            chatbot_config={
                "name": chatbot_config.get("name", "AI Assistant"),
                "avatar_url": chatbot_config.get("avatar_url"),
                "color_hex": chatbot_config.get("color_hex", "#3B82F6"),
                "tone": chatbot_config.get("tone", "helpful"),
                "id": chatbot_config.get("id")
            }
        )

    except Exception as e:
        error_time = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000)
        print(f"[ERROR] Chat conversation failed after {error_time}ms: {e}")
      
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}") from e


@router.get("/conversations")
async def list_conversations(
    limit: int = 20,
    user=Depends(verify_jwt_token)
):
    """List recent conversations with enhanced metadata"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        # Get dummy config for service initialization
        chatbot_config = {"name": "Assistant", "org_id": org_id}
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)

        conversations = await chat_service.get_recent_conversations(limit=limit)

        print(
            f"[SUCCESS] Listed {len(conversations)} conversations for org {org_id}")
        return {"conversations": conversations}

    except Exception as e:
        print(f"[ERROR] List conversations failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load conversations: {str(e)}") from e


@router.post("/feedback")
async def add_feedback(
    request: FeedbackRequest,
    user=Depends(verify_jwt_token)
):
    """Add message feedback with analytics integration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        chatbot_config = {"name": "Assistant", "org_id": org_id}
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)

        success = await chat_service.add_feedback(
            message_id=request.message_id,
            rating=request.rating,
            feedback_text=request.feedback_text
        )

        if success:
            print(f"[SUCCESS] Feedback added for message {request.message_id}")
            return {"success": True, "message": "Feedback recorded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Message not found")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Add feedback failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to add feedback: {str(e)}") from e

# ==========================================
# ANALYTICS AND CONFIGURATION ENDPOINTS
# ==========================================


@router.get("/analytics/dashboard")
async def get_analytics_dashboard(
    days: int = 7,
    user=Depends(verify_jwt_token)
):
    """Get analytics dashboard for the organization"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        dashboard = await context_analytics.get_performance_dashboard(org_id, days)

        print(f"[SUCCESS] Analytics dashboard retrieved for org {org_id}")
        return dashboard

    except Exception as e:
        print(f"[ERROR] Get analytics dashboard failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics: {str(e)}") from e


@router.get("/context-config")
async def get_context_config(user=Depends(verify_jwt_token)):
    """Get current context engineering configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        config = await context_config_manager.get_config(org_id)

        return {
            "success": True,
            "config": config.dict(),
            "message": "Context configuration retrieved successfully"
        }

    except Exception as e:
        print(f"[ERROR] Get context config failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get context configuration: {str(e)}") from e


@router.put("/context-config")
async def update_context_config(
    request: ContextConfigRequest,
    user=Depends(verify_jwt_token)
):
    """Update context engineering configuration"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        success = await context_config_manager.update_config(org_id, request.config_updates)

        if success:
            return {
                "success": True,
                "message": "Context configuration updated successfully"
            }
        else:
            raise HTTPException(
                status_code=500, detail="Failed to update configuration")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Update context config failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update context configuration: {str(e)}") from e


@router.get("/performance-insights")
async def get_performance_insights(user=Depends(verify_jwt_token)):
    """Get AI performance insights and recommendations"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]

        chatbot_config = {"name": "Assistant", "org_id": org_id}
        chat_service = ChatService(
            org_id=org_id, chatbot_config=chatbot_config)

        insights = await chat_service.get_performance_insights()

        return {
            "success": True,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }

    except (ConnectionError, TimeoutError) as e:
        print(f"[ERROR] Database connection error in performance insights: {e}")
        raise HTTPException(
            status_code=503, detail="Service temporarily unavailable") from e
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Data validation error in performance insights: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid data format") from e
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Get performance insights failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get performance insights: {str(e)}") from e

@router.get("/analytics/context")
async def get_context_analytics(
    days: int = 7,
    user=Depends(verify_jwt_token)
):
    """Get context engineering analytics"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]
        
        # Use the existing context_analytics instance
        analytics_data = await context_analytics.get_query_analysis(org_id, "", days)
        
        return {
            "success": True,
            "analytics": analytics_data,
            "summary": {
                "total_queries": analytics_data.get("similar_queries_found", 0),
                "avg_quality_score": 0.7,  # Default values
                "avg_retrieval_time": 1500,
                "model_distribution": {"gpt-4": 100}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.post("/analytics/query-optimization")
async def analyze_query_optimization(
    request: dict,
    user=Depends(verify_jwt_token)
):
    """Analyze query for optimization suggestions"""
    try:
        user_data = await get_user_with_org(user["user_id"])
        org_id = user_data["org_id"]
        
        query = request.get("query", "")
        
        # Use existing context analytics
        analysis = await context_analytics.get_query_analysis(org_id, query, 30)
        
        return {
            "original_query": query,
            "enhanced_query": f"Enhanced: {query}",  # Simple enhancement
            "expected_quality": 0.8,
            "strategy_recommendation": "hybrid",
            "estimated_response_time": 2000
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
# ==========================================
# HEALTH CHECK
# ==========================================


@router.get("/health")
async def health_check():
    """Health check endpoint for chat services"""
    try:
        # Test database connection
        test_response = supabase.table(
            "organizations").select("id").limit(1).execute()
        db_status = "healthy" if test_response.data is not None else "error"

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": db_status,
                "chat_service": "healthy",
                "context_engine": "healthy"
            },
            "version": "2.0.0"
        }

    except (ConnectionError, TimeoutError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Database connection error: {str(e)}",
            "version": "2.0.0"
        }
    except (ValueError, KeyError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Data validation error: {str(e)}",
            "version": "2.0.0"
        }
    except (AttributeError, ImportError) as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Service configuration error: {str(e)}",
            "version": "2.0.0"
        }

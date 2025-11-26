"""
MCP Server Router - FastAPI endpoints for MCP functionality

Exposes MCP tools as REST API endpoints for integration with:
- LLM clients (Claude, GPT-4, etc.)
- Frontend applications
- Other backend services
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...utils.logging_config import get_logger
from ..agents.rag_agent import get_rag_system
from ..auth.middleware import get_current_user
from ..mcp_server import get_mcp_server

router = APIRouter(prefix="/api/mcp", tags=["MCP"])
logger = get_logger(__name__)


# Request/Response Models
class ToolCallRequest(BaseModel):
    """Request to call an MCP tool"""

    tool_name: str
    parameters: Dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    """Response from tool call"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str


class RAGQueryRequest(BaseModel):
    """Request for agentic RAG query"""

    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""

    response: str
    sources: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    success: bool
    error: Optional[str] = None


class ToolSchema(BaseModel):
    """Tool schema information"""

    name: str
    description: str
    input_schema: Dict[str, Any]


# Dependencies
async def get_org_id(current_user: dict = Depends(get_current_user)) -> str:
    """Extract org_id from current user"""
    org_id = current_user.get("org_id")
    if not org_id:
        raise HTTPException(status_code=401, detail="org_id not found in user context")
    return org_id


# Endpoints
@router.get("/tools")
async def list_tools(org_id: str = Depends(get_org_id)) -> Dict[str, Any]:
    """
    List all available MCP tools

    Returns:
        List of tools with descriptions and input schemas
    """
    try:
        mcp_server = get_mcp_server(org_id)
        tools_schema = mcp_server.get_tools_schema()

        tools_list = [
            ToolSchema(
                name=name,
                description=schema["description"],
                input_schema=schema["inputSchema"],
            )
            for name, schema in tools_schema.items()
        ]

        logger.info(f"Listed {len(tools_list)} tools for org {org_id}")

        return {
            "success": True,
            "tools": [tool.dict() for tool in tools_list],
            "total": len(tools_list),
        }

    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_name}/call")
async def call_tool(
    tool_name: str,
    request: ToolCallRequest,
    org_id: str = Depends(get_org_id),
) -> ToolCallResponse:
    """
    Call a specific MCP tool

    Args:
        tool_name: Name of the tool to call
        request: Tool parameters

    Returns:
        Tool execution result
    """
    try:
        logger.info(f"Calling tool {tool_name} for org {org_id}")

        mcp_server = get_mcp_server(org_id)
        result = await mcp_server.call_tool(tool_name, **request.parameters)

        return ToolCallResponse(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error"),
            timestamp=result.get("timestamp"),
        )

    except KeyError:
        logger.error(f"Tool not found: {tool_name}")
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/{tool_name}")
async def get_tool_info(
    tool_name: str,
    org_id: str = Depends(get_org_id),
) -> Dict[str, Any]:
    """
    Get detailed information about a specific tool

    Args:
        tool_name: Name of the tool

    Returns:
        Tool description, input schema, and usage examples
    """
    try:
        mcp_server = get_mcp_server(org_id)
        tools_schema = mcp_server.get_tools_schema()

        if tool_name not in tools_schema:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

        schema = tools_schema[tool_name]

        return {
            "name": tool_name,
            "description": schema["description"],
            "input_schema": schema["inputSchema"],
            "examples": get_tool_examples(tool_name),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def rag_query(
    request: RAGQueryRequest,
    org_id: str = Depends(get_org_id),
) -> RAGQueryResponse:
    """
    Process query with agentic RAG system

    Combines MCP tools with document retrieval for intelligent responses

    Args:
        request: Query and optional conversation history

    Returns:
        Response with sources and intermediate tool calls
    """
    try:
        logger.info(f"Processing RAG query for org {org_id}: {request.query[:50]}")

        rag_system = get_rag_system(org_id)
        result = await rag_system.process_query(
            request.query, request.conversation_history
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("response", "Query processing failed"),
            )

        return RAGQueryResponse(
            response=result["response"],
            sources=result.get("sources", []),
            tool_calls=result.get("tool_calls", []),
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/batch")
async def batch_call_tools(
    requests: List[ToolCallRequest],
    org_id: str = Depends(get_org_id),
) -> Dict[str, Any]:
    """
    Call multiple tools in parallel

    Useful for fetching data from multiple sources simultaneously

    Args:
        requests: List of tool call requests

    Returns:
        Results for all tool calls
    """
    try:
        import asyncio

        logger.info(f"Batch calling {len(requests)} tools for org {org_id}")

        mcp_server = get_mcp_server(org_id)

        # Execute all tool calls concurrently
        tasks = [
            mcp_server.call_tool(req.tool_name, **req.parameters) for req in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "tool_name": requests[i].tool_name,
                        "success": False,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(
                    {
                        "tool_name": requests[i].tool_name,
                        **result,
                    }
                )

        return {
            "success": True,
            "total": len(requests),
            "results": processed_results,
        }

    except Exception as e:
        logger.error(f"Error in batch tool call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper function for tool examples
def get_tool_examples(tool_name: str) -> List[Dict[str, Any]]:
    """Get example usage for a tool"""

    examples = {
        "scrape_url": [
            {
                "description": "Scrape a website",
                "parameters": {
                    "url": "https://example.com",
                    "timeout_ms": 30000,
                },
            }
        ],
        "scrape_with_products": [
            {
                "description": "Scrape e-commerce products",
                "parameters": {"url": "https://example-store.myshopify.com"},
            }
        ],
        "get_calendar_events": [
            {
                "description": "Get next 7 days of calendar events",
                "parameters": {
                    "days_ahead": 7,
                    "calendar_id": "primary",
                },
            }
        ],
        "read_google_sheet": [
            {
                "description": "Read from a Google Sheet",
                "parameters": {
                    "spreadsheet_id": "1BxiMVs0XRA5nFMKUVfIc45PsAPw0izJsnVgIFnRy6WE",
                    "range_name": "Sheet1!A1:Z1000",
                },
            }
        ],
        "get_shopify_products": [
            {
                "description": "Get all active products from Shopify",
                "parameters": {
                    "limit": 50,
                    "status": "active",
                },
            }
        ],
        "get_crm_contacts": [
            {
                "description": "Get contacts from CRM",
                "parameters": {
                    "limit": 50,
                    "search_query": None,
                },
            }
        ],
        "search_documents": [
            {
                "description": "Search company documents",
                "parameters": {
                    "query": "product inventory",
                    "limit": 5,
                },
            }
        ],
    }

    return examples.get(tool_name, [])


@router.get("/health")
async def health_check(org_id: str = Depends(get_org_id)) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        mcp_server = get_mcp_server(org_id)
        tools_count = len(mcp_server.get_all_tools())

        return {
            "status": "healthy",
            "mcp_server": "active",
            "tools_available": tools_count,
            "org_id": org_id,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "org_id": org_id,
        }

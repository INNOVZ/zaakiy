"""
Agentic RAG System for Zaakiy

Combines LLM agents with MCP tools and retrieval-augmented generation
to provide intelligent, context-aware responses with access to:
- Real-time web scraping
- Company documents
- Business system data (Shopify, CRM, Google services)
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool, tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..utils.logging_config import get_logger
from .mcp_server import ZaakiyMCPServer

logger = get_logger(__name__)


class AgenticRAGSystem:
    """
    Intelligent agent that orchestrates MCP tools with RAG capabilities
    """

    def __init__(
        self,
        org_id: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.2,
        max_iterations: int = 10,
    ):
        self.org_id = org_id
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations

        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.mcp_server = ZaakiyMCPServer(org_id)
        self._tools = self._build_tools()
        self.agent_executor = self._create_agent()

        logger.info(f"Initialized Agentic RAG for org {org_id}")

    def _build_tools(self) -> List[Tool]:
        """Build LangChain tools from MCP server"""

        # Get MCP tools
        mcp_tools = self.mcp_server.get_all_tools()

        tools = []

        # Wrap each MCP tool as a LangChain tool
        for tool_name, mcp_tool in mcp_tools.items():

            def make_handler(tool_name):
                async def handler(**kwargs):
                    result = await self.mcp_server.call_tool(tool_name, **kwargs)
                    return json.dumps(result)

                return handler

            # Build input schema description
            input_desc = ", ".join(
                f"{k}: {v}" for k, v in mcp_tool.input_schema.items()
            )

            def make_func(tn):
                def func(**kwargs):
                    return asyncio.run(self.mcp_server.call_tool(tn, **kwargs))

                return func

            langchain_tool = Tool(
                name=tool_name,
                func=make_func(tool_name),
                description=f"{mcp_tool.description}. Inputs: {input_desc}",
                return_direct=False,
            )

            tools.append(langchain_tool)
            logger.debug(f"Registered tool: {tool_name}")

        return tools

    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with system prompt"""

        system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an intelligent business assistant for {org_id} with access to powerful tools.

Your capabilities:
1. **Live Web Scraping**: Extract content from any website, including JavaScript-rendered pages
2. **E-Commerce Data**: Access Shopify products, orders, and inventory in real-time
3. **Google Services**: Read/write Google Calendar and Sheets, manage schedules
4. **CRM Integration**: Access contacts, leads, and interaction history
5. **Document Search**: Query company documents and knowledge base with semantic search
6. **Data Synthesis**: Combine information from multiple sources to answer complex questions

Guidelines:
- Always search relevant documents first for context
- Use web scraping to get real-time data when needed
- Provide source attribution for information
- If data is not available, be transparent about limitations
- For multi-step tasks, show progress and reasoning
- Ask clarifying questions if the user request is ambiguous

When answering questions:
1. Break down complex queries into steps
2. Gather data from relevant sources
3. Retrieve context from documents
4. Synthesize into clear, actionable responses
5. Always cite sources and timestamps""",
                ),
                ("human", "{input}"),
                ("assistant", "I'll help you with that. Let me break this down:\n"),
            ]
        )

        agent = create_tool_calling_agent(
            self.llm,
            self._tools,
            system_prompt,
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self._tools,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process user query with full agentic RAG pipeline

        Args:
            query: User question or request
            conversation_history: Previous messages for context

        Returns:
            Dict with response, sources, and intermediate steps
        """

        logger.info(f"Processing query for org {self.org_id}: {query[:100]}")

        try:
            # Add conversation context if provided
            input_text = query
            if conversation_history:
                history_str = "\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in conversation_history[-5:]
                    ]
                )
                input_text = (
                    f"Conversation history:\n{history_str}\n\nNew question: {query}"
                )

            # Execute agent
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"input": input_text},
            )

            # Extract sources and tool calls
            sources = self._extract_sources(result)
            tool_calls = self._extract_tool_calls(result)

            return {
                "success": True,
                "response": result.get("output", ""),
                "sources": sources,
                "tool_calls": tool_calls,
                "intermediate_steps": len(result.get("intermediate_steps", [])),
                "org_id": self.org_id,
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "response": f"I encountered an error: {str(e)}",
                "error": str(e),
                "org_id": self.org_id,
            }

    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data sources from agent result"""
        sources = []

        # Parse intermediate steps for tool calls and their results
        for step in result.get("intermediate_steps", []):
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                if hasattr(action, "tool"):
                    try:
                        obs_data = json.loads(observation)
                        sources.append(
                            {
                                "tool": action.tool,
                                "timestamp": obs_data.get("timestamp"),
                                "success": obs_data.get("success", False),
                            }
                        )
                    except (json.JSONDecodeError, AttributeError):
                        pass

        return sources

    def _extract_tool_calls(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from agent result"""
        tool_calls = []

        for step in result.get("intermediate_steps", []):
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                if hasattr(action, "tool"):
                    tool_calls.append(
                        {
                            "tool": action.tool,
                            "input": action.tool_input
                            if hasattr(action, "tool_input")
                            else {},
                        }
                    )

        return tool_calls

    async def stream_response(
        self,
        query: str,
        callback=None,
    ):
        """
        Stream response for real-time updates

        Useful for UI that wants to show thinking process
        """

        logger.info(f"Streaming response for query: {query[:50]}")

        try:
            # Note: This is a simplified streaming implementation
            # For full streaming, integrate with LangChain's streaming APIs
            result = await self.process_query(query)

            if callback:
                callback(result)

            return result

        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            raise

    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available tools"""
        return self.mcp_server.get_tools_schema()


class MultiOrgAgenticRAG:
    """Manager for multiple org RAG instances"""

    def __init__(self):
        self._instances: Dict[str, AgenticRAGSystem] = {}

    def get_system(self, org_id: str) -> AgenticRAGSystem:
        """Get or create RAG system for org"""
        if org_id not in self._instances:
            self._instances[org_id] = AgenticRAGSystem(org_id)
            logger.info(f"Created RAG system for org {org_id}")

        return self._instances[org_id]

    async def process_query(self, org_id: str, query: str) -> Dict[str, Any]:
        """Process query for specific org"""
        system = self.get_system(org_id)
        return await system.process_query(query)

    def cleanup_org(self, org_id: str):
        """Clean up RAG system for org"""
        if org_id in self._instances:
            del self._instances[org_id]
            logger.info(f"Cleaned up RAG system for org {org_id}")


# Global instance
_rag_manager: Optional[MultiOrgAgenticRAG] = None


def get_rag_system(org_id: str) -> AgenticRAGSystem:
    """Get RAG system for org"""
    global _rag_manager

    if _rag_manager is None:
        _rag_manager = MultiOrgAgenticRAG()

    return _rag_manager.get_system(org_id)


async def process_rag_query(org_id: str, query: str) -> Dict[str, Any]:
    """Process query with agentic RAG"""
    system = get_rag_system(org_id)
    return await system.process_query(query)

import os
import logging
from typing import Dict, List, Optional
import openai
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from services.user_service import get_user_with_org


class ChatService:
    def __init__(self, org_id: Optional[str] = None, chatbot_config: Optional[dict] = None):
        """Initialize ChatService with optional parameters for flexibility"""
        self.org_id = org_id
        self.namespace = f"org-{org_id}" if org_id else None
        self.chatbot_config = chatbot_config or {}

        # Initialize OpenAI
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX"))

        # Initialize embeddings
        self.embedder = OpenAIEmbeddings()

    async def generate_response(self, message: str, conversation_id: str = "sandbox") -> Dict:
        """Generate AI response using RAG with organization's documents"""
        try:
            # Step 1: Generate embedding for user query
            query_embedding = await self.generate_embedding(message)

            # Step 2: Search relevant documents in organization's namespace
            relevant_docs = await self.search_documents(query_embedding)

            # Step 3: Build context from retrieved documents
            context = self.build_context(relevant_docs)

            # Step 4: Generate AI response using OpenAI with chatbot personality
            response = await self.generate_openai_response(message, context)

            # Step 5: Extract sources
            sources = [
                doc.get('metadata', {}).get('source', '')
                for doc in relevant_docs
            ]

            return {
                "response": response,
                "sources": list(set(filter(None, sources))),
            }

        except Exception as e:
            logging.error("Error generating response: %s", e)
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": []
            }

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise

    async def search_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for relevant documents in organization's Pinecone namespace"""
        try:
            if not self.namespace:
                return []

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )

            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            return []

    def build_context(self, relevant_docs: List[Dict]) -> str:
        """Build context string from relevant documents"""
        if not relevant_docs:
            return "No relevant information found in the uploaded documents."

        context_parts = []
        for doc in relevant_docs:
            metadata = doc.get('metadata', {})
            chunk_text = metadata.get('chunk', '')
            source = metadata.get('source', 'Unknown source')

            if chunk_text:
                context_parts.append(
                    f"Source: {source}\nContent: {chunk_text}\n")

        return "\n---\n".join(context_parts)

    async def generate_openai_response(self, user_message: str, context: str) -> str:
        """Generate response using OpenAI with context and chatbot personality"""
        try:
            # Build personality-aware system prompt
            chatbot_name = self.chatbot_config.get('name', 'AI Assistant')
            tone = self.chatbot_config.get('tone', 'helpful and professional')
            behavior = self.chatbot_config.get(
                'behavior', 'Be helpful and informative')

            system_prompt = f"""You are {chatbot_name}, a {tone} AI assistant. {behavior}

Context from uploaded documents:
{context}

Instructions:
- You are a knowledgeable assistant for this business
- Answer questions using the information you know about our products and services
- NEVER mention "documents", "uploaded files", "training data", or "provided information"
- Respond naturally as if this information is part of your knowledge
- If you don't have specific information about something, politely say you don't have details on that
- Maintain a {tone} and professional tone
- Be helpful and provide accurate information about our offerings
- Stay in character as {chatbot_name}

Remember: You are NOT referencing documents - this information is simply what you know about the business.
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error("Error generating OpenAI response: %s", e)
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    async def generate_contextual_response(
        self,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        chatbot_id: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate response using document context - FIXED VERSION"""
        try:
            # 1. Get user's organization if not already set
            if not self.org_id:
                user_data = await get_user_with_org(user_id)
                self.org_id = user_data.get("org_id")
                self.namespace = f"org-{self.org_id}" if self.org_id else None

            # 2. Search for relevant document context
            context_chunks = await self.search_relevant_context(message, user_id)

            # 3. Build context string
            context_text = ""
            sources = []

            if context_chunks:
                context_text = "\n\nRelevant information from your documents:\n"
                for chunk in context_chunks[:3]:
                    context_text += f"\n{chunk['content']}\n"
                    sources.append(chunk['upload_id'])

            # 4. Enhanced prompt
            system_prompt = f"""You are {self.chatbot_config.get('name', 'AI Assistant')}, a helpful AI assistant.

Instructions:
- Use the provided information to answer questions when relevant
- Provide natural, conversational responses
- DO NOT mention sources, documents, or citations in your response
- Answer as if the information is your own knowledge
- If the provided information doesn't contain relevant details, use your general knowledge
- Be concise but thorough
- Maintain a {self.chatbot_config.get('tone', 'helpful')} tone"""

            user_prompt = f"""
User Question: {message}
{context_text}

Please provide a helpful, natural response without mentioning any sources or documents.
"""

            # 5. Generate AI response
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )

            ai_response = response.choices[0].message.content

            # 6. Return response without sources
            return {
                "response": ai_response,
                "context_used": len(context_chunks) > 0,
                "sources": [],  # Always empty - no sources shown
                "context_chunks_count": len(context_chunks),
                "conversation_id": conversation_id
            }

        except Exception as e:
            logging.error(f"[Error] Response generation failed: {e}")
            # Fallback to basic response
            return await self.generate_basic_response(message)

    async def search_relevant_context(self, query: str, user_id: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant context from uploaded documents"""
        try:
            # Get user's organization if not set
            if not self.org_id:
                user_data = await get_user_with_org(user_id)
                self.org_id = user_data.get("org_id")
                self.namespace = f"org-{self.org_id}" if self.org_id else None

            if not self.org_id:
                return []

            # Convert query to embedding
            query_embedding = self.embedder.embed_query(query)

            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True,
                include_values=False
            )

            # Format context
            context_chunks = []
            for match in search_results.matches:
                if match.score > 0.7:  # Only use high-confidence matches
                    context_chunks.append({
                        "content": match.metadata.get("chunk", ""),
                        "score": float(match.score),
                        "upload_id": match.metadata.get("upload_id", ""),
                        "source": match.id
                    })

            return context_chunks

        except Exception as e:
            logging.error(f"[Error] Context search failed: {e}")
            return []

    async def generate_basic_response(self, message: str) -> Dict[str, any]:
        """Fallback method for basic responses without context"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide natural, conversational responses."},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return {
                "response": response.choices[0].message.content,
                "context_used": False,
                "sources": [],
                "context_chunks_count": 0
            }

        except Exception as e:
            logging.error(f"[Error] Basic response generation failed: {e}")
            return {
                "response": "I'm sorry, I'm having trouble generating a response right now. Please try again.",
                "context_used": False,
                "sources": [],
                "context_chunks_count": 0,
                "error": str(e)
            }

    async def generate_public_response(
        self,
        message: str,
        org_id: str,
        chatbot_config: dict,
        session_id: Optional[str] = None
    ) -> Dict[str, any]:
        """Generate response for public/embedded chatbot"""
        try:
            # Set organization context
            namespace = f"org-{org_id}"

            # Convert query to embedding
            query_embedding = self.embedder.embed_query(message)

            # Search Pinecone for organization's content
            search_results = self.index.query(
                vector=query_embedding,
                top_k=3,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )

            # Build context
            context_text = ""
            if search_results.matches:
                context_text = "\n\nRelevant information:\n"
                for match in search_results.matches:
                    if match.score > 0.7:
                        context_text += f"\n{match.metadata.get('chunk', '')}\n"

            # Use chatbot's custom configuration
            system_prompt = chatbot_config.get("system_prompt") or f"""
You are {chatbot_config['name']}, a helpful AI assistant.

Personality: {chatbot_config.get('tone', 'helpful')}
Behavior: {chatbot_config.get('behavior', 'Be helpful and informative')}

Instructions:
- Use the provided information to answer questions when relevant
- Maintain your specified personality and tone
- Be conversational and helpful
- Don't mention sources unless specifically asked
"""

            user_prompt = f"""
User Question: {message}
{context_text}

Please provide a helpful response in your characteristic style.
"""

            # Generate response with chatbot's settings
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=chatbot_config.get("max_tokens", 1000),
                temperature=chatbot_config.get("temperature", 0.7)
            )

            return {
                "response": response.choices[0].message.content,
                "context_used": len(search_results.matches) > 0,
                "session_id": session_id
            }

        except Exception as e:
            logging.error(f"[Error] Public response generation failed: {e}")
            return {
                "response": "I'm sorry, I'm having trouble right now. Please try again.",
                "context_used": False,
                "session_id": session_id
            }

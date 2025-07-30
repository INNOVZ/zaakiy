import os
import openai
from pinecone import Pinecone
from typing import Dict, List, Optional
import logging


class ChatService:
    def __init__(self, org_id: str, chatbot_config: dict):
        self.org_id = org_id
        self.namespace = f"org-{org_id}"
        self.chatbot_config = chatbot_config

        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX"))

    async def generate_response(
        self,
        message: str,
        conversation_id: str = "sandbox"
    ) -> Dict:
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
            logging.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": []
            }

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
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
- Answer the user's question based on the provided context
- Maintain a {tone} tone throughout the conversation
- If the context doesn't contain relevant information, politely say so
- Be concise but informative
- Cite specific information from the documents when relevant
- Stay in character as {chatbot_name}
"""

            response = openai.chat.completions.create(
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
            logging.error(f"Error generating OpenAI response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

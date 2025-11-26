"""
Text Processing Module
Handles text chunking, filtering, and embedding generation
"""

import logging

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


def split_into_chunks(text: str) -> list:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    if not text.strip():
        return []

    # Clean the text first using shared TextCleaner utility
    cleaned_text = TextCleaner.clean_text(text, remove_noise=False)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(cleaned_text)
    return [chunk for chunk in chunks if chunk.strip()]


def filter_noise_chunks(chunks: list) -> list:
    """
    Filter out chunks that are clearly UI noise (login/cart/country lists, etc.).

    The goal is to avoid embedding/upserting low-signal text that hurts search quality.

    This function uses the shared TextCleaner utility for consistency.
    """
    return TextCleaner.filter_noise_chunks(chunks, min_length=60)


def get_embeddings_for_chunks(chunks: list) -> list:
    """Generate embeddings for text chunks using OpenAI"""
    if not chunks:
        return []

    embedder = OpenAIEmbeddings()
    vectors = embedder.embed_documents(chunks)
    return vectors

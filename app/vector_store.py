"""ChromaDB vector store initialization and management"""
import os
from pathlib import Path
from chromadb import PersistentClient
from app.config import CHROMA_DB_PATH, COLLECTION_NAME


def get_chroma_client() -> PersistentClient:
    """Initialize and return a persistent ChromaDB client"""
    # Ensure the directory exists
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    client = PersistentClient(path=CHROMA_DB_PATH)
    return client


def get_or_create_collection(client: PersistentClient = None):
    """Get or create the anime embeddings collection"""
    if client is None:
        client = get_chroma_client()
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def get_collection_info(collection) -> dict:
    """Get information about the collection"""
    data = collection.get()
    return {
        "total_documents": len(data["ids"]),
        "collection_name": COLLECTION_NAME
    }


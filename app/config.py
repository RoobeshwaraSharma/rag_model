"""Configuration management using environment variables from .env file"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Anime_embeddings")

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0"))

# Data Configuration
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", str(BASE_DIR / "data" / "Anime_Cleaned.csv"))

# Vector Store Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
SEARCH_K = int(os.getenv("SEARCH_K", "10"))  # Reduced from 1000 to avoid token limits

# Validate required configuration
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in .env file")


"""Script to initialize ChromaDB vector store from CSV file"""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BASE_DIR = Path(__file__).parent.parent

# Get config values (without GROQ_API_KEY validation)
CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", str(BASE_DIR / "data" / "Anime_Cleaned.csv"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Anime_embeddings")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def initialize_vector_store():
    """Initialize ChromaDB vector store from CSV file"""
    print("🚀 Starting vector store initialization...")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_FILE_PATH}")
    
    print(f"📂 Loading CSV from: {CSV_FILE_PATH}")
    
    # Load CSV using pandas (more reliable than CSVLoader)
    try:
        df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert DataFrame to LangChain Documents
        # Combine all columns into a single text representation
        data = []
        for idx, row in df.iterrows():
            # Create a text representation of the row
            text_parts = []
            for col in df.columns:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
            text = "\n".join(text_parts)
            data.append(Document(page_content=text, metadata={"row_index": idx}))
        
        print(f"✅ Converted to {len(data)} documents")
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        print(f"   File path: {CSV_FILE_PATH}")
        print(f"   File exists: {os.path.exists(CSV_FILE_PATH)}")
        raise
    
    # Split text
    print("✂️  Splitting documents...")
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(data)
    print(f"✅ Split into {len(texts)} chunks")
    
    # Initialize embedding model
    print(f"🤖 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Embedding model loaded")
    
    # Initialize ChromaDB client
    print(f"💾 Initializing ChromaDB at: {CHROMA_DB_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = PersistentClient(path=CHROMA_DB_PATH)
    
    # Get or create collection
    print(f"📚 Creating/getting collection: {COLLECTION_NAME}")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Check if collection already has data
    existing_count = len(collection.get()["ids"])
    if existing_count > 0:
        response = input(
            f"⚠️  Collection already has {existing_count} documents. "
            "Do you want to delete and rebuild? (yes/no): "
        )
        if response.lower() == "yes":
            client.delete_collection(name=COLLECTION_NAME)
            collection = client.create_collection(name=COLLECTION_NAME)
            print("✅ Existing collection deleted")
        else:
            print("❌ Aborted. Keeping existing collection.")
            return
    
    # Batch process embeddings
    print(f"🔄 Processing embeddings in batches of {BATCH_SIZE}...")
    documents = [doc.page_content for doc in texts]
    ids = [str(i) for i in range(len(texts))]
    
    for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        end = min(start + BATCH_SIZE, len(texts))
        batch_docs = texts[start:end]
        batch_ids = ids[start:end]
        batch_texts = [doc.page_content for doc in batch_docs]
        
        # Generate embeddings
        batch_embeddings = [model.encode(text) for text in batch_texts]
        
        # Normalize embeddings
        batch_embeddings = np.array(batch_embeddings)
        batch_embeddings = batch_embeddings / np.linalg.norm(
            batch_embeddings, axis=1, keepdims=True
        )
        
        # Add to collection
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings.tolist()
        )
    
    # Verify
    total_docs = len(collection.get()["ids"])
    print(f"\n✅ Batch upload completed!")
    print(f"📊 Total embeddings in collection '{COLLECTION_NAME}': {total_docs}")
    print(f"💾 Vector store saved at: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    try:
        initialize_vector_store()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


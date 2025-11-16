"""Script to initialize ChromaDB vector store from CSV file"""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from app.config import (
    CSV_FILE_PATH,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    EMBEDDING_MODEL_NAME
)


def initialize_vector_store():
    """Initialize ChromaDB vector store from CSV file"""
    print("🚀 Starting vector store initialization...")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_FILE_PATH}")
    
    print(f"📂 Loading CSV from: {CSV_FILE_PATH}")
    
    # Load CSV
    loader = CSVLoader(file_path=CSV_FILE_PATH)
    data = loader.load()
    print(f"✅ Loaded {len(data)} documents from CSV")
    
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


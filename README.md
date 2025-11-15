# Anime Movie Recommender - RAG Model API

A RAG (Retrieval-Augmented Generation) based FastAPI application that recommends anime movies based on user preferences. The system uses ChromaDB for vector storage, Groq LLM for recommendations, and HuggingFace embeddings.

## Features

- **RAG-based Recommendations**: Uses retrieval-augmented generation to provide context-aware anime recommendations
- **FastAPI REST API**: Clean RESTful API with automatic documentation
- **CORS Enabled**: Public access from any domain
- **Docker Support**: Containerized for easy deployment
- **Persistent Vector Store**: ChromaDB for efficient similarity search

## Project Structure

```
rag_model/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── rag_service.py       # RAG chain and recommendation logic
│   ├── vector_store.py      # ChromaDB initialization
│   └── config.py            # Configuration management
├── data/
│   └── Anime_Cleaned.csv    # Input CSV file
├── chroma_db/               # ChromaDB persistence (created at runtime)
├── scripts/
│   └── initialize_db.py     # Script to build vector store
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Prerequisites

- Python 3.11+
- Groq API key ([Get one here](https://console.groq.com/))
- Docker (optional, for containerized deployment)

## Local Setup

### 1. Clone and Setup

```bash
# Navigate to project directory
cd rag_model

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Initialize Vector Store

Before running the API, you need to initialize the ChromaDB vector store from the CSV file:

```bash
python scripts/initialize_db.py
```

This will:

- Load the CSV file
- Split documents into chunks
- Generate embeddings using the HuggingFace model
- Store embeddings in ChromaDB

**Note**: This process may take 15-20 minutes depending on your system.

### 4. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## API Endpoints

### POST `/recommend`

Get anime recommendations based on user query.

**Request Body:**

```json
{
  "query": "Naruto"
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "recommended_title": "Naruto Shippuden",
      "genre": ["Action", "Drama", "Fantasy"],
      "rating": 4.25,
      "match_score": 0.95
    }
  ],
  "query": "Naruto",
  "error": null
}
```

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy"
}
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t anime-recommender .
```

### Run Docker Container

```bash
docker run -d \
  --name anime-recommender \
  -p 8000:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender
```

**Note**: Make sure to initialize the vector store before running the container, or mount the `chroma_db` directory if it already exists.

### Initialize DB in Docker

If you need to initialize the database inside the container:

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  --env-file .env \
  anime-recommender \
  python scripts/initialize_db.py
```

## Testing the API

### Using curl

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "action anime with ninjas"}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={"query": "Naruto"}
)
print(response.json())
```

### Using the Interactive Docs

Visit http://localhost:8000/docs for the Swagger UI where you can test the API interactively.

## AWS EC2 Deployment

See [deploy/aws_deployment_guide.md](deploy/aws_deployment_guide.md) for detailed step-by-step instructions on deploying to AWS EC2.

## Configuration

All configuration is managed through environment variables in the `.env` file:

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_MODEL_NAME`: Groq model to use (default: llama-3.3-70b-versatile)
- `EMBEDDING_MODEL_NAME`: HuggingFace embedding model (default: all-MiniLM-L6-v2)
- `CHROMA_DB_PATH`: Path to ChromaDB persistence directory
- `COLLECTION_NAME`: Name of the ChromaDB collection
- `SEARCH_K`: Number of documents to retrieve (default: 1000)

## Troubleshooting

### Vector Store Not Found

If you get an error about the vector store not being initialized:

```bash
python scripts/initialize_db.py
```

### Out of Memory

If you encounter memory issues during initialization, try reducing `BATCH_SIZE` in `.env`:

```env
BATCH_SIZE=50
```

### API Key Errors

Make sure your `.env` file contains a valid `GROQ_API_KEY`.

## License

This project is for development and testing purposes.

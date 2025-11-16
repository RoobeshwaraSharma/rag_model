"""FastAPI application for Anime Recommendation RAG Model"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models import QueryRequest, RecommendationResponse, AnimeRecommendation
from app.rag_service import recommend_anime

# Initialize FastAPI app
app = FastAPI(
    title="Anime Movie Recommender API",
    description="A RAG-based API that recommends anime movies based on user preferences.",
    version="1.0.0"
)

# Enable CORS for public access from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for public access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Anime Movie Recommender API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/recommend")
async def recommend_anime_endpoint(request: QueryRequest):
    """
    Get anime recommendations based on user query
    
    Args:
        request: QueryRequest with user query
        
    Returns:
        Array of anime recommendations (for frontend compatibility)
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = recommend_anime(request.query)
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(
                status_code=500, 
                detail=f"Recommendation error: {result.get('error')}"
            )
        
        # Convert recommendations list to AnimeRecommendation objects
        recommendations = []
        for rec in result.get("recommendations", []):
            try:
                # Validate and convert to dict for JSON serialization
                anime_rec = AnimeRecommendation(**rec)
                recommendations.append(anime_rec.model_dump())
            except Exception as e:
                # Skip invalid recommendations
                continue
        
        # Return array directly (as expected by frontend)
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


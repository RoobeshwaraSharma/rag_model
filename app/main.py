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


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_anime_endpoint(request: QueryRequest):
    """
    Get anime recommendations based on user query
    
    Args:
        request: QueryRequest with user query
        
    Returns:
        RecommendationResponse with anime recommendations
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = recommend_anime(request.query)
        
        # Convert recommendations list to AnimeRecommendation objects
        recommendations = []
        for rec in result.get("recommendations", []):
            try:
                recommendations.append(AnimeRecommendation(**rec))
            except Exception as e:
                # Skip invalid recommendations
                continue
        
        # Convert to response model
        response = RecommendationResponse(
            recommendations=recommendations,
            query=result.get("query", request.query),
            error=result.get("error")
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


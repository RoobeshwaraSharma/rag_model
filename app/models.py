"""Pydantic models for request/response validation"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for anime recommendation query"""
    query: str = Field(..., description="User query or anime preference")


class AnimeRecommendation(BaseModel):
    """Individual anime recommendation"""
    recommended_title: str = Field(..., description="Anime title")
    genre: List[str] = Field(..., description="List of genres")
    rating: float = Field(..., description="Anime rating")
    match_score: float = Field(..., ge=0, le=1, description="Match score between 0 and 1")


class RecommendationResponse(BaseModel):
    """Response model for anime recommendations"""
    recommendations: List[AnimeRecommendation] = Field(..., description="List of recommended anime")
    query: str = Field(..., description="Original query")
    error: Optional[str] = Field(None, description="Error message if any")


"""RAG service for anime recommendations"""
import json
import os
from typing import Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from app.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    EMBEDDING_MODEL_NAME,
    SEARCH_K
)


# Initialize embeddings model
_embeddings = None
_vectorstore = None
_qa_chain = None


def get_embeddings():
    """Get or initialize the embeddings model"""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vectorstore():
    """Get or initialize the Chroma vector store"""
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    return _vectorstore


def get_qa_chain():
    """Get or initialize the RAG QA chain"""
    global _qa_chain
    if _qa_chain is None:
        # Initialize LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL_NAME,
            temperature=GROQ_TEMPERATURE
        )
        
        # Initialize vector store
        vectorstore = get_vectorstore()
        # Retrieve more documents to get 10-12 recommendations (retrieve 15-20 for context)
        retriever = vectorstore.as_retriever(search_kwargs={"k": min(SEARCH_K, 20)})
        
        # Create prompt template (LangChain 1.0.0 uses LCEL pattern)
        system_prompt = """You are an intelligent anime recommender that uses content-based filtering and cosine similarity.
You are given context data about various anime, including their name, genre, rating, and synopsis.

Your job:
- Understand the user's interest.
- If user search by exact anime movie title suggest that movie and all relevant suggested anime movies from 'Anime_embeddings'.
- If the user's input refers to a *Hollywood or non-anime movie*, do NOT attempt to match it directly.
- Instead, recommend the **top-rated anime** from the dataset (sorted by rating or relevance).
- Match the user's preferences with similar anime from the context.
- **IMPORTANT: Provide 10-12 anime recommendations if available in the context. If fewer are available, provide as many as possible.**
- Respond strictly in JSON format for frontend use.
- Do not include any extra text outside the JSON.

Your response should be a JSON array with 10-12 recommendations like this:
[
  {{
    "recommended_title": "string",
    "genre": ["string"],
    "rating": float,
    "match_score": float (between 0 and 1)
  }}
]

Do not include any extra text outside the JSON."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nUser question or preference: {question}")
        ])
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            # Limit context length to avoid token limits
            # Take first 15 documents and truncate each to max 200 chars to fit more documents
            formatted = []
            for doc in docs[:15]:  # Limit to 15 documents to have enough context for 10-12 recommendations
                content = doc.page_content[:200]  # Truncate each document to 200 chars to fit more
                formatted.append(content)
            return "\n\n".join(formatted)
        
        _qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return _qa_chain


def recommend_anime(query: str) -> Dict[str, Any]:
    """
    Get anime recommendations based on user query
    
    Args:
        query: User's query or preference
        
    Returns:
        Dictionary with recommendations or error
    """
    try:
        qa_chain = get_qa_chain()
        # LCEL chain returns string directly
        recommendation_str = qa_chain.invoke(query)
        
        # Try to extract JSON from the response
        # Sometimes LLM adds extra text, so we try to find JSON array
        try:
            # Try direct JSON parsing
            recommendation_json = json.loads(recommendation_str)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks or other formats
            import re
            json_match = re.search(r'\[.*\]', recommendation_str, re.DOTALL)
            if json_match:
                recommendation_json = json.loads(json_match.group())
            else:
                raise ValueError("Could not parse JSON from model response")
        
        return {
            "recommendations": recommendation_json,
            "query": query,
            "error": None
        }
    except Exception as e:
        return {
            "recommendations": [],
            "query": query,
            "error": str(e)
        }


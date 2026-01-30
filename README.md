# K-Drama Recommendation System

A content-based hybrid recommendation system for Korean dramas using
semantic embeddings, metadata, and similarity search with KNN.

## Features
- Semantic similarity using text embeddings
- Actor-based similarity
- Metadata re-ranking (rating, episodes, release year)
- Interactive Streamlit app
- Poster retrieval via TMDB API

## Methodology
The system combines:
- Cosine similarity on narrative embeddings
- Cosine similarity on cast embeddings
- Re-ranking using normalized metadata features

Final recommendations are generated using a weighted scoring strategy.


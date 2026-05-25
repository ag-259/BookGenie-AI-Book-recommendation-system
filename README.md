BookGenie – AI Powered Book Recommendation System

BookGenie is a full-stack AI-based book recommendation platform that combines Machine Learning, Semantic Search, Collaborative Filtering, and Generative AI (RAG) to deliver highly personalized book recommendations.

The system allows users to:

Discover books using natural language
Get AI-generated recommendations
Like, rate, and organize books
Receive personalized suggestions based on reading behavior
Explore semantic and content-based recommendations

This project was built using Flask, Scikit-Learn, ChromaDB, OpenAI Embeddings, and SQLite.

🚀 Features
🔍 Hybrid Recommendation System

Combines multiple recommendation techniques:

TF-IDF Content-Based Filtering
Semantic Vector Search
Collaborative Filtering
Hybrid Recommendation Engine
🤖 AI RAG Book Assistant

Users can ask natural language queries like:

“Recommend emotional fantasy books”
“Books similar to Harry Potter”
“Suspenseful mystery novels with strong female leads”

The system:

Retrieves relevant books using vector search
Uses OpenAI to generate intelligent recommendations
Explains WHY books match the query
📖 Personalized User Experience
User Authentication System
Reading Lists
Want to Read / Reading / Completed tracking
Liked Books
Personalized Home Feed
User Profiles & Statistics
⭐ Rating & Recommendation Engine
5-Star Rating System
Trending Books
Highest Rated Books
Similar User Recommendations
Item-Based Collaborative Filtering
🧠 Semantic Search with Vector Database

Implemented using:

OpenAI Embeddings
ChromaDB Vector Store
Cosine Similarity Search

This enables contextual recommendations instead of simple keyword matching.

🛠️ Tech Stack
Backend
Python
Flask
SQLite
Machine Learning / AI
Scikit-Learn
TF-IDF Vectorization
Cosine Similarity
OpenAI Embeddings
ChromaDB
RAG (Retrieval-Augmented Generation)
Frontend
HTML
CSS
JavaScript
Jinja Templates
Data Processing
Pandas
NumPy
🧩 Recommendation Architecture
1. Content-Based Filtering

Uses:

TF-IDF Vectorization
Cosine Similarity

Recommends books based on:

Titles
Categories
Metadata
2. Semantic Search

Uses:

OpenAI Embeddings
Chroma Vector Database

Allows semantic understanding of user queries.

Example:

“dark psychological thrillers”

instead of exact keyword matching.

3. Collaborative Filtering

Recommends books liked by users with similar preferences.

Includes:

User-based filtering
Item-based filtering
4. Hybrid Recommendation Engine

Combines:

Content-based recommendations
Semantic search results
Collaborative filtering

to improve recommendation quality and diversity.

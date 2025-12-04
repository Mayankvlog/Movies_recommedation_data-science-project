import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import logging
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from numpy.linalg import norm
import os
from datetime import datetime

# ===== STEP 2: Logging aur MLflow (exact same) =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===== STEP 3: EXACT SAME MongoDB SETUP =====
MONGO_URI = "mongodb://localhost:27017"  
DB_NAME = "movie_recommendation_db"
COLLECTION_NAME = "movies"

@st.cache_resource
def get_mongodb_connection():
    """Exact same MongoDB connection jaise notebook mein"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    movies_collection = db[COLLECTION_NAME]
    
    # Test connection (exact same print)
    st.info("MongoDB connected: True")
    logging.info("MongoDB connected successfully!")
    
    return client, db, movies_collection

# Global connection
client, db, movies_collection = get_mongodb_connection()

# ===== MOVIES LOAD AND PREPROCESS (exact same functions) =====
def extract_names(json_str):
    """Exact same function from notebook"""
    try:
        if pd.isna(json_str): return []
        items = json.loads(json_str)
        return [item['name'] for item in items]
    except (json.JSONDecodeError, TypeError):
        return []

def create_soup(x):
    """Exact same soup function"""
    return ' '.join(x['genres'])

@st.cache_data
def load_and_preprocess_movies():
    """Load and preprocess exactly jaise notebook mein"""
    logging.info("Loading dataset...")
    
    # Try MongoDB first
    movies_list = list(movies_collection.find({}, {"_id": 0}).limit(5000))
    if movies_list:
        logging.info("Movies loaded from MongoDB!")
        movies_df = pd.DataFrame(movies_list)
    else:
        # Fallback to CSV
        logging.info("Loading from CSV...")
        movies_df = pd.read_csv('data/tmdb_5000_movies.csv')
    
    # Exact same preprocessing
    logging.info("Preprocessing data...")
    movies_df['genres'] = movies_df['genres'].apply(extract_names)
    movies_df['soup'] = movies_df.apply(create_soup, axis=1)
    logging.info("Data preprocessing complete.")
    
    st.success(f"✅ Loaded {len(movies_df)} movies!")
    return movies_df

# ===== MAIN APP =====
st.title("🎬 Movie Recommendation System")

# Load movies (exact notebook flow)
movies_df = load_and_preprocess_movies()

# Load model files
try:
    encoder = load_model("model/movie_recommender.h5")
    tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
    st.success("✅ Model loaded!")
except:
    st.error("❌ Model files missing!")

# ===== RECOMMENDATION FUNCTION (MongoDB + Model) =====
def get_recommendations(title, top_k=5):
    """Recommendations using MongoDB embeddings"""
    
    # Find movie in MongoDB
    movie_doc = movies_collection.find_one({"title": {"$regex": title, "$options": "i"}})
    if not movie_doc or not movie_doc.get("embedding"):
        st.warning(f"No embedding for '{title}'")
        return []
    
    query_emb = np.array(movie_doc["embedding"])
    
    # Get all movies with embeddings
    all_movies = list(movies_collection.find({"embedding": {"$exists": True}}))
    
    similarities = []
    for movie in all_movies:
        movie_emb = np.array(movie["embedding"])
        sim = float(query_emb @ movie_emb / (norm(query_emb) * norm(movie_emb) + 1e-10))
        similarities.append({
            "title": movie["title"],
            "similarity": sim,
            "genres": movie.get("genres", [])
        })
    
    # Sort and return top recommendations
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return [r["title"] for r in similarities[1:top_k+1]]

# ===== UI =====
tab1, tab2 = st.tabs(["🎯 Recommendations", "📊 Database Stats"])

with tab1:
    selected_movie = st.selectbox("Select movie:", movies_df['title'].tolist())
    
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Finding similar movies..."):
            recs = get_recommendations(selected_movie)
            if recs:
                st.success("✅ Recommendations from MongoDB!")
                for i, movie in enumerate(recs, 1):
                    st.write(f"{i}. **{movie}**")
            else:
                st.warning("No recommendations found!")

with tab2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_movies = movies_collection.count_documents({})
        st.metric("Total Movies", total_movies)
    
    with col2:
        movies_with_emb = movies_collection.count_documents({"embedding": {"$exists": True}})
        st.metric("Movies with Embeddings", movies_with_emb)
    
    with col3:
        coverage = (movies_with_emb/total_movies*100) if total_movies else 0
        st.metric("Embedding Coverage", f"{coverage:.1f}%")

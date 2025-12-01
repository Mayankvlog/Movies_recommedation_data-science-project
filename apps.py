import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from recommender import recommend 
from pymongo import MongoClient
import os

# --- MongoDB Setup ---
@st.cache_resource
def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/moviesdb")
    client = MongoClient(mongo_uri)
    return client, client.get_database()

# --- Caching to load resources only once ---
@st.cache_resource
def load_artifacts():
    """
    Loads the machine learning model and vectorizer from disk.
    """
    try:
        encoder = load_model("model/movie_recommender.h5")
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        return encoder, tfidf, movies_df
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure 'movie_recommender.h5', 'tfidf_vectorizer.pkl', and 'movies_df.pkl' are in a 'model/' directory.")
        return None, None, None

@st.cache_data
def generate_embeddings(_encoder, _tfidf, _movies_df):
    """
    Generates and caches movie embeddings.
    """
    if _encoder is None or _tfidf is None or _movies_df is None:
        return None
    
    tfidf_matrix = _tfidf.transform(_movies_df['soup'])
    movie_embeddings = _encoder.predict(tfidf_matrix.toarray())
    return movie_embeddings

# --- Main App ---

st.title("🎬 Movie Recommendation System")

# Mongo client and DB
mongo_client, mongo_db = get_mongo_client()
recommendations_collection = mongo_db.get_collection("recommendations")

# Load artifacts and generate embeddings
encoder_model, tfidf_vectorizer, movies_data = load_artifacts()
movie_embeddings = generate_embeddings(encoder_model, tfidf_vectorizer, movies_data)

if movies_data is not None and movie_embeddings is not None:
    movie_list = movies_data['title'].values
    selected_movie = st.selectbox("Choose a movie you like:", movie_list)

    if st.button("Recommend Movies", type="primary"):
        with st.spinner("Finding recommendations..."):
            # The recommend function is now imported
            recommendations = recommend(selected_movie, movies_data, movie_embeddings)
        
        if recommendations:
            # Save query and response to MongoDB
            try:
                doc = {
                    "selected_movie": selected_movie,
                    "recommendations": list(recommendations),
                    "source": "streamlit_app"
                }
                recommendations_collection.insert_one(doc)
            except Exception as e:
                st.warning(f"Could not save recommendations to MongoDB: {e}")

            st.subheader(f"Because you watched '{selected_movie}', you might also like:")
            for i, movie in enumerate(recommendations, 1):
                st.write(f"{i}. {movie}")
        else:
            st.warning("Could not find any recommendations for the selected movie.")
else:
    st.info("The application could not start because the required model files are missing.")


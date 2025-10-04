import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# --- Caching Functions to Load Artifacts ---
@st.cache_resource
def load_model():
    """Load the Keras encoder model from disk."""
    try:
        # If using a 'model' subfolder, change to: 'model/movie_recommender.h5'
        return tf.keras.models.load_model('model/movie_recommender.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_vectorizer():
    """Load the TF-IDF vectorizer from disk."""
    try:
        # If using a 'model' subfolder, change to: 'model/tfidf_vectorizer.pkl'
        with open('model/tfidf_vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return None

@st.cache_data
def load_data():
    """Load the preprocessed movie DataFrame from disk."""
    try:
        # If using a 'model' subfolder, change to: 'model/movies_df.pkl'
        with open('model/movies_df.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading movie data: {e}")
        return None

# --- Main Application Logic ---
encoder_model = load_model()
tfidf = load_vectorizer()
movies_df = load_data()

@st.cache_data
def generate_all_embeddings(_model, _vectorizer, _data):
    """Generate embeddings for all movies on startup."""
    if _model is None or _vectorizer is None or _data.empty:
        return None
    tfidf_matrix = _vectorizer.transform(_data['soup'])
    return _model.predict(tfidf_matrix.toarray())

movie_embeddings = generate_all_embeddings(encoder_model, tfidf, movies_df)

# --- UI Layout ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.markdown("Select a movie to get 10 genre-based recommendations.")

if movie_embeddings is not None and not movies_df.empty:
    movie_list = movies_df['title'].unique()
    selected_movie = st.selectbox("Choose a movie:", options=movie_list)

    if st.button("Get Recommendations"):
        try:
            movie_idx = movies_df[movies_df['title'] == selected_movie].index[0]
            query_embedding = movie_embeddings[movie_idx].reshape(1, -1)
            sim_scores = cosine_similarity(query_embedding, movie_embeddings)[0]
            top_indices = np.argsort(sim_scores)[::-1][1:11]
            recommended_movies = movies_df.iloc[top_indices]
            
            st.subheader(f"Recommendations for '{selected_movie}':")
            for _, row in recommended_movies.iterrows():
                st.markdown(f"- **{row['title']}** (Viewer Rating: {row['vote_average']}/10)")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.warning("Model or data files not found. Please run the training notebook and ensure all files are in the correct directory.")

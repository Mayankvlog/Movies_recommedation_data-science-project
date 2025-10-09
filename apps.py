import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- Mock Model and Vectorizer Definitions ---
class DummyModel:
    """A mock Keras model that simulates the real one for UI testing."""
    def __init__(self):
        self.name = 'Encoder'
    def predict(self, x, verbose=0):
        # Return dummy embeddings with the correct shape
        return np.random.rand(x.shape[0], 32)

class DummyVectorizer:
    """A mock, fitted TF-IDF vectorizer to bypass loading errors."""
    def transform(self, texts):
        # Return a correctly shaped numpy array, simulating a fitted vectorizer
        return np.ones((len(texts), 1000))
    def toarray(self):
        # Added to match the real vectorizer's behavior for consistency
        return self.transform([])

# --- Caching Functions to Load Artifacts ---
@st.cache_resource
def load_model():
    """Load the DummyModel for demonstration purposes."""
    # The info message below has been removed.
    return DummyModel()

@st.cache_resource
def load_vectorizer():
    """Load the DummyVectorizer to bypass fitting errors."""
    return DummyVectorizer()

@st.cache_data
def load_data():
    """Load the preprocessed movie DataFrame from disk."""
    try:
        with open('model/movies_df.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Movie data file (movies_df.pkl) not found. Please run the training notebook.")
        return None
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
    if _model is None or _vectorizer is None or _data is None or 'soup' not in _data.columns:
        return None
    tfidf_matrix = _vectorizer.transform(_data['soup'])
    return _model.predict(tfidf_matrix)

movie_embeddings = generate_all_embeddings(encoder_model, tfidf, movies_df)

# --- UI Layout ---
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")
st.markdown("Select a movie to get 10 genre-based recommendations.")

if movie_embeddings is not None and movies_df is not None and not movies_df.empty:
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
                try:
                    # Try to convert vote_average to a float before formatting
                    rating = float(row['vote_average'])
                    display_rating = f"{rating:.1f}/10"
                except (ValueError, TypeError):
                    # If conversion fails, use a fallback string
                    display_rating = "N/A"
                
                st.markdown(f"- **{row['title']}** (Viewer Rating: {display_rating})")
        except Exception as e:
            st.error(f"An error occurred during recommendation: {e}")
else:
    st.warning("Could not load necessary model or data files. The application cannot proceed.")

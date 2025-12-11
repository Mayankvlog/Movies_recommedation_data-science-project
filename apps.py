import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception as e:
    st.error(f"⚠️ TensorFlow import error: {e}")
    TF_AVAILABLE = False

try:
    from recommender import recommend
    RECOMMENDER_AVAILABLE = True
except Exception as e:
    st.error(f"⚠️ Recommender import error: {e}")
    RECOMMENDER_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .movie-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: translateX(5px);
    }
</style>
""", unsafe_allow_html=True)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    try:
        if not TF_AVAILABLE:
            st.error("❌ TensorFlow not available. Cannot load model.")
            return None, None, None
            
        encoder = load_model("model/movie_recommender.h5", compile=False)
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        return encoder, tfidf, movies_df
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading artifacts: {e}")
        return None, None, None

@st.cache_data
def generate_embeddings(_encoder, _tfidf, _movies_df):
    try:
        if _encoder is None or _tfidf is None or _movies_df is None:
            return None
        
        if 'soup' not in _movies_df.columns:
            if 'genres' in _movies_df.columns:
                _movies_df['soup'] = _movies_df['genres'].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
                )
            else:
                st.warning("⚠️ No 'genres' or 'soup' column found in dataset")
                return None
        
        tfidf_matrix = _tfidf.transform(_movies_df['soup'])
        movie_embeddings = _encoder.predict(tfidf_matrix.toarray(), verbose=0)
        return movie_embeddings
    except Exception as e:
        st.error(f"❌ Error generating embeddings: {e}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Recommendation System</h1>
    <p style="color: white; margin: 0.5rem 0 0 0;">Discover your next favorite movie using AI</p>
</div>
""", unsafe_allow_html=True)

# Load Data
encoder_model, tfidf_vectorizer, movies_data = load_artifacts()
movie_embeddings = generate_embeddings(encoder_model, tfidf_vectorizer, movies_data)

if movies_data is None or movie_embeddings is None:
    st.error("❌ Could not load model. Please check model files.")
    st.stop()

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Find Similar Movies")
    
    # Movie selection
    movie_list = sorted([str(title) for title in movies_data['title'].values if pd.notna(title)])
    
    search = st.text_input("🔍 Search movies", placeholder="Type to filter...")
    filtered = [m for m in movie_list if search.lower() in m.lower()] if search else movie_list
    
    selected = st.selectbox("Select a movie:", filtered)
    
    num_recs = st.slider("Number of recommendations", 5, 15, 10)
    
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        if not RECOMMENDER_AVAILABLE:
            st.error("❌ Recommender function not available")
        else:
            try:
                with st.spinner("Finding matches..."):
                    recommendations = recommend(selected, movies_data, movie_embeddings)[:num_recs]
                
                if recommendations:
                    st.success(f"✅ Found {len(recommendations)} recommendations!")
                    st.markdown("### 🎥 Recommended Movies")
                    
                    for i, movie in enumerate(recommendations, 1):
                        st.markdown(f"""
                        <div class="movie-card">
                            <strong>{i}.</strong> {movie}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No recommendations found.")
            except Exception as e:
                st.error(f"❌ Error getting recommendations: {e}")

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Total Movies", f"{len(movies_data):,}")
    st.metric("Embedding Dim", movie_embeddings.shape[1])
    
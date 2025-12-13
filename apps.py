import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from recommender import recommend

# Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="wide")

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 { color: white; margin: 0; }
    .movie-card {
        background: white; border-radius: 10px; padding: 1rem;
        margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea; transition: transform 0.2s;
    }
    .movie-card:hover { transform: translateX(5px); }
</style>
""", unsafe_allow_html=True)

# Verify files
@st.cache_resource
def verify_files():
    """Check if all required files exist"""
    files = {
        'model/movie_recommender.h5': 'Neural Network Model',
        'model/tfidf_vectorizer.pkl': 'TF-IDF Vectorizer',
        'model/movies_df.pkl': 'Movies Dataset',
        'data/tmdb_5000_movies.csv': 'Movie Data'
    }
    missing = [f for f in files if not os.path.exists(f)]
    return missing, files

# Load artifacts
@st.cache_resource
def load_artifacts():
    """Load model, vectorizer, and movies data"""
    try:
        missing, _ = verify_files()
        if missing:
            return None, None, None, missing
        
        encoder = load_model("model/movie_recommender.h5", compile=False)
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        
        logger.info(f"✓ Loaded {len(movies_df)} movies")
        return encoder, tfidf, movies_df, []
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None, None, None, [str(e)]

# Generate embeddings
@st.cache_data
def generate_embeddings(_encoder, _tfidf, _movies_df):
    """Generate movie embeddings"""
    try:
        if _encoder is None or _tfidf is None or _movies_df is None:
            return None
        
        if 'soup' not in _movies_df.columns:
            if 'genres' in _movies_df.columns:
                _movies_df['soup'] = _movies_df['genres'].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
                )
            else:
                return None
        
        tfidf_matrix = _tfidf.transform(_movies_df['soup'])
        embeddings = _encoder.predict(tfidf_matrix.toarray(), verbose=0)
        logger.info(f"✓ Generated embeddings: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Recommendation System</h1>
    <p style="color: white; margin: 0.5rem 0 0 0;">Discover your next favorite movie using AI</p>
</div>
""", unsafe_allow_html=True)

# Load data
encoder, tfidf, movies_data, errors = load_artifacts()

# Error handling
if errors or encoder is None:
    st.error("❌ Missing or Failed to Load Model Files")
    missing, files = verify_files()
    
    if missing:
        st.write("**Missing files:**")
        for f in missing:
            st.write(f"  - `{f}`")
        st.warning("""
        **Fix:** Copy files to VPS:
        ```bash
        scp -r model/* root@139.59.56.109:~/Movies_recommedation_data-science-project/model/
        scp -r data/* root@139.59.56.109:~/Movies_recommedation_data-science-project/data/
        docker-compose restart web
        ```
        """)
    else:
        st.error(f"Load error: {errors[0] if errors else 'Unknown'}")
    st.stop()

# Generate embeddings
embeddings = generate_embeddings(encoder, tfidf, movies_data)

if embeddings is None:
    st.error("❌ Failed to Generate Embeddings")
    st.stop()

logger.info("✓ App Ready")

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Find Similar Movies")
    
    movie_list = sorted([str(t) for t in movies_data['title'].values if pd.notna(t)])
    search = st.text_input("🔍 Search movies", placeholder="Type to filter...")
    filtered = [m for m in movie_list if search.lower() in m.lower()] if search else movie_list
    
    selected = st.selectbox("Select a movie:", filtered)
    num_recs = st.slider("Number of recommendations", 5, 15, 10)
    
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        try:
            with st.spinner("Finding matches..."):
                recs = recommend(selected, movies_data, embeddings)[:num_recs]
            
            if recs:
                st.success(f"✅ Found {len(recs)} recommendations!")
                st.markdown("### 🎥 Recommended Movies")
                for i, movie in enumerate(recs, 1):
                    st.markdown(f"""
                    <div class="movie-card">
                        <strong>{i}.</strong> {movie}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No recommendations found.")
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            st.error(f"❌ Error: {e}")

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Total Movies", f"{len(movies_data):,}")
    st.metric("Embedding Dim", embeddings.shape[1])
    
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.info("AI-powered movie recommendations using neural embeddings.")
        st.markdown("---")
        st.markdown("**Status:** ✅ Running")
        st.markdown(f"**Models:** ✓ Loaded")
        st.markdown(f"**Embeddings:** ✓ Ready")

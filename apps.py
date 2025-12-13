import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sys
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Log startup
logger.info("🚀 Starting Movie Recommender App...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Model files check: {os.path.exists('model/movie_recommender.h5')}")

TF_AVAILABLE = False
RECOMMENDER_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
    logger.info("✓ TensorFlow imported successfully")
except Exception as e:
    logger.error(f"❌ TensorFlow import error: {e}")
    logger.error(traceback.format_exc())

try:
    from recommender import recommend
    RECOMMENDER_AVAILABLE = True
    logger.info("✓ Recommender module imported successfully")
except Exception as e:
    logger.error(f"❌ Recommender import error: {e}")
    logger.error(traceback.format_exc())

# Page Configuration
try:
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    logger.info("✓ Streamlit page config set")
except Exception as e:
    logger.error(f"Error setting page config: {e}")

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

# Check model files
@st.cache_resource
def verify_model_files():
    """Verify all required model files exist"""
    files_to_check = [
        "model/movie_recommender.h5",
        "model/tfidf_vectorizer.pkl",
        "model/movies_df.pkl"
    ]
    missing_files = []
    for file in files_to_check:
        if not os.path.exists(file):
            missing_files.append(file)
            logger.error(f"Missing file: {file}")
    
    return missing_files

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    try:
        logger.info("Loading model artifacts...")
        
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return None, None, None
        
        # Check files first
        missing_files = verify_model_files()
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return None, None, None
        
        logger.info("Loading encoder model...")
        encoder = load_model("model/movie_recommender.h5", compile=False)
        
        logger.info("Loading TF-IDF vectorizer...")
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        
        logger.info("Loading movies dataframe...")
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        
        logger.info(f"✓ All artifacts loaded successfully")
        logger.info(f"  - Movies in dataset: {len(movies_df)}")
        logger.info(f"  - Model output shape: {encoder.output_shape}")
        
        return encoder, tfidf, movies_df
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

@st.cache_data
def generate_embeddings(_encoder, _tfidf, _movies_df):
    try:
        logger.info("Generating embeddings...")
        
        if _encoder is None or _tfidf is None or _movies_df is None:
            logger.error("Cannot generate embeddings: missing dependencies")
            return None
        
        if 'soup' not in _movies_df.columns:
            if 'genres' in _movies_df.columns:
                logger.info("Creating soup column from genres...")
                _movies_df['soup'] = _movies_df['genres'].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
                )
            else:
                logger.warning("No 'genres' or 'soup' column found")
                return None
        
        logger.info("Computing TF-IDF matrix...")
        tfidf_matrix = _tfidf.transform(_movies_df['soup'])
        
        logger.info("Generating embeddings using encoder...")
        movie_embeddings = _encoder.predict(tfidf_matrix.toarray(), verbose=0)
        
        logger.info(f"✓ Embeddings generated: shape {movie_embeddings.shape}")
        return movie_embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        logger.error(traceback.format_exc())
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Recommendation System</h1>
    <p style="color: white; margin: 0.5rem 0 0 0;">Discover your next favorite movie using AI</p>
</div>
""", unsafe_allow_html=True)

# Load Data
logger.info("=" * 50)
logger.info("LOADING ARTIFACTS AND EMBEDDINGS")
logger.info("=" * 50)

encoder_model, tfidf_vectorizer, movies_data = load_artifacts()

if encoder_model is None or tfidf_vectorizer is None or movies_data is None:
    logger.error("Failed to load artifacts - stopping app")
    st.error("❌ Failed to load model artifacts. Please check the logs:")
    st.info("Missing files or import error detected during startup.")
    st.stop()

movie_embeddings = generate_embeddings(encoder_model, tfidf_vectorizer, movies_data)

if movie_embeddings is None:
    logger.error("Failed to generate embeddings - stopping app")
    st.error("❌ Could not generate embeddings. Please check the logs:")
    st.info("Model files may be corrupted or missing.")
    st.stop()

logger.info("=" * 50)
logger.info("✓ ALL SYSTEMS GO - APP READY")
logger.info("=" * 50)

# Main Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎯 Find Similar Movies")
    
    # Movie selection
    try:
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
                    logger.info(f"Getting recommendations for: {selected}")
                    with st.spinner("Finding matches..."):
                        recommendations = recommend(selected, movies_data, movie_embeddings)[:num_recs]
                    
                    if recommendations:
                        logger.info(f"✓ Found {len(recommendations)} recommendations")
                        st.success(f"✅ Found {len(recommendations)} recommendations!")
                        st.markdown("### 🎥 Recommended Movies")
                        
                        for i, movie in enumerate(recommendations, 1):
                            st.markdown(f"""
                            <div class="movie-card">
                                <strong>{i}.</strong> {movie}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        logger.warning(f"No recommendations found for: {selected}")
                        st.warning("⚠️ No recommendations found.")
                except Exception as e:
                    logger.error(f"Error getting recommendations: {e}")
                    logger.error(traceback.format_exc())
                    st.error(f"❌ Error getting recommendations: {e}")
    except Exception as e:
        logger.error(f"Error in main interface: {e}")
        logger.error(traceback.format_exc())
        st.error(f"❌ Error in main interface: {e}")

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Total Movies", f"{len(movies_data):,}")
    st.metric("Embedding Dim", movie_embeddings.shape[1])
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.info("This app uses neural embeddings to find similar movies based on their descriptions and genres.")
        st.markdown("---")
        st.markdown("**Status:** ✅ Running")
        st.markdown(f"**Models Loaded:** {encoder_model is not None}")
        st.markdown(f"**Embeddings Ready:** {movie_embeddings is not None}")

    
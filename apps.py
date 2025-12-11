import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from recommender import recommend 
from pymongo import MongoClient
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f4068;
        --secondary-color: #e43f5a;
        --accent-color: #f9a826;
        --bg-dark: #162447;
        --bg-light: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1f4068 0%, #162447 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: #f8f9fa;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Movie card styling */
    .movie-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        border-left: 4px solid var(--accent-color);
    }
    
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .movie-title {
        color: var(--primary-color);
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--bg-light);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- MongoDB Setup ---
@st.cache_resource
def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/moviesdb")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        return client, client.get_database()
    except Exception as e:
        st.sidebar.warning(f"⚠️ MongoDB offline: {str(e)[:50]}...")
        return None, None

# --- MongoDB Helper Functions ---
def save_recommendation_to_db(mongo_db, selected_movie, recommendations):
    """Save user recommendations to MongoDB"""
    if mongo_db is None:
        return False
    try:
        recommendations_collection = mongo_db.get_collection("recommendations")
        doc = {
            "timestamp": datetime.now(),
            "selected_movie": selected_movie,
            "recommendations": list(recommendations),
            "source": "streamlit_app",
            "user_session_id": st.session_state.get("session_id", "unknown")
        }
        recommendations_collection.insert_one(doc)
        return True
    except Exception as e:
        return False

def get_recommendation_history(mongo_db, limit=10):
    """Retrieve recent recommendations from MongoDB"""
    if mongo_db is None:
        return []
    try:
        recommendations_collection = mongo_db.get_collection("recommendations")
        history = list(recommendations_collection.find().sort("timestamp", -1).limit(limit))
        return history
    except Exception as e:
        return []

def get_training_metrics(mongo_db):
    """Retrieve training metrics from MongoDB"""
    if mongo_db is None:
        return []
    try:
        training_logs_collection = mongo_db.get_collection("training_logs")
        metrics = list(training_logs_collection.find().sort("timestamp", -1).limit(1))
        return metrics
    except Exception as e:
        return []

# --- Caching to load resources only once ---
@st.cache_resource
def load_artifacts():
    """Loads the machine learning model and vectorizer from disk."""
    try:
        encoder = load_model("model/movie_recommender.h5")
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        return encoder, tfidf, movies_df
    except FileNotFoundError:
        st.error("❌ Model artifacts not found. Please train the model first.")
        return None, None, None

@st.cache_data
def generate_embeddings(_encoder, _tfidf, _movies_df):
    """Generates and caches movie embeddings."""
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
    movie_embeddings = _encoder.predict(tfidf_matrix.toarray())
    return movie_embeddings

# --- UI Helper Functions ---
def display_movie_card(rank, movie_title, show_poster=True):
    """Display a styled movie card"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if show_poster:
            # Use placeholder movie poster
            st.image(f"https://picsum.photos/seed/{hash(movie_title)}/150/200", 
                    use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="movie-card">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; width: 40px; height: 40px; border-radius: 50%; 
                           display: flex; align-items: center; justify-content: center; 
                           font-weight: bold; font-size: 1.2rem;">
                    {rank}
                </div>
                <div class="movie-title">{movie_title}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_metric_card(label, value, icon):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---

# Header
st.markdown("""
<div class="main-header">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Powered by Deep Learning | Discover your next favorite movie</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())
if "recommendation_count" not in st.session_state:
    st.session_state.recommendation_count = 0

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Number of recommendations slider
    num_recommendations = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    st.markdown("---")
    
    # Display session info
    st.markdown("### 📊 Session Info")
    st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")
    st.info(f"**Recommendations Made:** {st.session_state.recommendation_count}")
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system uses a **5-layer neural network autoencoder** 
    to analyze movie genres and generate personalized recommendations 
    using cosine similarity on learned embeddings.
    
    **Tech Stack:**
    - TensorFlow/Keras
    - Streamlit
    - MongoDB
    - Scikit-learn
    """)

# Mongo client and DB
mongo_client, mongo_db = get_mongo_client()

# Load artifacts and generate embeddings
encoder_model, tfidf_vectorizer, movies_data = load_artifacts()
movie_embeddings = generate_embeddings(encoder_model, tfidf_vectorizer, movies_data)

if movies_data is None or encoder_model is None or tfidf_vectorizer is None:
    st.error("❌ Could not load required model files. Please train the model first.")
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Get Recommendations", 
    "📜 History", 
    "📊 Analytics",
    "🔧 Model Info"
])

# Tab 1: Get Recommendations
with tab1:
    st.markdown("### 🎬 Find Your Next Movie")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if movies_data is not None and movie_embeddings is not None:
            movie_list = sorted(movies_data['title'].values)
            
            # Search box
            search_query = st.text_input(
                "🔍 Search for a movie",
                placeholder="Type to search...",
                help="Start typing to filter movies"
            )
            
            # Filter movies based on search
            if search_query:
                filtered_movies = [m for m in movie_list if search_query.lower() in m.lower()]
            else:
                filtered_movies = movie_list
            
            selected_movie = st.selectbox(
                "Choose a movie you like:",
                filtered_movies,
                index=0 if filtered_movies else 0
            )
            
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("🔮 Finding perfect matches..."):
                    recommendations = recommend(selected_movie, movies_data, movie_embeddings)
                    
                    # Limit to user-selected number
                    recommendations = recommendations[:num_recommendations]
                
                if recommendations:
                    # Save to MongoDB
                    save_recommendation_to_db(mongo_db, selected_movie, recommendations)
                    st.session_state.recommendation_count += 1
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="margin: 0; color: #2e7d32;">✅ Found {len(recommendations)} recommendations!</h3>
                        <p style="margin: 0.5rem 0 0 0;">Based on your selection: <strong>{selected_movie}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### 🎥 Recommended Movies")
                    
                    # Display recommendations as cards
                    for i, movie in enumerate(recommendations, 1):
                        display_movie_card(i, movie)
                else:
                    st.warning("⚠️ Could not find recommendations for this movie.")
        else:
            st.error("❌ Embeddings could not be generated.")
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        if movies_data is not None:
            create_metric_card("Movies", len(movies_data), "🎬")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if movie_embeddings is not None:
                create_metric_card("Dimensions", movie_embeddings.shape[1], "🔢")

# Tab 2: History
with tab2:
    st.markdown("### 📜 Recommendation History")
    
    if mongo_db is not None:
        # Controls
        col1, col2 = st.columns([3, 1])
        with col1:
            history_limit = st.slider("Number of records to display", 5, 50, 20)
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        history = get_recommendation_history(mongo_db, limit=history_limit)
        
        if history:
            # Create DataFrame
            history_df = pd.DataFrame([{
                "Timestamp": doc.get("timestamp", "N/A"),
                "Selected Movie": doc.get("selected_movie", "N/A"),
                "Recommendations": ", ".join(doc.get("recommendations", [])[:3]) + "..."
            } for doc in history])
            
            # Display as interactive table
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"recommendation_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📭 No recommendation history yet. Start by getting some recommendations!")
    else:
        st.warning("⚠️ MongoDB is not connected. History feature unavailable.")

# Tab 3: Analytics
with tab3:
    st.markdown("### 📊 Analytics Dashboard")
    
    if mongo_db is not None:
        history = get_recommendation_history(mongo_db, limit=100)
        
        if history:
            # Create analytics
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Recommendations over time
                daily_counts = df.groupby('date').size().reset_index(name='count')
                fig1 = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title='Recommendations Over Time',
                    labels={'date': 'Date', 'count': 'Number of Recommendations'}
                )
                fig1.update_traces(line_color='#667eea', line_width=3)
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Top searched movies
                top_movies = df['selected_movie'].value_counts().head(10)
                fig2 = px.bar(
                    x=top_movies.values,
                    y=top_movies.index,
                    orientation='h',
                    title='Top 10 Most Searched Movies',
                    labels={'x': 'Search Count', 'y': 'Movie'}
                )
                fig2.update_traces(marker_color='#764ba2')
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # Summary metrics
            st.markdown("### 📈 Summary Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card("Total Searches", len(df), "🔍")
            with col2:
                create_metric_card("Unique Movies", df['selected_movie'].nunique(), "🎬")
            with col3:
                create_metric_card("Avg/Day", f"{len(df)/max(1, (df['timestamp'].max() - df['timestamp'].min()).days):.1f}", "📅")
            with col4:
                create_metric_card("Sessions", df['user_session_id'].nunique(), "👥")
        else:
            st.info("📊 No data available for analytics yet.")
    else:
        st.warning("⚠️ MongoDB is not connected. Analytics unavailable.")

# Tab 4: Model Info
with tab4:
    st.markdown("### 🔧 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Dataset Info")
        st.metric("Total Movies", len(movies_data) if movies_data is not None else 0)
        
        if tfidf_vectorizer is not None:
            try:
                num_features = len(tfidf_vectorizer.get_feature_names_out())
                st.metric("TF-IDF Features", f"{num_features:,}")
            except:
                st.metric("TF-IDF Features", "N/A")
    
    with col2:
        st.markdown("#### 🧠 Model Architecture")
        if movie_embeddings is not None:
            st.metric("Embedding Dimension", movie_embeddings.shape[1])
            st.metric("Model Type", "Autoencoder")
    
    with col3:
        st.markdown("#### 📈 Usage Stats")
        if mongo_db is not None:
            try:
                count = mongo_db.get_collection('recommendations').count_documents({})
                st.metric("Total Recommendations", f"{count:,}")
            except:
                st.metric("Total Recommendations", "N/A")
    
    st.markdown("---")
    
    # Training metrics
    if mongo_db is not None:
        metrics = get_training_metrics(mongo_db)
        if metrics:
            st.markdown("### 📉 Latest Training Metrics")
            latest = metrics[0]
            
            col1, col2 = st.columns(2)
            with col1:
                try:
                    train_loss = latest.get('train_loss', 'N/A')
                    if isinstance(train_loss, (int, float)):
                        st.metric("Training Loss", f"{train_loss:.4f}")
                    else:
                        st.metric("Training Loss", train_loss)
                except:
                    st.metric("Training Loss", "N/A")
            
            with col2:
                try:
                    val_loss = latest.get('val_loss', 'N/A')
                    if isinstance(val_loss, (int, float)):
                        st.metric("Validation Loss", f"{val_loss:.4f}")
                    else:
                        st.metric("Validation Loss", val_loss)
                except:
                    st.metric("Validation Loss", "N/A")
    
    # Model details expander
    with st.expander("🔍 View Model Details"):
        st.markdown("""
        **Architecture:**
        - 5-layer neural network autoencoder
        - Activation functions: ReLU, Tanh, ELU, SELU, Sigmoid
        - Input: TF-IDF vectorized movie genres
        - Output: Compressed embeddings for similarity matching
        
        **Recommendation Algorithm:**
        - Cosine similarity on learned embeddings
        - Top-K nearest neighbors selection
        - Real-time inference
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎬 Movie Recommendation System | Built with Streamlit & TensorFlow</p>
    <p style="font-size: 0.9rem;">Powered by Deep Learning • Data Science Project</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from recommender import recommend 
from pymongo import MongoClient
import os
from datetime import datetime

# --- MongoDB Setup ---
@st.cache_resource
def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/moviesdb")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')  # Test connection
        return client, client.get_database()
    except Exception as e:
        st.warning(f"MongoDB connection failed: {e}. Running in offline mode.")
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
        st.warning(f"Could not save recommendations to MongoDB: {e}")
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
        st.warning(f"Could not retrieve recommendation history: {e}")
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
        st.warning(f"Could not retrieve training metrics: {e}")
        return []

def get_movies_from_db(mongo_db):
    """Retrieve movies from MongoDB"""
    if mongo_db is None:
        return []
    try:
        movies_collection = mongo_db.get_collection("movies")
        movies = list(movies_collection.find({}, {"_id": 0}))
        return movies
    except Exception as e:
        st.warning(f"Could not retrieve movies from MongoDB: {e}")
        return []

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
    
    # Check if 'soup' column exists, if not create it from genres
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

# --- Main App ---

st.title("🎬 Movie Recommendation System")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())

# Mongo client and DB
mongo_client, mongo_db = get_mongo_client()

# Load artifacts and generate embeddings
encoder_model, tfidf_vectorizer, movies_data = load_artifacts()
movie_embeddings = generate_embeddings(encoder_model, tfidf_vectorizer, movies_data)

if movies_data is None or encoder_model is None or tfidf_vectorizer is None:
    st.error("❌ Could not load required model files. Please train the model first by running the notebook.")
    st.stop()

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Get Recommendations", "Recommendation History", "Model Info"])

with tab1:
    if movies_data is not None and movie_embeddings is not None:
        movie_list = movies_data['title'].values
        selected_movie = st.selectbox("Choose a movie you like:", movie_list)

        if st.button("Recommend Movies", type="primary"):
            with st.spinner("Finding recommendations..."):
                # The recommend function is now imported
                recommendations = recommend(selected_movie, movies_data, movie_embeddings)
            
            if recommendations:
                # Save query and response to MongoDB
                save_recommendation_to_db(mongo_db, selected_movie, recommendations)
                st.success(f"✅ Found {len(recommendations)} recommendations!")

                st.subheader(f"Because you watched '{selected_movie}', you might also like:")
                for i, movie in enumerate(recommendations, 1):
                    st.write(f"{i}. {movie}")
            else:
                st.warning("Could not find any recommendations for the selected movie.")
    else:
        st.error("The application could not start because embeddings could not be generated.")

with tab2:
    st.subheader("Recent Recommendation History")
    if mongo_db is not None:
        history = get_recommendation_history(mongo_db, limit=20)
        
        if history:
            history_df = pd.DataFrame([{
                "Timestamp": doc.get("timestamp", "N/A"),
                "Selected Movie": doc.get("selected_movie", "N/A"),
                "Recommendations": ", ".join(doc.get("recommendations", []))
            } for doc in history])
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("📊 Download History as CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="recommendation_history.csv",
                    mime="text/csv"
                )
        else:
            st.info("No recommendation history yet.")
    else:
        st.info("MongoDB is not connected. History not available.")

with tab3:
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Movies in Database:** {len(movies_data) if movies_data is not None else 0}")
        if tfidf_vectorizer is not None:
            try:
                num_features = len(tfidf_vectorizer.get_feature_names_out())
                st.write(f"**TF-IDF Features:** {num_features}")
            except:
                st.write("**TF-IDF Features:** N/A")
    
    with col2:
        st.write(f"**Embedding Dimension:** {movie_embeddings.shape[1] if movie_embeddings is not None else 0}")
        if mongo_db is not None:
            try:
                count = mongo_db.get_collection('recommendations').count_documents({})
                st.write(f"**Total Recommendations Made:** {count}")
            except:
                st.write("**Total Recommendations Made:** N/A")
    
    # Show latest training metrics
    if mongo_db is not None:
        metrics = get_training_metrics(mongo_db)
        if metrics:
            st.subheader("Latest Training Metrics")
            latest = metrics[0]
            try:
                train_loss = latest.get('train_loss', 'N/A')
                val_loss = latest.get('val_loss', 'N/A')
                st.metric("Train Loss", f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else train_loss)
                st.metric("Validation Loss", f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else val_loss)
            except Exception as e:
                st.warning(f"Could not display metrics: {e}")


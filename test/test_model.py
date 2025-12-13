import pytest
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recommender import recommend

@pytest.fixture(scope="module")
def artifacts():
    try:
        encoder = load_model("model/movie_recommender.h5")
        tfidf = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
        movies_df = pickle.load(open("model/movies_df.pkl", "rb"))
        return {"encoder": encoder, "tfidf": tfidf, "movies_df": movies_df}
    except FileNotFoundError:
        pytest.fail("One or more model artifacts are missing from the 'model/' directory.")

@pytest.fixture(scope="module")
def movie_embeddings(artifacts):
    encoder = artifacts["encoder"]
    tfidf = artifacts["tfidf"]
    movies_df = artifacts["movies_df"]
    
    tfidf_matrix = tfidf.transform(movies_df['soup'])
    embeddings = encoder.predict(tfidf_matrix.toarray())
    return embeddings

def test_artifact_loading(artifacts):
    assert artifacts["encoder"] is not None
    assert artifacts["tfidf"] is not None
    assert isinstance(artifacts["movies_df"], pd.DataFrame)
    assert not artifacts["movies_df"].empty

def test_data_columns(artifacts):
    required_columns = {'title', 'soup'}
    assert required_columns.issubset(artifacts["movies_df"].columns)

def test_recommendation_function_logic():
    # Create dummy data for a simple unit test
    dummy_titles = ['Movie A', 'Movie B', 'Movie C']
    dummy_df = pd.DataFrame({'title': dummy_titles})
    dummy_embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]])
    
    recommendations = recommend('Movie A', dummy_df, dummy_embeddings)
    
    assert isinstance(recommendations, list)
    assert 'Movie B' in recommendations
    assert 'Movie A' not in recommendations

def test_full_recommendation_flow(artifacts, movie_embeddings):
    movies_df = artifacts["movies_df"]
    test_movie = "Avatar"
    
    if test_movie not in movies_df['title'].values:
        pytest.skip(f"'{test_movie}' not found in the dataset.")

    recommendations = recommend(test_movie, movies_df, movie_embeddings, num_recommendations=15)

    assert isinstance(recommendations, list)
    assert len(recommendations) == 15
    assert all(isinstance(m, str) for m in recommendations)
    assert test_movie not in recommendations

    


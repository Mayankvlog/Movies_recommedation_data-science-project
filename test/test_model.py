# test/test_model.py

import pytest
import numpy as np
import tensorflow as tf
import pickle
import os


@pytest.fixture(scope="module")
def encoder_model():
    """Load the trained encoder model."""
    model_path = 'model/movie_recommender.h5'
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    return tf.keras.models.load_model(model_path)


@pytest.fixture(scope="module")
def tfidf_vectorizer():
    """Load the trained TF-IDF vectorizer."""
    vectorizer_path = 'model/tfidf_vectorizer.pkl'
    assert os.path.exists(vectorizer_path), f"Vectorizer file not found at {vectorizer_path}"
    with open(vectorizer_path, 'rb') as f:
        return pickle.load(f)


def test_model_loading(encoder_model):
    """Test that the model loads correctly and is a Keras Model."""
    assert isinstance(encoder_model, tf.keras.Model)
    assert encoder_model.name == "Encoder"


def test_vectorizer_loading(tfidf_vectorizer):
    """Test that the vectorizer loads correctly and has the transform method."""
    assert hasattr(tfidf_vectorizer, 'transform')
    assert hasattr(tfidf_vectorizer, 'get_feature_names_out')


def test_full_pipeline_prediction(encoder_model, tfidf_vectorizer):
    """Test the full pipeline from text input to embedding output."""
    sample_soup = "Action Adventure Fantasy"
    
    # Transform the input text
    vectorized_input = tfidf_vectorizer.transform([sample_soup]).toarray()
    
    # Check that we got a valid 2D array
    assert vectorized_input.ndim == 2
    assert vectorized_input.shape[0] == 1  # One sample
    
    # Get the actual number of features from the vectorizer (dynamic check)
    expected_features = len(tfidf_vectorizer.get_feature_names_out())
    assert vectorized_input.shape[1] == expected_features
    
    # Generate embedding
    embedding = encoder_model.predict(vectorized_input, verbose=0)
    
    # Check embedding shape (32-dimensional output based on model architecture)
    assert embedding.shape == (1, 32)
    
    # Check that embedding contains valid finite numbers
    assert np.all(np.isfinite(embedding))

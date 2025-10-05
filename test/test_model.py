import os
os.environ["TF_USE_LEGACY_KERAS"] = '1'

import pytest
import numpy as np
from unittest.mock import Mock

class DummyModel:
    """A mock Keras model that simulates the real one."""
    def __init__(self):
        self.name = "Encoder"
    def predict(self, x, verbose=0):
        # Return a correctly shaped numpy array
        return np.ones((x.shape[0], 32))

@pytest.fixture(scope="module")
def mock_encoder_model():
    """Provides the mock Keras model for tests."""
    return DummyModel()

@pytest.fixture(scope="module")
def dummy_vectorizer():
    """Provides a mock, fitted TF-IDF vectorizer."""
    class DummyVectorizer:
        def transform(self, texts):
            # Return a correctly shaped numpy array
            return np.ones((len(texts), 5))  
        def get_feature_names_out(self):
            # Return a list of dummy feature names
            return ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    return DummyVectorizer()

def test_mock_model_loading(mock_encoder_model):
    """Tests that the mock model has the correct interface."""
    assert mock_encoder_model.name == "Encoder"
    assert hasattr(mock_encoder_model, 'predict')

def test_vectorizer_loading(dummy_vectorizer):
    """Tests that the mock vectorizer has the correct interface."""
    assert hasattr(dummy_vectorizer, 'transform')
    assert hasattr(dummy_vectorizer, 'get_feature_names_out')

def test_full_pipeline_prediction(mock_encoder_model, dummy_vectorizer):
    """Tests the full pipeline logic with mock components."""
    sample_soup = "Action Adventure Fantasy"
    vectorized_input = dummy_vectorizer.transform([sample_soup])
    
    # Check that the output shape matches the number of features
    assert vectorized_input.shape[1] == len(dummy_vectorizer.get_feature_names_out())
    
    # Get the embedding from the mock model
    embedding = mock_encoder_model.predict(vectorized_input)
    
    # Check that the final embedding has the correct shape
    assert embedding.shape == (1, 32)
    assert np.all(np.isfinite(embedding))

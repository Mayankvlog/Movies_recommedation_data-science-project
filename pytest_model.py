# Create a new file named test_model.py

import pytest
import pickle
import numpy as np
from tensorflow.keras.models import load_model

@pytest.fixture
def model_and_preprocessors():
    """Loads the model and preprocessing objects."""
    model = load_model('movie_recommender_rnn.h5')
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    return model, mlb

def test_model_prediction(model_and_preprocessors):
    """Tests if the model can make a prediction without crashing."""
    model, mlb = model_and_preprocessors
    
    # Create a dummy input for one movie with genres 'Action' and 'Adventure'
    dummy_genres = [['Action', 'Adventure']]
    dummy_input_encoded = mlb.transform(dummy_genres)
    
    # Reshape for RNN
    dummy_input_rnn = np.reshape(dummy_input_encoded, (1, 1, dummy_input_encoded.shape[1]))

    # Make a prediction
    prediction = model.predict(dummy_input_rnn)

    # Check if prediction is a single value between 0 and 1
    assert prediction.shape == (1, 1)
    assert 0 <= prediction[0][0] <= 1

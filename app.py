# Create a new file named app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load all necessary files
@st.cache(allow_output_mutation=True)
def load_assets():
    model = load_model('movie_recommender_rnn.h5')
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df = pd.read_csv('tmdb_5000_movies.csv')
    df['genre_names'] = df['genres'].apply(lambda x: [g['name'] for g in json.loads(x)])
    return model, mlb, scaler, df

model, mlb, scaler, df = load_assets()

st.title("🎬 Advanced Movie Recommender")
st.write("Select genres to get a predicted rating and movie recommendations.")

# Genre selection in the sidebar
st.sidebar.header("Select Genres")
available_genres = sorted(mlb.classes_)
selected_genres = st.sidebar.multiselect("Choose one or more genres:", available_genres)

if st.button("Get Recommendation"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        # 1. Encode selected genres
        input_genres_encoded = mlb.transform([selected_genres])
        
        # 2. Reshape for RNN model
        input_rnn = np.reshape(input_genres_encoded, (1, 1, input_genres_encoded.shape[1]))

        # 3. Predict the rating
        predicted_scaled_rating = model.predict(input_rnn)[0][0]
        
        # 4. Inverse transform the rating to the original 0-10 scale
        predicted_rating = scaler.inverse_transform([[predicted_scaled_rating]])[0][0]

        st.subheader(f"Predicted Rating for a movie with these genres: **{predicted_rating:.2f} / 10**")

        # 5. Find movies that match the criteria
        def is_subset(list1, list2):
            return set(list1).issubset(set(list2))

        # Find movies containing ALL selected genres
        recommended_movies = df[df['genre_names'].apply(lambda x: is_subset(selected_genres, x))]

        # Sort by vote average and get top 10
        recommended_movies = recommended_movies.sort_values(by='vote_average', ascending=False).head(10)
        
        st.subheader("Top Recommended Movies:")
        if recommended_movies.empty:
            st.write("No movies found with that exact combination of genres.")
        else:
            for index, row in recommended_movies.iterrows():
                st.write(f"**{row['title']}** - Rating: {row['vote_average']}/10")


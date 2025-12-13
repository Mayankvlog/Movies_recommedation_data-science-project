import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend(movie_title, movies_df, movie_embeddings):
    """
    Recommends 10 similar movies based on cosine similarity of their embeddings.
    """
    if movie_embeddings is None or movies_df.empty:
        return []

    # Find the index of the selected movie
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        # Return an empty list if the movie is not found
        return []

    # Get the embedding of the selected movie
    movie_embedding = movie_embeddings[movie_index]

    # Calculate cosine similarity
    similarities = cosine_similarity([movie_embedding], movie_embeddings)[0]

    # Get indices of top 5 similar movies
    similar_movies_indices = np.argsort(similarities)[::-1][1:16]

    # Get the titles
    recommended_movies = movies_df.iloc[similar_movies_indices]['title'].tolist()
    
    return recommended_movies

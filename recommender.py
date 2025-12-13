import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend(movie_title, movies_df, movie_embeddings, num_recommendations=15):
    """
    Recommends similar movies based on cosine similarity of their embeddings.
    
    Args:
        movie_title: Title of the movie to find recommendations for
        movies_df: DataFrame containing movie data
        movie_embeddings: Embeddings of all movies
        num_recommendations: Number of recommendations to return (default: 15)
    
    Returns:
        List of recommended movie titles
    """
    if movie_embeddings is None or movies_df.empty:
        return []

    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        return []

    movie_embedding = movie_embeddings[movie_index]
    similarities = cosine_similarity([movie_embedding], movie_embeddings)[0]
    similar_movies_indices = np.argsort(similarities)[::-1][1:num_recommendations+1]
    recommended_movies = movies_df.iloc[similar_movies_indices]['title'].tolist()
    
    return recommended_movies

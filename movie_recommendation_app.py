import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load the MovieLens dataset
# movies = pd.read_csv('path/to/movies.csv')
#r
movies = pd.read_csv (r'D:\MyOwnPracticeProject\Data Science Carrer\Sardar Data Scientist\Projects\movie_recommendation_app\movies.csv')
# ratings = pd.read_csv('path/to/ratings.csv')
ratings = pd.read_csv(r'D:\MyOwnPracticeProject\Data Science Carrer\Sardar Data Scientist\Projects\movie_recommendation_app\ratings.csv')

# Merge movies and ratings data
movie_ratings = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_movie_ratings = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

# Fill missing values with 0
user_movie_ratings = user_movie_ratings.fillna(0)

# Build a Nearest Neighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
knn_model.fit(user_movie_ratings.values)

# Function to get movie recommendations
def get_movie_recommendations(movie_title):
    movie_idx = user_movie_ratings.columns.get_loc(movie_title)
    distances, indices = knn_model.kneighbors(user_movie_ratings.iloc[:, movie_idx].values.reshape(1, -1), n_neighbors=10 + 1)
    recommended_movies = []
    for i in range(1, len(distances.flatten())):
        recommended_movies.append({
            'title': user_movie_ratings.columns[indices.flatten()[i]],
            'distance': distances.flatten()[i]
        })
    return recommended_movies

# Streamlit app
st.title('Movie Recommendation System')

# Movie selection
selected_movie = st.selectbox('Select a movie:', movies['title'].values)

# Get recommendations
if st.button('Get Recommendations'):
    recommendations = get_movie_recommendations(selected_movie)
    st.subheader('Recommended Movies:')
    for movie in recommendations:
        st.write(f"{movie['title']} (Distance: {movie['distance']:.3f})")

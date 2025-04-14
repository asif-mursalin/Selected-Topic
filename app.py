import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
    <style>
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
    .movie-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .movie-year {
        color: #666;
        font-size: 14px;
    }
    .movie-genre {
        font-size: 12px;
        color: #444;
        margin-bottom: 5px;
    }
    .rating {
        color: #ff9900;
        font-weight: bold;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 32px;
        font-weight: bold;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load MovieLens dataset
@st.cache_data
def load_movie_data():
    """Load movie data from pickle file"""
    try:
        # Try to load pre-saved pickle data if available
        if os.path.exists('models/movies_data.pkl'):
            with open('models/movies_data.pkl', 'rb') as f:
                movies_df = pickle.load(f)
                return movies_df
        else:
            st.warning("Movie data file not found. Using sample data instead.")
            return create_sample_movie_data()
    except Exception as e:
        st.warning(f"Error loading movie data: {e}. Using sample data instead.")
        return create_sample_movie_data()

def create_sample_movie_data():
    """Create a sample movie dataset for demo purposes"""
    data = {
        'movie_id': list(range(1, 101)),
        'title': [f"Sample Movie {i} ({1990 + i % 30})" for i in range(1, 101)],
        'year': [1990 + i % 30 for i in range(1, 101)],
        'genre_names': [
            np.random.choice(['Action', 'Adventure', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Thriller'], 
                          size=np.random.randint(1, 4), replace=False).tolist()
            for _ in range(100)
        ]
    }
    return pd.DataFrame(data)

@st.cache_data
def load_ratings_data():
    """Load ratings data from CSV or create sample data"""
    try:
        # Try to load from CSV if available
        if os.path.exists('ml-100k/u.data'):
            ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', 
                                  names=['user_id', 'movie_id', 'rating', 'timestamp'])
            return ratings_df
        else:
            st.warning("Ratings data file not found. Using sample data instead.")
            return create_sample_ratings_data()
    except Exception as e:
        st.warning(f"Error loading ratings data: {e}. Using sample data instead.")
        return create_sample_ratings_data()

def create_sample_ratings_data():
    """Create a sample ratings dataset for demo purposes"""
    # Create sample data with 1000 ratings
    user_ids = np.random.choice(range(1, 100), 1000)
    movie_ids = np.random.choice(range(1, 101), 1000)
    ratings = np.random.choice([1, 2, 3, 4, 5], 1000, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    timestamps = np.random.randint(1000000000, 1100000000, 1000)
    
    return pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })

@st.cache_data
def load_model_info():
    """Load model info from pickle file"""
    try:
        if os.path.exists('models/model_info.pkl'):
            with open('models/model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                return model_info
        else:
            st.warning("Model info file not found. Using default values.")
            return {
                'model_type': 'SVD (default)',
                'rmse': 0.935,
                'precision': 0.36,
                'recall': 0.29
            }
    except Exception as e:
        st.warning(f"Error loading model info: {e}. Using default values.")
        return {
            'model_type': 'SVD (default)',
            'rmse': 0.935,
            'precision': 0.36,
            'recall': 0.29
        }

def get_movie_recommendations(user_id, movies_df, ratings_df, n=10):
    """Get movie recommendations for a specific user using pre-computed data
    or collaborative filtering approach"""
    
    # In a real implementation, this would use the pickled model
    # For this simplified version, we'll use a collaborative filtering approach
    
    # Get movies the user has already rated
    user_rated_movies = ratings_df[ratings_df['user_id'] == user_id]
    
    # If the user hasn't rated any movies, return popular movies
    if len(user_rated_movies) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
    
    # Get all users who rated the same movies as our target user
    similar_users = ratings_df[ratings_df['movie_id'].isin(user_rated_movies['movie_id'])]
    similar_users = similar_users[similar_users['user_id'] != user_id]['user_id'].unique()
    
    # If no similar users found, return popular movies
    if len(similar_users) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
    
    # Get movies rated by similar users that target user hasn't rated
    movies_to_recommend = ratings_df[
        (ratings_df['user_id'].isin(similar_users)) & 
        (~ratings_df['movie_id'].isin(user_rated_movies['movie_id']))
    ]
    
    # Calculate average rating for each movie
    movie_ratings = movies_to_recommend.groupby('movie_id')['rating'].agg(['mean', 'count'])
    movie_ratings.columns = ['avg_rating', 'count']
    
    # Filter out movies with fewer than 3 ratings
    movie_ratings = movie_ratings[movie_ratings['count'] >= 3]
    
    # Sort by average rating
    movie_ratings = movie_ratings.sort_values(by=['avg_rating', 'count'], ascending=False)
    
    # Get top-n movie IDs
    top_movie_ids = movie_ratings.index[:n].tolist()
    
    # Get movie details
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)]
    
    # Add predicted ratings
    movie_ratings_dict = movie_ratings['avg_rating'].to_dict()
    recommended_movies['predicted_rating'] = recommended_movies['movie_id'].map(movie_ratings_dict)
    
    # Sort by predicted rating
    recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)
    
    return recommended_movies

def get_popular_movies(movies_df, ratings_df, n=10):
    """Get popular movies based on average rating and number of ratings"""
    # Calculate average rating and rating count for each movie
    movie_stats = ratings_df.groupby('movie_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    # Flatten the column hierarchy
    movie_stats.columns = ['movie_id', 'avg_rating', 'rating_count']
    
    # Filter movies with at least 5 ratings
    popular_movies = movie_stats[movie_stats['rating_count'] >= 5]
    
    # Sort by average rating and count
    popular_movies = popular_movies.sort_values(by=['avg_rating', 'rating_count'], ascending=False)
    
    # Get top-n movie IDs
    top_movie_ids = popular_movies['movie_id'].head(n).tolist()
    
    # Get movie details
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)]
    
    # Add average rating
    movie_ratings = {row['movie_id']: row['avg_rating'] for _, row in popular_movies.iterrows()}
    recommended_movies['predicted_rating'] = recommended_movies['movie_id'].map(movie_ratings)
    
    # Sort by average rating
    recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)
    
    return recommended_movies

def get_movies_by_genre(movies_df, genre, n=10):
    """Get movies by genre"""
    # Filter movies by genre
    genre_movies = movies_df[movies_df['genre_names'].apply(lambda x: genre in x)]
    
    # If no movies found, return empty DataFrame
    if len(genre_movies) == 0:
        return pd.DataFrame()
    
    # Return top-n movies (or all if less than n)
    return genre_movies.head(n)

def get_movie_details(movies_df, ratings_df, movie_id):
    """Get detailed information about a movie"""
    # Get movie details
    movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
    
    # Get movie ratings
    movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating']
    
    # Calculate rating stats
    if len(movie_ratings) > 0:
        avg_rating = movie_ratings.mean()
        rating_count = len(movie_ratings)
        rating_dist = movie_ratings.value_counts().sort_index()
    else:
        avg_rating = 0
        rating_count = 0
        rating_dist = pd.Series([0] * 5, index=[1, 2, 3, 4, 5])
    
    return movie, avg_rating, rating_count, rating_dist

def extract_year_from_title(title):
    """Extract year from movie title"""
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return match.group(1)
    return "Unknown"

def display_movie_card(movie, col, show_prediction=False):
    """Display a movie card in a Streamlit column"""
    with col:
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-year">{movie['year'] if 'year' in movie else extract_year_from_title(movie['title'])}</div>
            <div class="movie-genre">{', '.join(movie['genre_names']) if 'genre_names' in movie and isinstance(movie['genre_names'], list) else 'No genres'}</div>
            {f'<div class="rating">Predicted rating: {movie["predicted_rating"]:.1f}⭐</div>' if show_prediction and 'predicted_rating' in movie else ''}
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"View Details", key=f"view_{movie['movie_id']}"):
            st.session_state.selected_movie = movie['movie_id']
            st.session_state.page = "movie_details"
            st.rerun()

def user_authentication():
    """Simulated user authentication"""
    st.sidebar.title("User Authentication")
    
    # Check if user is already logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # If not logged in, show login form
    if not st.session_state.logged_in:
        # For demo purposes, let user select from a list of user IDs
        user_id = st.sidebar.selectbox(
            "Select User ID (for demo)",
            options=list(range(1, 11)),
            index=0,
            key="login_user_id"
        )
        
        if st.sidebar.button("Login"):
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.sidebar.success(f"Logged in as User {user_id}")
            st.rerun()
    else:
        st.sidebar.success(f"Logged in as User {st.session_state.user_id}")
        
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()

def display_home_page(movies_df, ratings_df, model_info):
    """Display the home page with recommendations"""
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">MovieLens Recommender</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # If user is logged in, show personalized recommendations
    if st.session_state.logged_in:
        st.subheader(f"Personalized Recommendations for User {st.session_state.user_id}")
        
        # Get recommendations
        recommended_movies = get_movie_recommendations(
            st.session_state.user_id, movies_df, ratings_df, n=10
        )
        
        # Display recommendations in a grid (3 columns)
        cols = st.columns(3)
        for i, (_, movie) in enumerate(recommended_movies.iterrows()):
            display_movie_card(movie, cols[i % 3], show_prediction=True)
    
    # Popular Movies Section
    st.subheader("Popular Movies")
    popular_movies = get_popular_movies(movies_df, ratings_df, n=6)
    
    # Display popular movies in a grid (3 columns)
    cols = st.columns(3)
    for i, (_, movie) in enumerate(popular_movies.iterrows()):
        display_movie_card(movie, cols[i % 3])
    
    # Browse by Genre Section
    st.subheader("Browse by Genre")
    
    # Get all unique genres
    all_genres = []
    for genres in movies_df['genre_names']:
        if isinstance(genres, list):
            all_genres.extend(genres)
    all_genres = sorted(list(set(all_genres)))
    
    # Create selectbox for genres
    selected_genre = st.selectbox("Select a genre", all_genres)
    
    # Get movies by genre
    genre_movies = get_movies_by_genre(movies_df, selected_genre, n=6)
    
    # Display genre movies in a grid (3 columns)
    if len(genre_movies) > 0:
        cols = st.columns(3)
        for i, (_, movie) in enumerate(genre_movies.iterrows()):
            display_movie_card(movie, cols[i % 3])
    else:
        st.info(f"No movies found in the {selected_genre} genre.")
    
    # Show model information in expander
    with st.expander("Model Information"):
        st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.write(f"**RMSE:** {model_info.get('rmse', 'N/A'):.4f}")
        st.write(f"**Precision:** {model_info.get('precision', 'N/A'):.4f}")
        st.write(f"**Recall:** {model_info.get('recall', 'N/A'):.4f}")

def display_movie_details_page(movies_df, ratings_df):
    """Display detailed information about a selected movie"""
    # Get movie ID from session state
    movie_id = st.session_state.selected_movie
    
    # Get movie details
    movie, avg_rating, rating_count, rating_dist = get_movie_details(movies_df, ratings_df, movie_id)
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    # Movie title and basic info
    st.title(movie['title'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Movie Information")
        st.write(f"**Year:** {movie['year'] if 'year' in movie else 'Unknown'}")
        st.write(f"**Genres:** {', '.join(movie['genre_names']) if 'genre_names' in movie and isinstance(movie['genre_names'], list) else 'No genres'}")
        
        # Description placeholder (would be real data in a full implementation)
        st.write("**Description:**")
        st.write("This is a placeholder description for the movie. In a real implementation, this would contain the actual movie plot or synopsis.")
    
    with col2:
        st.subheader("Rating Information")
        st.write(f"**Average Rating:** {avg_rating:.1f}/5.0")
        st.write(f"**Number of Ratings:** {rating_count}")
        
        # Rating distribution
        if rating_count > 0:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(rating_dist.index, rating_dist.values, color='skyblue')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_xticks([1, 2, 3, 4, 5])
            st.pyplot(fig)
    
    # User rating section
    if st.session_state.logged_in:
        st.subheader("Rate This Movie")
        user_rating = st.slider("Your Rating", 1.0, 5.0, 3.0, 0.5)
        
        if st.button("Submit Rating"):
            st.success(f"Thank you for rating '{movie['title']}' with {user_rating} stars!")
            # In a real implementation, this would update the ratings data

def display_search_page(movies_df):
    """Display search page for finding movies"""
    st.title("Search Movies")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    # Search query
    search_query = st.text_input("Search for movies by title")
    
    if search_query:
        # Search movies by title
        search_results = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
        
        if len(search_results) > 0:
            st.subheader(f"Found {len(search_results)} results for '{search_query}'")
            
            # Display search results in a grid (3 columns)
            cols = st.columns(3)
            for i, (_, movie) in enumerate(search_results.head(12).iterrows()):
                display_movie_card(movie, cols[i % 3])
        else:
            st.info(f"No movies found matching '{search_query}'")

def main():
    """Main function for the Streamlit app"""
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    
    # User authentication in sidebar
    user_authentication()
    
    # Navigation in sidebar
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("Search Movies"):
        st.session_state.page = "search"
        st.rerun()
    
    # Load data
    movies_df = load_movie_data()
    ratings_df = load_ratings_data()
    model_info = load_model_info()
    
    # Display info about data
    with st.sidebar.expander("Dataset Information"):
        st.write(f"**Movies:** {len(movies_df)}")
        st.write(f"**Ratings:** {len(ratings_df)}")
        st.write(f"**Users:** {ratings_df['user_id'].nunique()}")
    
    # Route to appropriate page
    if st.session_state.page == "home":
        display_home_page(movies_df, ratings_df, model_info)
    elif st.session_state.page == "movie_details":
        display_movie_details_page(movies_df, ratings_df)
    elif st.session_state.page == "search":
        display_search_page(movies_df)

if __name__ == "__main__":
    main()
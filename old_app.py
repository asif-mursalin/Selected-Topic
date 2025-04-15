import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from PIL import Image
import time

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
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        transition: transform 0.2s;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 5px;
        color: #1e3a8a;
    }
    .movie-year {
        color: #666;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .movie-genre {
        font-size: 13px;
        color: #444;
        margin-bottom: 10px;
    }
    .rating {
        color: #ff9900;
        font-weight: bold;
        font-size: 16px;
        margin-top: 8px;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 30px;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
    }
    .header-title {
        font-size: 36px;
        font-weight: bold;
        margin: 0;
        color: #1e3a8a;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #2c5282;
        border-left: 4px solid #4299e1;
        padding-left: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #4a5568;
        font-weight: 400;
    }
    .tag {
        display: inline-block;
        background-color: #e2e8f0;
        border-radius: 4px;
        padding: 3px 8px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #4a5568;
    }
    .view-details-btn {
        background-color: #4299e1;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 14px;
        cursor: pointer;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .view-details-btn:hover {
        background-color: #3182ce;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== DATA LOADING FUNCTIONS =====

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
    # Create genres for sample movies
    genres_list = [
        ['Action', 'Adventure', 'Sci-Fi'],
        ['Comedy', 'Romance'],
        ['Drama', 'Thriller'],
        ['Animation', 'Family'],
        ['Horror', 'Mystery'],
        ['Action', 'Comedy'],
        ['Drama', 'Romance'],
        ['Sci-Fi', 'Thriller'],
        ['Documentary'],
        ['Fantasy', 'Adventure']
    ]
    
    # Create sample movie data
    data = {
        'movie_id': list(range(1, 101)),
        'title': [f"Sample Movie {i}" for i in range(1, 101)],
        'year': [1990 + i % 30 for i in range(1, 101)],
        'genre_names': [genres_list[i % 10] for i in range(100)]
    }
    
    # Add movie titles with year in parentheses
    for i in range(len(data['title'])):
        data['title'][i] = f"{data['title'][i]} ({data['year'][i]})"
    
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
    # Create sample data with 5000 ratings (more ratings for better recommendations)
    user_ids = np.random.choice(range(1, 100), 5000)
    movie_ids = np.random.choice(range(1, 101), 5000)
    ratings = np.random.choice([1, 2, 3, 4, 5], 5000, p=[0.1, 0.2, 0.3, 0.25, 0.15])
    timestamps = np.random.randint(1000000000, 1100000000, 5000)
    
    return pd.DataFrame({
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })

@st.cache_resource
def load_model():
    """Load the SVD model from pickle file"""
    try:
        # Try to load SVD model if available
        if os.path.exists('models/svd_model.pkl'):
            with open('models/svd_model.pkl', 'rb') as f:
                model = pickle.load(f)
                return model
        # Fallback to best model
        elif os.path.exists('models/best_model.pkl'):
            with open('models/best_model.pkl', 'rb') as f:
                model = pickle.load(f)
                return model
        else:
            st.warning("Model file not found. Using default recommendations.")
            return None
    except Exception as e:
        st.warning(f"Error loading model: {e}. Using default recommendations.")
        return None

@st.cache_data
def load_model_info():
    """Load model info from pickle file"""
    try:
        # Try to load SVD model info if available
        if os.path.exists('models/svd_model_info.pkl'):
            with open('models/svd_model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                return model_info
        # Fallback to best model info
        elif os.path.exists('models/model_info.pkl'):
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

# ===== RECOMMENDATION FUNCTIONS =====

def predict_rating(model, user_id, movie_id):
    """Predict rating for a user-movie pair using the SVD model"""
    try:
        if model is None:
            return 0
        prediction = model.predict(str(user_id), str(movie_id))
        return prediction.est
    except Exception as e:
        print(f"Error predicting rating: {e}")
        return 0

def get_recommendations_with_model(model, user_id, movies_df, ratings_df, n=10):
    """Generate personalized recommendations using the SVD model"""
    
    # Get movies the user has already rated
    if user_id in ratings_df['user_id'].values:
        user_rated_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
    else:
        user_rated_movies = []
    
    # Get all movies the user hasn't rated
    unwatched_movies = movies_df[~movies_df['movie_id'].isin(user_rated_movies)]
    
    if len(unwatched_movies) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
    
    # Generate predictions for unwatched movies
    predictions = []
    with st.spinner("Generating personalized recommendations..."):
        for _, movie in unwatched_movies.iterrows():
            movie_id = movie['movie_id']
            # Predict rating
            pred_rating = predict_rating(model, user_id, movie_id)
            predictions.append((movie_id, pred_rating))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-n movie IDs
    top_movie_ids = [movie_id for movie_id, _ in predictions[:n]]
    
    # Get movie details
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)].copy()
    
    if len(recommended_movies) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
    
    # Add predicted ratings
    movie_ratings = {movie_id: rating for movie_id, rating in predictions}
    recommended_movies['predicted_rating'] = recommended_movies['movie_id'].map(movie_ratings)
    
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
    
    # Sort by average rating and count (weighted)
    popular_movies['popularity_score'] = popular_movies['avg_rating'] * 0.7 + \
                                       (popular_movies['rating_count'] / popular_movies['rating_count'].max()) * 0.3
    popular_movies = popular_movies.sort_values(by='popularity_score', ascending=False)
    
    # Get top-n movie IDs
    top_movie_ids = popular_movies['movie_id'].head(n).tolist()
    
    # Get movie details
    recommended_movies = movies_df[movies_df['movie_id'].isin(top_movie_ids)].copy()
    
    # Add average rating
    movie_ratings = {row['movie_id']: row['avg_rating'] for _, row in popular_movies.iterrows()}
    recommended_movies['predicted_rating'] = recommended_movies['movie_id'].map(movie_ratings)
    
    # Sort by popularity score
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

def get_similar_movies(model, movie_id, movies_df, ratings_df, n=6):
    """Get movies similar to a given movie based on rating patterns"""
    
    # Get users who rated this movie highly
    if movie_id in ratings_df['movie_id'].values:
        high_raters = ratings_df[(ratings_df['movie_id'] == movie_id) & 
                               (ratings_df['rating'] >= 4)]['user_id'].unique()
    else:
        # If no rating data, return popular movies
        return get_popular_movies(movies_df, ratings_df, n)
        
    if len(high_raters) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
    
    # Get other movies these users rated highly
    other_highly_rated = ratings_df[(ratings_df['user_id'].isin(high_raters)) & 
                                  (ratings_df['rating'] >= 4) &
                                  (ratings_df['movie_id'] != movie_id)]
    
    # Count occurrences and sort by frequency
    movie_counts = other_highly_rated['movie_id'].value_counts().reset_index()
    movie_counts.columns = ['movie_id', 'count']
    
    # Get top-n movie IDs
    similar_movie_ids = movie_counts['movie_id'].head(n).tolist()
    
    # Get movie details
    similar_movies = movies_df[movies_df['movie_id'].isin(similar_movie_ids)].copy()
    
    # If no similar movies found, return popular movies
    if len(similar_movies) == 0:
        return get_popular_movies(movies_df, ratings_df, n)
        
    return similar_movies

def get_movie_details(movies_df, ratings_df, movie_id):
    """Get detailed information about a movie"""
    # Get movie details
    movie = movies_df[movies_df['movie_id'] == movie_id]
    
    if len(movie) == 0:
        return None, 0, 0, None
        
    movie = movie.iloc[0]
    
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

# ===== UI DISPLAY FUNCTIONS =====

def display_movie_card(movie, col, show_prediction=False):
    """Display a movie card in a Streamlit column"""
    with col:
        card_html = f"""
        <div class="movie-card">
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-year">{movie['year'] if 'year' in movie else extract_year_from_title(movie['title'])}</div>
            <div class="movie-genre">
        """
        
        # Add genre tags
        for genre in movie['genre_names']:
            card_html += f'<span class="tag">{genre}</span>'
        
        card_html += "</div>"
        
        # Add rating if available
        if show_prediction and 'predicted_rating' in movie:
            card_html += f'<div class="rating">{"★" * int(movie["predicted_rating"])}{"☆" * (5 - int(movie["predicted_rating"]))} ({movie["predicted_rating"]:.1f})</div>'
        
        card_html += "</div>"
        
        st.markdown(card_html, unsafe_allow_html=True)
        
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
        st.sidebar.subheader("Login")
        
        # For demo purposes, let user select from a list of user IDs
        user_id = st.sidebar.selectbox(
            "Select User ID (for demo)",
            options=list(range(1, 20)),
            index=0,
            key="login_user_id"
        )
        
        # Option to login as a new user
        new_user = st.sidebar.checkbox("I'm a new user")
        
        if new_user:
            st.sidebar.info("As a new user, you'll receive recommendations based on popular movies until you rate some movies.")
        
        if st.sidebar.button("Login", key="login_button"):
            with st.sidebar:
                with st.spinner("Logging in..."):
                    time.sleep(0.5)  # Simulate login process
            
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.sidebar.success(f"Logged in as User {user_id}")
            st.rerun()
    else:
        st.sidebar.success(f"Logged in as User {st.session_state.user_id}")
        
        # Show user ratings info
        ratings_df = load_ratings_data()
        user_ratings = ratings_df[ratings_df['user_id'] == st.session_state.user_id]
        
        if len(user_ratings) > 0:
            st.sidebar.info(f"You have rated {len(user_ratings)} movies.")
            avg_rating = user_ratings['rating'].mean()
            st.sidebar.info(f"Your average rating: {avg_rating:.1f} ⭐")
        else:
            st.sidebar.info("You haven't rated any movies yet.")
        
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()

def display_home_page(movies_df, ratings_df, model, model_info):
    """Display the home page with recommendations"""
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎬 MovieLens Recommender</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # If user is logged in, show personalized recommendations
    if st.session_state.logged_in:
        st.markdown(f"""<h2 class="section-header">Personalized Recommendations for You</h2>
        <p class="subtitle">Based on your rating history and preferences</p>""", unsafe_allow_html=True)
        
        # Get recommendations using the model
        recommended_movies = get_recommendations_with_model(
            model, st.session_state.user_id, movies_df, ratings_df, n=12
        )
        
        # Display recommendations in a grid (3 columns)
        if len(recommended_movies) > 0:
            # Create rows of 3 movies each
            for i in range(0, len(recommended_movies), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(recommended_movies):
                        display_movie_card(recommended_movies.iloc[i + j], cols[j], show_prediction=True)
        else:
            st.info("Not enough data to provide personalized recommendations. Here are some popular movies you might enjoy.")
            popular_movies = get_popular_movies(movies_df, ratings_df, n=6)
            
            # Display popular movies in a grid (3 columns)
            cols = st.columns(3)
            for i, (_, movie) in enumerate(popular_movies.iterrows()):
                display_movie_card(movie, cols[i % 3])
    
    # Popular Movies Section
    st.markdown("""<h2 class="section-header">Popular Movies</h2>
    <p class="subtitle">Trending among all users</p>""", unsafe_allow_html=True)
    
    popular_movies = get_popular_movies(movies_df, ratings_df, n=6)
    
    # Display popular movies in a grid (3 columns)
    cols = st.columns(3)
    for i, (_, movie) in enumerate(popular_movies.iterrows()):
        display_movie_card(movie, cols[i % 3])
    
    # Browse by Genre Section
    st.markdown("""<h2 class="section-header">Browse by Genre</h2>
    <p class="subtitle">Explore movies by category</p>""", unsafe_allow_html=True)
    
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
            if 'parameters' in model_info:
                st.write("**Parameters:**")
                for param, value in model_info['parameters'].items():
                    st.write(f"- {param}: {value}")
        
        with col2:
            st.subheader("Performance Metrics")
            st.write(f"**RMSE:** {model_info.get('rmse', 'N/A'):.4f}")
            st.write(f"**Precision:** {model_info.get('precision', 'N/A'):.4f}")
            st.write(f"**Recall:** {model_info.get('recall', 'N/A'):.4f}")
            
            # Create a simple visualization of metrics
            metrics = {
                'RMSE': model_info.get('rmse', 0),
                'Precision': model_info.get('precision', 0),
                'Recall': model_info.get('recall', 0)
            }
            
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(metrics.keys(), metrics.values(), color=['#ff9900', '#36b37e', '#0065ff'])
            ax.set_ylim(0, 1.0)
            st.pyplot(fig)

def display_movie_details_page(movies_df, ratings_df, model):
    """Display detailed information about a selected movie"""
    # Get movie ID from session state
    movie_id = st.session_state.selected_movie
    
    # Get movie details
    movie, avg_rating, rating_count, rating_dist = get_movie_details(movies_df, ratings_df, movie_id)
    
    if movie is None:
        st.error("Movie not found.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    # Movie title and basic info
    st.title(movie['title'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Movie Information")
        st.write(f"**Year:** {movie['year'] if 'year' in movie else extract_year_from_title(movie['title'])}")
        
        # Display genres as tags
        st.write("**Genres:**")
        genre_html = ""
        for genre in movie['genre_names']:
            genre_html += f'<span class="tag">{genre}</span>'
        st.markdown(genre_html, unsafe_allow_html=True)
        
        # Description placeholder (would be real data in a full implementation)
        st.write("**Description:**")
        st.write("This is a placeholder description for the movie. In a real implementation, this would contain the actual movie plot or synopsis.")
        
        # Similar movies section
        st.subheader("Similar Movies You Might Enjoy")
        similar_movies = get_similar_movies(model, movie_id, movies_df, ratings_df)
        
        # Display similar movies in a grid
        if len(similar_movies) > 0:
            similar_cols = st.columns(3)
            for i, (_, sim_movie) in enumerate(similar_movies.iterrows()):
                display_movie_card(sim_movie, similar_cols[i % 3])
        else:
            st.info("No similar movies found.")
    
    with col2:
        st.subheader("Rating Information")
        
        # Display star rating visually
        st.markdown(f"""
        <div style="font-size: 24px; color: #ff9900; margin-bottom: 10px;">
            {"★" * int(avg_rating)}{"☆" * (5 - int(avg_rating))} {avg_rating:.1f}/5.0
        </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Number of Ratings:** {rating_count}")
        
        # Rating distribution
        if rating_count > 0:
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(rating_dist.index, rating_dist.values, color='#4299e1')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       int(height), ha='center', va='bottom')
            
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            ax.set_xticks([1, 2, 3, 4, 5])
            st.pyplot(fig)
    
    # User rating section
    if st.session_state.logged_in:
        st.markdown("""<h2 class="section-header">Rate This Movie</h2>""", unsafe_allow_html=True)
        
        # Check if user has already rated this movie
        user_rating = None
        ratings_user = ratings_df[(ratings_df['user_id'] == st.session_state.user_id) & 
                                (ratings_df['movie_id'] == movie_id)]
        
        if len(ratings_user) > 0:
            user_rating = ratings_user.iloc[0]['rating']
            st.info(f"You previously rated this movie: {user_rating:.1f} stars")
        
        # Rating slider
        new_rating = st.slider("Your Rating", 0.5, 5.0, 
                             float(user_rating) if user_rating is not None else 3.0, 0.5)
        
        submit_label = "Update Rating" if user_rating is not None else "Submit Rating"
        if st.button(submit_label):
            # In a real app, this would update the database
            st.success(f"Thank you for rating '{movie['title']}' with {new_rating} stars!")
            
            # For demo purposes, let's just show how this would affect recommendations
            st.info("Your recommendations will be updated based on this rating.")

def display_search_page(movies_df, ratings_df):
    """Display search page for finding movies"""
    st.title("Search Movies")
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()
    
    # Search options
    search_tabs = st.tabs(["Title Search", "Advanced Search"])
    
    with search_tabs[0]:
        # Simple title search
        search_query = st.text_input("Search for movies by title")
        
        if search_query:
            # Search movies by title
            search_results = movies_df[movies_df['title'].str.contains(search_query, case=False, na=False)]
            
            if len(search_results) > 0:
                st.subheader(f"Found {len(search_results)} results for '{search_query}'")
                
                # Display search results in a grid (3 columns)
                for i in range(0, min(len(search_results), 12), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(search_results):
                            display_movie_card(search_results.iloc[i + j], cols[j])
            else:
                st.info(f"No movies found matching '{search_query}'")
    
    with search_tabs[1]:
        # Advanced search with filters
        st.subheader("Find Movies by Genre and Year")
        
        # Get all unique genres
        all_genres = []
        for genres in movies_df['genre_names']:
            if isinstance(genres, list):
                all_genres.extend(genres)
        all_genres = sorted(list(set(all_genres)))
        
        # Create multiselect for genres
        selected_genres = st.multiselect("Select genres", all_genres)
        
        # Year range
        min_year = int(movies_df['year'].min()) if 'year' in movies_df.columns else 1900
        max_year = int(movies_df['year'].max()) if 'year' in movies_df.columns else 2025
        
        year_range = st.slider("Release year range", min_year, max_year, (min_year, max_year))
        
        # Sort options
        sort_option = st.selectbox("Sort by", ["Year (newest first)", "Year (oldest first)", "Title (A-Z)"])
        
        if st.button("Search"):
            # Filter by genre
            if selected_genres:
                filtered_movies = movies_df[movies_df['genre_names'].apply(
                    lambda x: any(genre in x for genre in selected_genres))]
            else:
                filtered_movies = movies_df.copy()
            
            # Filter by year
            if 'year' in filtered_movies.columns:
                filtered_movies = filtered_movies[(filtered_movies['year'].astype(int) >= year_range[0]) & 
                                               (filtered_movies['year'].astype(int) <= year_range[1])]
            
            # Sort results
            if sort_option == "Year (newest first)" and 'year' in filtered_movies.columns:
                filtered_movies = filtered_movies.sort_values(by='year', ascending=False)
            elif sort_option == "Year (oldest first)" and 'year' in filtered_movies.columns:
                filtered_movies = filtered_movies.sort_values(by='year', ascending=True)
            else:  # Title (A-Z)
                filtered_movies = filtered_movies.sort_values(by='title')
            
            # Display results
            if len(filtered_movies) > 0:
                st.subheader(f"Found {len(filtered_movies)} movies matching your criteria")
                
                # Display results in a grid (3 columns)
                for i in range(0, min(len(filtered_movies), 12), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(filtered_movies):
                            display_movie_card(filtered_movies.iloc[i + j], cols[j])
                
                # Add pagination if many results
                if len(filtered_movies) > 12:
                    st.info(f"Showing 12 of {len(filtered_movies)} results. Use more specific filters to narrow down your search.")
            else:
                st.info("No movies found matching your criteria.")

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
    
    if st.sidebar.button("Home", key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("Search Movies", key="nav_search"):
        st.session_state.page = "search"
        st.rerun()
    
    # Load data
    movies_df = load_movie_data()
    ratings_df = load_ratings_data()
    model = load_model()
    model_info = load_model_info()
    
    # Display info about data
    with st.sidebar.expander("Dataset Information"):
        st.write(f"**Movies:** {len(movies_df)}")
        st.write(f"**Ratings:** {len(ratings_df)}")
        st.write(f"**Users:** {ratings_df['user_id'].nunique()}")
        
        # Add a small visualization
        fig, ax = plt.subplots(figsize=(4, 2))
        
        # Get top 5 genres by movie count
        genre_counts = defaultdict(int)
        for genres in movies_df['genre_names']:
            for genre in genres:
                genre_counts[genre] += 1
        
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        genres = [g[0] for g in top_genres]
        counts = [g[1] for g in top_genres]
        
        ax.barh(genres, counts, color='#4299e1')
        ax.set_title('Top Genres', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Route to appropriate page
    if st.session_state.page == "home":
        display_home_page(movies_df, ratings_df, model, model_info)
    elif st.session_state.page == "movie_details":
        display_movie_details_page(movies_df, ratings_df, model)
    elif st.session_state.page == "search":
        display_search_page(movies_df, ratings_df)

if __name__ == "__main__":
    main()
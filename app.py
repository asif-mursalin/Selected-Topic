import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import time
from surprise import SVD, Dataset, Reader

# Set page configuration
st.set_page_config(
    page_title="MovieLens Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .movie-card {
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
        transition: transform 0.2s;
    }
    .movie-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .recommendation-header {
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0 10px 0;
    }
    .genre-tag {
        background-color: #e1e1e1;
        border-radius: 15px;
        padding: 3px 10px;
        margin: 2px;
        font-size: 12px;
        display: inline-block;
    }
    .movie-title {
        font-weight: bold;
        font-size: 18px;
    }
    .movie-year {
        color: #888;
        margin-left: 5px;
    }
    .rating-stars {
        color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# ------------ Helper Functions ------------

@st.cache_data
def load_data():
    """Load movie and model data."""
    # Load model info
    model_info_path = 'models/model_info.pkl'
    if os.path.exists(model_info_path):
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
    else:
        model_info = {
            'model_type': 'Unknown',
            'rmse': 0,
            'precision': 0,
            'recall': 0
        }
    
    # Load movie data
    movies_path = 'models/movies_data.pkl'
    if os.path.exists(movies_path):
        with open(movies_path, 'rb') as f:
            movies_df = pickle.load(f)
    else:
        # Create a mock movies dataframe if file doesn't exist
        st.warning("Movie data not found. Using sample data.")
        movies_df = pd.DataFrame({
            'movie_id': range(1, 11),
            'title': [f"Sample Movie {i}" for i in range(1, 11)],
            'year': [2000 + i for i in range(1, 11)],
            'genre_names': [['Drama', 'Action'] for _ in range(10)]
        })
    
    # Load best model
    model_path = 'models/best_tf_model_Neural Collaborative Filtering.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = None
        st.warning("Model file not found.")
    
    # Load or create ratings data
    # For a real app, you'd load actual ratings, but we'll simulate it
    ratings_df = pd.DataFrame({
        'user_id': np.random.randint(1, 611, size=1000),
        'movie_id': np.random.choice(movies_df['movie_id'].values, size=1000),
        'rating': np.random.uniform(0.5, 5, size=1000).round(1)
    })
    
    return model, model_info, movies_df, ratings_df

@st.cache_data
def get_movie_stats(movies_df, ratings_df):
    """Get movie statistics."""
    movie_stats = defaultdict(dict)
    
    for movie_id in movies_df['movie_id'].unique():
        # Get ratings for this movie
        movie_ratings = ratings_df[ratings_df['movie_id'] == movie_id]['rating']
        
        if len(movie_ratings) > 0:
            movie_stats[movie_id]['avg_rating'] = movie_ratings.mean()
            movie_stats[movie_id]['rating_count'] = len(movie_ratings)
        else:
            movie_stats[movie_id]['avg_rating'] = 0
            movie_stats[movie_id]['rating_count'] = 0
    
    return movie_stats

def get_top_movies(movies_df, movie_stats, n=10, by_rating=True, min_ratings=5):
    """Get top N movies by average rating or popularity."""
    # Create DataFrame with movie stats
    stats_df = pd.DataFrame.from_dict(movie_stats, orient='index')
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'movie_id'}, inplace=True)
    
    # Merge with movies_df
    movie_data = movies_df.merge(stats_df, on='movie_id')
    
    # Filter by minimum number of ratings
    movie_data = movie_data[movie_data['rating_count'] >= min_ratings]
    
    if by_rating:
        # Sort by average rating
        return movie_data.sort_values('avg_rating', ascending=False).head(n)
    else:
        # Sort by popularity (rating count)
        return movie_data.sort_values('rating_count', ascending=False).head(n)

def get_genre_movies(movies_df, movie_stats, genre, n=10, min_ratings=5):
    """Get top N movies for a specific genre."""
    # Filter movies by genre
    genre_movies = movies_df[movies_df['genre_names'].apply(lambda x: genre in x)]
    
    # Create DataFrame with movie stats
    stats_df = pd.DataFrame.from_dict(movie_stats, orient='index')
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'movie_id'}, inplace=True)
    
    # Merge with filtered movies
    movie_data = genre_movies.merge(stats_df, on='movie_id')
    
    # Filter by minimum number of ratings and sort by average rating
    movie_data = movie_data[movie_data['rating_count'] >= min_ratings]
    return movie_data.sort_values('avg_rating', ascending=False).head(n)

def find_similar_movies(movie_id, model, movies_df, n=5):
    """Find similar movies using the trained model."""
    if model is None:
        # Fallback if model isn't loaded
        return movies_df.sample(n)
    
    try:
        # Check if model has prediction capability
        if hasattr(model, 'get_neighbors'):
            # For KNN models
            inner_id = model.trainset.to_inner_iid(str(movie_id))
            neighbors = model.get_neighbors(inner_id, k=n)
            similar_movies = [int(model.trainset.to_raw_iid(inner_id)) for inner_id in neighbors]
            return movies_df[movies_df['movie_id'].isin(similar_movies)]
        else:
            # For matrix factorization models like SVD
            # We can't directly get similar movies, so we'll use a hybrid approach
            # Get the movie's genres and return other highly-rated movies in same genres
            movie_genres = movies_df[movies_df['movie_id'] == movie_id]['genre_names'].iloc[0]
            similar_genre_movies = movies_df[
                (movies_df['movie_id'] != movie_id) &
                (movies_df['genre_names'].apply(lambda x: any(g in movie_genres for g in x)))
            ]
            # Sort by overlap in genres
            similar_genre_movies['genre_overlap'] = similar_genre_movies['genre_names'].apply(
                lambda x: len(set(x).intersection(set(movie_genres)))
            )
            return similar_genre_movies.sort_values('genre_overlap', ascending=False).head(n)
    except Exception as e:
        st.error(f"Error finding similar movies: {e}")
        return movies_df.sample(n)

def predict_user_ratings(user_id, movies_df, model):
    """Predict ratings for all movies for a specific user."""
    if model is None:
        return {}
    
    predictions = {}
    
    try:
        for movie_id in movies_df['movie_id'].unique():
            pred = model.predict(str(user_id), str(movie_id))
            predictions[movie_id] = pred.est
    except Exception as e:
        st.error(f"Error predicting ratings: {e}")
    
    return predictions

def get_user_recommendations(user_id, movies_df, model, n=10, exclude_rated=True, rated_movies=None):
    """Get personalized recommendations for a user."""
    if model is None:
        return movies_df.sample(n)
    
    # Get predictions for all movies
    predictions = predict_user_ratings(user_id, movies_df, model)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame([
        {'movie_id': movie_id, 'predicted_rating': rating}
        for movie_id, rating in predictions.items()
    ])
    
    # Merge with movies_df
    movie_data = movies_df.merge(pred_df, on='movie_id')
    
    # Exclude already rated movies if needed
    if exclude_rated and rated_movies is not None:
        movie_data = movie_data[~movie_data['movie_id'].isin(rated_movies)]
    
    # Sort by predicted rating
    return movie_data.sort_values('predicted_rating', ascending=False).head(n)

def display_movie_card(movie, avg_rating=None, rating_count=None, predicted_rating=None, col_width=1):
    """Display a movie card with details."""
    with st.container():
        # Get year from title or use the year column if available
        if 'year' in movie:
            year = movie['year']
        else:
            # Extract year from title (assuming format: "Movie Title (YYYY)")
            title = movie['title']
            year = title[-5:-1] if title[-5:-1].isdigit() else "Unknown"
            title = title[:-7] if title[-5:-1].isdigit() else title
        
        # Display movie information
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{movie['title']}
                <span class="movie-year">({year})</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Display genres as tags
        if 'genre_names' in movie:
            genre_html = ""
            for genre in movie['genre_names']:
                genre_html += f'<span class="genre-tag">{genre}</span> '
            st.markdown(f"<div>{genre_html}</div>", unsafe_allow_html=True)
        
        # Display ratings information
        if avg_rating is not None:
            # Create star rating display
            stars = "★" * int(avg_rating) + "☆" * (5 - int(avg_rating))
            if rating_count is not None:
                st.markdown(f"""
                <div class="rating-stars">{stars}</div>
                <div>Avg: {avg_rating:.1f}/5 ({rating_count} ratings)</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rating-stars">{stars}</div>
                <div>Avg: {avg_rating:.1f}/5</div>
                """, unsafe_allow_html=True)
        
        # Display predicted rating if available
        if predicted_rating is not None:
            # Create star rating display
            pred_stars = "★" * int(predicted_rating) + "☆" * (5 - int(predicted_rating))
            st.markdown(f"""
            <div>Predicted rating: {predicted_rating:.1f}/5</div>
            <div class="rating-stars">{pred_stars}</div>
            """, unsafe_allow_html=True)
        
        # Add rate button
        rate_value = st.slider(f"Rate '{movie['title']}'", 0.5, 5.0, 2.5, 0.5, key=f"rate_{movie['movie_id']}")
        submit = st.button("Submit Rating", key=f"submit_{movie['movie_id']}")
        
        if submit:
            st.success(f"Rating of {rate_value} submitted for {movie['title']}!")
            # Here you would typically save this rating to a database
        
        st.markdown("</div>", unsafe_allow_html=True)

def render_movie_details(movie_id, movies_df, movie_stats, model):
    """Render detailed view for a selected movie."""
    # Get movie data
    movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0].to_dict()
    
    # Get movie statistics
    avg_rating = movie_stats[movie_id]['avg_rating'] if movie_id in movie_stats else 0
    rating_count = movie_stats[movie_id]['rating_count'] if movie_id in movie_stats else 0
    
    # Display movie header
    st.title(movie['title'])
    
    # Display movie information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Movie poster (placeholder)
        st.image("https://via.placeholder.com/300x450?text=Movie+Poster", use_column_width=True)
        
        # Display average rating
        stars = "★" * int(avg_rating) + "☆" * (5 - int(avg_rating))
        st.markdown(f"""
        <div style='text-align: center; margin-top: 10px;'>
            <div style='font-size: 24px; color: #FFD700;'>{stars}</div>
            <div style='font-size: 18px;'>{avg_rating:.1f}/5 ({rating_count} ratings)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Rate this movie
        st.subheader("Rate this movie")
        rate_value = st.slider("Your rating", 0.5, 5.0, 2.5, 0.5)
        submit = st.button("Submit Rating")
        
        if submit:
            st.success(f"Rating of {rate_value} submitted for {movie['title']}!")
            # Here you would typically save this rating to a database
    
    with col2:
        # Movie details
        if 'year' in movie:
            year = movie['year']
        else:
            # Extract year from title (assuming format: "Movie Title (YYYY)")
            title = movie['title']
            year = title[-5:-1] if title[-5:-1].isdigit() else "Unknown"
        
        st.subheader("Details")
        st.write(f"**Release Year:** {year}")
        
        # Display genres
        if 'genre_names' in movie:
            genre_html = "<div style='margin-bottom: 10px;'><strong>Genres:</strong> "
            for genre in movie['genre_names']:
                genre_html += f'<span class="genre-tag">{genre}</span> '
            genre_html += "</div>"
            st.markdown(genre_html, unsafe_allow_html=True)
        
        # Movie description (placeholder)
        st.subheader("Description")
        st.write("No description available in the MovieLens dataset. This would typically contain a plot summary.")
        
        # Movie cast and crew (placeholder)
        st.subheader("Cast & Crew")
        st.write("Cast and crew information not available in the basic MovieLens dataset.")
    
    # Similar movies section
    st.header("Similar Movies")
    similar_movies = find_similar_movies(movie_id, model, movies_df)
    
    # Display similar movies in a grid
    cols = st.columns(5)
    for i, (_, similar_movie) in enumerate(similar_movies.iterrows()):
        with cols[i % 5]:
            display_movie_card(
                similar_movie.to_dict(),
                avg_rating=movie_stats[similar_movie['movie_id']]['avg_rating'] if similar_movie['movie_id'] in movie_stats else None,
                rating_count=movie_stats[similar_movie['movie_id']]['rating_count'] if similar_movie['movie_id'] in movie_stats else None
            )
            if st.button(f"View Details", key=f"view_{similar_movie['movie_id']}"):
                st.session_state.selected_movie = similar_movie['movie_id']
                st.rerun()

def create_recommendations_page(user_id, movies_df, movie_stats, model):
    """Create the recommendations page."""
    st.title(f"Movie Recommendations for User {user_id}")
    
    # Get user's rated movies (in a real app, you'd retrieve this from a database)
    # For now, we'll simulate some ratings
    rated_movies = np.random.choice(movies_df['movie_id'].values, size=10, replace=False)
    
    # Get personalized recommendations
    recommendations = get_user_recommendations(user_id, movies_df, model, exclude_rated=True, rated_movies=rated_movies)
    
    # Display recommendations
    st.header("Recommended for You")
    cols = st.columns(5)
    for i, (_, movie) in enumerate(recommendations.iterrows()):
        with cols[i % 5]:
            display_movie_card(
                movie.to_dict(),
                avg_rating=movie_stats[movie['movie_id']]['avg_rating'] if movie['movie_id'] in movie_stats else None,
                rating_count=movie_stats[movie['movie_id']]['rating_count'] if movie['movie_id'] in movie_stats else None,
                predicted_rating=movie['predicted_rating'] if 'predicted_rating' in movie else None
            )
            if st.button(f"View Details", key=f"view_{movie['movie_id']}"):
                st.session_state.selected_movie = movie['movie_id']
                st.rerun()
    
    # Show top-rated movies
    st.header("Top-Rated Movies")
    top_movies = get_top_movies(movies_df, movie_stats, by_rating=True)
    
    top_cols = st.columns(5)
    for i, (_, movie) in enumerate(top_movies.iterrows()):
        with top_cols[i % 5]:
            display_movie_card(
                movie.to_dict(),
                avg_rating=movie['avg_rating'] if 'avg_rating' in movie else None,
                rating_count=movie['rating_count'] if 'rating_count' in movie else None
            )
            if st.button(f"View Details", key=f"view_top_{movie['movie_id']}"):
                st.session_state.selected_movie = movie['movie_id']
                st.rerun()
    
    # Show popular genres
    st.header("Explore by Genre")
    all_genres = set()
    for genres in movies_df['genre_names']:
        all_genres.update(genres)
    
    # Display a few select genres
    selected_genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']
    selected_genres = [g for g in selected_genres if g in all_genres]
    
    # Display movies by genre in tabs
    genre_tabs = st.tabs(selected_genres)
    
    for i, genre in enumerate(selected_genres):
        with genre_tabs[i]:
            genre_movies = get_genre_movies(movies_df, movie_stats, genre)
            
            genre_cols = st.columns(5)
            for j, (_, movie) in enumerate(genre_movies.iterrows()):
                with genre_cols[j % 5]:
                    display_movie_card(
                        movie.to_dict(),
                        avg_rating=movie['avg_rating'] if 'avg_rating' in movie else None,
                        rating_count=movie['rating_count'] if 'rating_count' in movie else None
                    )
                    if st.button(f"View Details", key=f"view_{genre}_{movie['movie_id']}"):
                        st.session_state.selected_movie = movie['movie_id']
                        st.rerun()

def create_explore_page(movies_df, movie_stats):
    """Create the explore movies page."""
    st.title("Explore Movies")
    
    # Search functionality
    search_query = st.text_input("Search for movies by title")
    
    if search_query:
        # Filter movies by title
        filtered_movies = movies_df[movies_df['title'].str.contains(search_query, case=False)]
        
        if len(filtered_movies) > 0:
            st.write(f"Found {len(filtered_movies)} movies matching '{search_query}'")
            
            # Display search results
            for _, movie in filtered_movies.iterrows():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Movie poster (placeholder)
                    st.image("https://via.placeholder.com/150x225?text=Movie", use_column_width=True)
                
                with col2:
                    avg_rating = movie_stats[movie['movie_id']]['avg_rating'] if movie['movie_id'] in movie_stats else 0
                    rating_count = movie_stats[movie['movie_id']]['rating_count'] if movie['movie_id'] in movie_stats else 0
                    
                    st.subheader(movie['title'])
                    
                    # Display genres as tags
                    if 'genre_names' in movie:
                        genre_html = ""
                        for genre in movie['genre_names']:
                            genre_html += f'<span class="genre-tag">{genre}</span> '
                        st.markdown(f"<div>{genre_html}</div>", unsafe_allow_html=True)
                    
                    # Display ratings
                    stars = "★" * int(avg_rating) + "☆" * (5 - int(avg_rating))
                    st.markdown(f"""
                    <div class="rating-stars">{stars}</div>
                    <div>Avg: {avg_rating:.1f}/5 ({rating_count} ratings)</div>
                    """, unsafe_allow_html=True)
                    
                    # View details button
                    if st.button("View Details", key=f"view_search_{movie['movie_id']}"):
                        st.session_state.selected_movie = movie['movie_id']
                        st.rerun()
                
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.warning(f"No movies found matching '{search_query}'")
    
    # Display filtering options
    st.header("Filter Movies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get all genres
        all_genres = set()
        for genres in movies_df['genre_names']:
            all_genres.update(genres)
        
        # Sort genres alphabetically
        all_genres = sorted(list(all_genres))
        
        # Genre filter
        selected_genre = st.selectbox("Select Genre", ["All Genres"] + all_genres)
    
    with col2:
        # Sort options
        sort_by = st.selectbox("Sort By", ["Rating (High to Low)", "Popularity", "Release Year"])
    
    # Apply filters
    filtered_df = movies_df.copy()
    
    # Filter by genre
    if selected_genre != "All Genres":
        filtered_df = filtered_df[filtered_df['genre_names'].apply(lambda x: selected_genre in x)]
    
    # Create movie stats DataFrame
    stats_df = pd.DataFrame.from_dict(movie_stats, orient='index')
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'movie_id'}, inplace=True)
    
    # Merge with filtered movies
    movie_data = filtered_df.merge(stats_df, on='movie_id', how='left')
    
    # Sort data
    if sort_by == "Rating (High to Low)":
        movie_data = movie_data.sort_values('avg_rating', ascending=False)
    elif sort_by == "Popularity":
        movie_data = movie_data.sort_values('rating_count', ascending=False)
    elif sort_by == "Release Year":
        # Extract year from title or use year column
        if 'year' in movie_data.columns:
            movie_data = movie_data.sort_values('year', ascending=False)
        else:
            # This would be more complex in a real app
            pass
    
    # Display filtered movies
    st.write(f"Showing {len(movie_data)} movies")
    
    # Display movies in a grid
    cols = st.columns(5)
    for i, (_, movie) in enumerate(movie_data.head(50).iterrows()):
        with cols[i % 5]:
            display_movie_card(
                movie.to_dict(),
                avg_rating=movie['avg_rating'] if 'avg_rating' in movie else None,
                rating_count=movie['rating_count'] if 'rating_count' in movie else None
            )
            if st.button(f"View Details", key=f"view_filter_{movie['movie_id']}"):
                st.session_state.selected_movie = movie['movie_id']
                st.rerun()

def create_model_info_page(model_info):
    """Create the model information page."""
    st.title("Recommendation System Information")
    
    # Display model information
    st.header("Model Details")
    st.write(f"**Model Type:** {model_info['model_type']}")
    st.write(f"**RMSE (Root Mean Square Error):** {model_info['rmse']:.4f}")
    st.write(f"**Precision:** {model_info['precision']:.4f}")
    st.write(f"**Recall:** {model_info['recall']:.4f}")
    
    # Explanation of metrics
    st.subheader("Understanding the Metrics")
    st.write("""
    - **RMSE (Root Mean Square Error):** Measures the average magnitude of errors in rating predictions. Lower values indicate better accuracy.
    - **Precision:** The fraction of recommended items that are relevant to the user.
    - **Recall:** The fraction of relevant items that are successfully recommended to the user.
    """)
    
    # About the recommendation system
    st.header("About This Recommendation System")
    st.write("""
    This movie recommendation system is built using the MovieLens dataset, which contains user ratings for movies.
    It employs collaborative filtering techniques to predict how users would rate movies they haven't seen yet,
    based on their past ratings and the ratings of similar users.
    
    The system provides:
    - Personalized movie recommendations
    - Movie exploration by genre, rating, and popularity
    - Detailed movie information
    - Similar movie suggestions
    """)
    
    # Implementation details based on project documents
    st.header("Implementation Details")
    st.write("""
    The recommendation system was developed following these phases:
    
    1. **Dataset Preparation**
       - Data acquisition and analysis of the MovieLens dataset
       - Preprocessing to handle missing values and create appropriate data structures
       - Feature engineering to extract useful information from the data
    
    2. **Model Implementation**
       - Baseline models including popularity-based and simple collaborative filtering
       - Matrix factorization techniques like SVD (Singular Value Decomposition)
       - Advanced hybrid models combining collaborative and content-based filtering
    
    3. **Model Evaluation**
       - Performance measurements using RMSE, precision, and recall
       - Hyperparameter tuning to optimize model performance
       - Comparative analysis of different recommendation approaches
    """)
    
    # Create a sample visualization
    st.header("Performance Comparison")
    st.write("Sample visualization of different model performances:")
    
    # Create dummy data for visualization
    models = ['Popularity Baseline', 'User-based CF', 'Item-based CF', 'SVD', 'Hybrid Model']
    rmse_values = [1.02, 0.98, 0.97, 0.94, 0.92]
    precision_values = [0.65, 0.68, 0.7, 0.75, 0.78]
    recall_values = [0.55, 0.58, 0.6, 0.65, 0.68]
    
    # Create a DataFrame for plotting
    performance_df = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'Precision': precision_values,
        'Recall': recall_values
    })
    
    # Plot performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE plot
    ax1.bar(performance_df['Model'], performance_df['RMSE'], color='skyblue')
    ax1.set_title('RMSE by Model (Lower is Better)')
    ax1.set_ylabel('RMSE')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Precision and Recall plot
    ax2.bar(performance_df['Model'], performance_df['Precision'], color='lightgreen', label='Precision')
    ax2.bar(performance_df['Model'], performance_df['Recall'], color='salmon', alpha=0.7, label='Recall')
    ax2.set_title('Precision and Recall by Model')
    ax2.set_ylabel('Score')
    ax2.legend()
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)

# ------------ Main App ------------

def main():
    # Initialize session state variables if they don't exist
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 1  # Default user ID
    if 'page' not in st.session_state:
        st.session_state.page = "recommendations"
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    
    # Load data
    model, model_info, movies_df, ratings_df = load_data()
    
    # Calculate movie statistics
    movie_stats = get_movie_stats(movies_df, ratings_df)
    
    # Sidebar navigation
    st.sidebar.title("MovieLens Recommender")
    
    # User selection
    st.sidebar.subheader("User Settings")
    user_id = st.sidebar.number_input("User ID", min_value=1, max_value=610, value=st.session_state.user_id)
    st.session_state.user_id = user_id
    
# Navigation
    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Recommendations", "Explore Movies", "Model Information"]
    )
    
    if page == "Recommendations":
        st.session_state.page = "recommendations"
    elif page == "Explore Movies":
        st.session_state.page = "explore"
    elif page == "Model Information":
        st.session_state.page = "model_info"
    
    # Sidebar information
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This app demonstrates a movie recommendation system built with the MovieLens dataset.
    
    Data: MovieLens 32M dataset
    Models: Collaborative filtering with matrix factorization
    """)
    
    # Main content area
    if st.session_state.selected_movie is not None:
        # Show movie details page
        render_movie_details(st.session_state.selected_movie, movies_df, movie_stats, model)
        
        # Add a back button
        if st.button("← Back to " + st.session_state.page.capitalize()):
            st.session_state.selected_movie = None
            st.rerun()
    
    elif st.session_state.page == "recommendations":
        create_recommendations_page(user_id, movies_df, movie_stats, model)
    
    elif st.session_state.page == "explore":
        create_explore_page(movies_df, movie_stats)
    
    elif st.session_state.page == "model_info":
        create_model_info_page(model_info)

if __name__ == "__main__":
    main()
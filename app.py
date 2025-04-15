import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from collections import defaultdict

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .genre-pill {
        display: inline-block;
        background-color: #e1f5fe;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        color: #0277bd;
        font-size: 14px;
    }
    .genre-pill-selected {
        background-color: #0277bd;
        color: white;
    }
    .recommendation-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    .recommendation-table th, .recommendation-table td {
        border: 1px solid #ddd;
        padding: 12px;
    }
    .recommendation-table th {
        background-color: #f2f2f2;
    }
    .recommendation-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .movie-title {
        font-weight: bold;
    }
    .movie-genres {
        font-size: 12px;
        color: #555;
    }
    .slider-container {
        padding: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def load_movie_data():
    """Load movie data from pickle file"""
    try:
        with open('models/movies_data.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Movie data file not found. Please run the data preparation script first.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Load recommendation model from pickle file"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Model file not found. Using genre-based recommendations instead.")
        return None

def get_all_genres(movies_df):
    """Extract all unique genres from the movies dataframe"""
    all_genres = set()
    for genres in movies_df['genre_names']:
        if isinstance(genres, list):
            all_genres.update(genres)
    return sorted(list(all_genres))

def recommend_movies(selected_genres, min_rating, movies_df, model=None, top_n=5):
    """
    Recommend movies based on selected genres and minimum rating
    """
    # Filter by minimum rating
    filtered_movies = movies_df[movies_df['avg_rating'] >= min_rating]
    
    # If no movies meet the criteria
    if filtered_movies.empty:
        return pd.DataFrame()
    
    # If no genres selected, return top rated movies
    if not selected_genres:
        return filtered_movies.sort_values('avg_rating', ascending=False).head(top_n)
    
    # Calculate genre matching score
    def genre_match_score(movie_genres):
        if not isinstance(movie_genres, list):
            return 0
        matching_genres = set(movie_genres).intersection(set(selected_genres))
        return len(matching_genres) / len(selected_genres) if selected_genres else 0
    
    # Apply scoring function
    filtered_movies['genre_score'] = filtered_movies['genre_names'].apply(genre_match_score)
    
    # Calculate final score (combination of genre match and rating)
    filtered_movies['final_score'] = (
        (0.7 * filtered_movies['genre_score']) + 
        (0.3 * (filtered_movies['avg_rating'] / 5.0))
    )
    
    # Get top recommendations
    recommendations = filtered_movies.sort_values(
        ['final_score', 'avg_rating'], 
        ascending=False
    ).head(top_n)
    
    return recommendations

def main():
    # Load data
    movies_df = load_movie_data()
    model = load_model()
    
    if movies_df.empty:
        st.error("No movie data available. Please check if the data files are properly prepared.")
        return
    
    # App title
    st.title("Movie Recommender")
    
    # Layout with two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Genre selection section
        st.subheader("Select your favorite genres and rate a movie:")
        
        # Get all available genres
        all_genres = get_all_genres(movies_df)
        
        # Initialize session state for selected genres
        if 'selected_genres' not in st.session_state:
            st.session_state.selected_genres = []
        
        # Display genre selection text
        st.markdown("**Favorite genres:**")
        
        # Create a grid of genre buttons (3 columns)
        genre_cols = st.columns(3)
        
        # Add genre buttons
        for i, genre in enumerate(all_genres):
            col_idx = i % 3
            
            # Check if genre is selected
            is_selected = genre in st.session_state.selected_genres
            
            # Create button with appropriate styling
            button_label = f"{genre} ×" if is_selected else genre
            button_type = "primary" if is_selected else "secondary"
            
            if genre_cols[col_idx].button(button_label, key=f"genre_{genre}", type=button_type):
                # Toggle genre selection
                if genre in st.session_state.selected_genres:
                    st.session_state.selected_genres.remove(genre)
                else:
                    st.session_state.selected_genres.append(genre)
                st.rerun()
        
        # Minimum rating slider
        st.markdown("### Minimum Rating:")
        min_rating = st.slider(
            "Select minimum rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            key="min_rating_slider",
            label_visibility="collapsed"
        )
        
        # Generate recommendations button
        if st.button("Get Recommendations", type="primary", use_container_width=True):
            if not st.session_state.selected_genres:
                st.warning("Please select at least one genre to get personalized recommendations.")
            
            # Get recommendations
            recommendations = recommend_movies(
                st.session_state.selected_genres,
                min_rating,
                movies_df,
                model,
                top_n=5
            )
            
            # Store recommendations in session state
            st.session_state.recommendations = recommendations
            
            # Force rerun to display recommendations
            st.rerun()
    
    with col2:
        # Display selected genres as pills
        if st.session_state.selected_genres:
            st.markdown("**Selected genres:**")
            
            # Create HTML for genre pills
            genres_html = ""
            for genre in st.session_state.selected_genres:
                genres_html += f'<div class="genre-pill genre-pill-selected">{genre}</div>'
            
            st.markdown(genres_html, unsafe_allow_html=True)
        
        # Display current minimum rating
        st.markdown(f"**Minimum rating:** {min_rating} ⭐")
    
    # Display recommendations
    st.markdown("### Recommended Movies:")
    
    if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
        recommendations = st.session_state.recommendations
        
        # Create HTML table for recommendations
        table_html = """
        <table class="recommendation-table">
            <tr>
                <th>Title</th>
                <th>Year</th>
                <th>Rating</th>
                <th>Genres</th>
            </tr>
        """
        
        for _, movie in recommendations.iterrows():
            # Format genres as a string
            genres_str = ", ".join(movie['genre_names']) if isinstance(movie['genre_names'], list) else ""
            
            # Add movie to table
            table_html += f"""
            <tr>
                <td class="movie-title">{movie['title']}</td>
                <td>{movie['year']}</td>
                <td>{"⭐" * int(round(movie['avg_rating']))}</td>
                <td class="movie-genres">{genres_str}</td>
            </tr>
            """
        
        table_html += "</table>"
        
        # Display the table
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        # Show placeholder when no recommendations yet
        st.info("Select your favorite genres and minimum rating, then click 'Get Recommendations'")

if __name__ == "__main__":
    main()
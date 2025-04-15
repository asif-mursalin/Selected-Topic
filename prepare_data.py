import numpy as np
import pandas as pd
import os
import pickle
import argparse

# Configure argument parser
parser = argparse.ArgumentParser(description='Process MovieLens 100K dataset for Streamlit app')
parser.add_argument('--data_path', type=str, default='ml-100k', help='Path to MovieLens dataset')
parser.add_argument('--output_dir', type=str, default='models', help='Directory to save processed data')
args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def load_movielens_100k(data_path):
    """
    Load MovieLens 100K dataset from the specified path
    """
    print(f"Loading MovieLens 100K dataset from {data_path}...")
    
    # Load ratings data (u.data file)
    ratings_path = os.path.join(data_path, 'u.data')
    if os.path.exists(ratings_path):
        ratings_df = pd.read_csv(
            ratings_path, 
            sep='\t', 
            names=['userId', 'movieId', 'rating', 'timestamp']
        )
    else:
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}")
    
    # Load movies data (u.item file)
    movies_path = os.path.join(data_path, 'u.item')
    if os.path.exists(movies_path):
        # Define genre columns
        genre_columns = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 
            'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
            'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        movies_df = pd.read_csv(
            movies_path,
            sep='|',
            encoding='latin-1',
            names=['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre_columns,
        )
    else:
        raise FileNotFoundError(f"Movies file not found at {movies_path}")
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    return ratings_df, movies_df

def preprocess_movies(movies_df):
    """
    Preprocess movies data:
    - Extract year from title
    - Process genres into a list format
    """
    print("Preprocessing movies data...")
    
    # Create a copy to avoid modifying the original
    movies = movies_df.copy()
    
    # Extract year from title (format: "Movie Title (1995)")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$', expand=False)
    
    # Convert year to numeric, fill missing with 0
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(0).astype(int)
    
    # Get genre columns (all columns after the first 5)
    genre_cols = movies.columns[5:]  # Skip movieId, title, dates, URL
    
    # Create genre lists based on binary indicators
    def get_genres(row):
        return [genre for genre, value in zip(genre_cols, row[genre_cols]) if value == 1]
    
    movies['genre_names'] = movies.apply(get_genres, axis=1)
    
    print(f"Processed {len(movies)} movies")
    return movies

def prepare_movie_data_for_app(movies_df, ratings_df):
    """
    Prepare movie data in a format suitable for the Streamlit app
    """
    # Create a dataframe with the needed columns
    app_movies = movies_df.copy()
    
    # Keep only the necessary columns
    app_movies = app_movies[['movieId', 'title', 'year', 'genre_names']]
    
    # Rename columns to match app expectations
    app_movies = app_movies.rename(columns={
        'movieId': 'movie_id'
    })
    
    # Calculate average rating for each movie
    movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])
    movie_ratings.columns = ['avg_rating', 'rating_count']
    
    # Map ratings to movies
    app_movies = app_movies.merge(
        movie_ratings, 
        left_on='movie_id', 
        right_index=True, 
        how='left'
    )
    
    # Fill missing ratings
    app_movies['avg_rating'] = app_movies['avg_rating'].fillna(0)
    app_movies['rating_count'] = app_movies['rating_count'].fillna(0).astype(int)
    
    return app_movies

def create_model_info():
    """
    Create a basic model info dictionary
    """
    return {
        'model_type': 'Neural Collaborative Filtering',
        'rmse': 0.923,  # Sample value, you might want to adjust
        'precision': 0.45,
        'recall': 0.39
    }

def save_processed_data(data_dict, output_dir):
    """
    Save all processed data to pickle files
    """
    print(f"Saving processed data to {output_dir}...")
    
    for name, data in data_dict.items():
        file_path = os.path.join(output_dir, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {name} to {file_path}")

def main():
    """
    Main function to execute the data preparation pipeline
    """
    print("="*50)
    print("MOVIELENS 100K DATA PREPARATION")
    print("="*50)
    
    try:
        # Load the dataset
        ratings_df, movies_df = load_movielens_100k(args.data_path)
        
        # Preprocess movies
        processed_movies = preprocess_movies(movies_df)
        
        # Prepare data for the app
        app_movies = prepare_movie_data_for_app(processed_movies, ratings_df)
        
        # Create model info dictionary
        model_info = create_model_info()
        
        # Save processed data
        save_data = {
            'movies_data': app_movies,
            'model_info': model_info
        }
        
        save_processed_data(save_data, args.output_dir)
        
        print("="*50)
        print("DATA PREPARATION COMPLETE")
        print("="*50)
        print(f"Saved processed data to {args.output_dir}")
        print(f"Make sure your pre-trained model file 'best_tf_model_Neural Collaborative Filtering.pkl' is in {args.output_dir}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()
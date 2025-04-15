import numpy as np
import pandas as pd
import os
import time
import pickle
from collections import defaultdict
from scipy.sparse import csr_matrix
import tensorflow as tf
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure argument parser
parser = argparse.ArgumentParser(description='Process MovieLens 100K dataset and train recommendation models')
parser.add_argument('--data_path', type=str, default='ml-100k', help='Path to MovieLens dataset')
parser.add_argument('--output_dir', type=str, default='models', help='Directory to save processed data and models')
parser.add_argument('--min_ratings', type=int, default=5, help='Minimum number of ratings for users and movies')
parser.add_argument('--train_split', type=float, default=0.7, help='Proportion of data for training')
parser.add_argument('--val_split', type=float, default=0.15, help='Proportion of data for validation')
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
        # u.item has 24 columns: movie id, title, release date, video release date,
        # IMDb URL, and 19 genre indicators (1=yes, 0=no)
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
            usecols=['movieId', 'title', 'release_date'] + genre_columns
        )
    else:
        raise FileNotFoundError(f"Movies file not found at {movies_path}")
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    return ratings_df, movies_df

def preprocess_ratings(ratings_df, min_ratings=5):
    """
    Preprocess ratings data:
    - Remove users and movies with fewer than min_ratings
    - Map user and movie IDs to consecutive integers
    """
    print("Preprocessing ratings data...")
    print(f"Initial ratings: {len(ratings_df):,}")
    
    # Create a copy to avoid modifying the original
    ratings = ratings_df.copy()
    
    # Filter out users and movies with too few ratings
    user_counts = ratings['userId'].value_counts()
    movie_counts = ratings['movieId'].value_counts()
    
    active_users = user_counts[user_counts >= min_ratings].index
    popular_movies = movie_counts[movie_counts >= min_ratings].index
    
    filtered_ratings = ratings[
        ratings['userId'].isin(active_users) & 
        ratings['movieId'].isin(popular_movies)
    ]
    
    print(f"Filtered ratings: {len(filtered_ratings):,} ({len(filtered_ratings)/len(ratings):.2%})")
    print(f"Active users: {len(active_users):,} (min {min_ratings} ratings)")
    print(f"Popular movies: {len(popular_movies):,} (min {min_ratings} ratings)")
    
    # Create mappings for user and movie IDs
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(active_users)}
    movie_id_map = {old_id: new_id for new_id, old_id in enumerate(popular_movies)}
    
    # Apply mappings
    filtered_ratings['user_idx'] = filtered_ratings['userId'].map(user_id_map)
    filtered_ratings['movie_idx'] = filtered_ratings['movieId'].map(movie_id_map)
    
    return filtered_ratings, user_id_map, movie_id_map, active_users, popular_movies

def preprocess_movies(movies_df, movie_id_map):
    """
    Preprocess movies data:
    - Filter to only include movies in movie_id_map
    - Extract year from title
    - Process genres
    """
    print("Preprocessing movies data...")
    
    # Create a copy to avoid modifying the original
    movies = movies_df.copy()
    
    # Filter to only include movies in movie_id_map
    movies = movies[movies['movieId'].isin(movie_id_map.keys())].copy()
    
    # Extract year from title (format: "Movie Title (1995)")
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$', expand=False)
    
    # Convert year to numeric, fill missing with 0
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce').fillna(0).astype(int)
    
    # Get genre columns (all columns except the first few)
    genre_cols = movies.columns[3:]  # Skip movieId, title, release_date
    
    # Create genre lists based on binary indicators
    def get_genres(row):
        return [genre for genre, value in zip(genre_cols, row[genre_cols]) if value == 1]
    
    movies['genre_names'] = movies.apply(get_genres, axis=1)
    
    # Map movie IDs to new indices
    movies['movie_idx'] = movies['movieId'].map(movie_id_map)
    
    print(f"Processed {len(movies)} movies")
    return movies

def create_data_splits(ratings, train_split=0.7, val_split=0.15):
    """
    Split ratings data into training, validation, and test sets
    """
    print("Creating data splits...")
    
    # Sort by timestamp to ensure chronological split
    sorted_ratings = ratings.sort_values('timestamp')
    
    # Calculate split indices
    n_ratings = len(sorted_ratings)
    train_idx = int(n_ratings * train_split)
    val_idx = int(n_ratings * (train_split + val_split))
    
    # Split the data
    train_data = sorted_ratings.iloc[:train_idx]
    val_data = sorted_ratings.iloc[train_idx:val_idx]
    test_data = sorted_ratings.iloc[val_idx:]
    
    print(f"Training set: {len(train_data):,} ratings ({len(train_data)/n_ratings:.2%})")
    print(f"Validation set: {len(val_data):,} ratings ({len(val_data)/n_ratings:.2%})")
    print(f"Test set: {len(test_data):,} ratings ({len(test_data)/n_ratings:.2%})")
    
    # Check user overlap to ensure all users in val/test are also in train
    train_users = set(train_data['user_idx'])
    val_users = set(val_data['user_idx'])
    test_users = set(test_data['user_idx'])
    
    print(f"Users in training set: {len(train_users):,}")
    print(f"Users in validation set: {len(val_users):,}")
    print(f"Users in test set: {len(test_users):,}")
    print(f"Users in validation set but not in training: {len(val_users - train_users):,}")
    print(f"Users in test set but not in training: {len(test_users - train_users):,}")
    
    return train_data, val_data, test_data

def create_sparse_matrices(train_data, val_data, test_data, n_users, n_movies):
    """
    Create sparse matrices for training, validation, and test data
    """
    print("Creating sparse matrices (optimized)...")
    
    # Create training matrix
    train_matrix = csr_matrix(
        (train_data['rating'], (train_data['user_idx'], train_data['movie_idx'])),
        shape=(n_users, n_movies)
    )
    
    # Create validation matrix
    val_matrix = csr_matrix(
        (val_data['rating'], (val_data['user_idx'], val_data['movie_idx'])),
        shape=(n_users, n_movies)
    )
    
    # Create test matrix
    test_matrix = csr_matrix(
        (test_data['rating'], (test_data['user_idx'], test_data['movie_idx'])),
        shape=(n_users, n_movies)
    )
    
    print("Created sparse matrices:")
    print(f"Training matrix: {train_matrix.shape}, nonzero: {train_matrix.nnz}")
    print(f"Validation matrix: {val_matrix.shape}, nonzero: {val_matrix.nnz}")
    print(f"Test matrix: {test_matrix.shape}, nonzero: {test_matrix.nnz}")
    
    return train_matrix, val_matrix, test_matrix

def prepare_tf_datasets(train_data, val_data, test_data, batch_size=128):
    """
    Prepare TensorFlow datasets for training neural models
    """
    # Convert to TensorFlow datasets
    def df_to_dataset(df, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices({
            'user_idx': df['user_idx'].values.astype('int32'),
            'movie_idx': df['movie_idx'].values.astype('int32'),
            'rating': df['rating'].values.astype('float32')
        })
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df))
            
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    train_dataset = df_to_dataset(train_data)
    val_dataset = df_to_dataset(val_data, shuffle=False)
    test_dataset = df_to_dataset(test_data, shuffle=False)
    
    return train_dataset, val_dataset, test_dataset

def create_ncf_model(n_users, n_movies, embedding_size=32):
    """
    Create a Neural Collaborative Filtering model
    """
    # Input layers
    user_input = tf.keras.layers.Input(shape=(1,), name='user_idx', dtype='int32')
    movie_input = tf.keras.layers.Input(shape=(1,), name='movie_idx', dtype='int32')
    
    # Embedding layers
    user_embedding = tf.keras.layers.Embedding(
        n_users, embedding_size, name='user_embedding'
    )(user_input)
    movie_embedding = tf.keras.layers.Embedding(
        n_movies, embedding_size, name='movie_embedding'
    )(movie_input)
    
    # Reshape embeddings to flatten them
    user_vector = tf.keras.layers.Flatten()(user_embedding)
    movie_vector = tf.keras.layers.Flatten()(movie_embedding)
    
    # Concatenate embeddings
    concat = tf.keras.layers.Concatenate()([user_vector, movie_vector])
    
    # Dense layers
    dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
    dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)
    
    # Output layer
    output = tf.keras.layers.Dense(1)(dense3)
    
    # Create and compile model
    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    return model

def train_ncf_model(model, train_dataset, val_dataset, epochs=20):
    """
    Train the Neural Collaborative Filtering model
    """
    print("Training Neural Collaborative Filtering model...")
    start_time = time.time()
    
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history, training_time

def evaluate_model(model, test_dataset):
    """
    Evaluate the model on test data
    """
    print("Evaluating model...")
    results = model.evaluate(test_dataset, return_dict=True, verbose=1)
    
    # Calculate RMSE
    rmse = np.sqrt(results['loss'])
    print(f"Test RMSE: {rmse:.4f}")
    
    return rmse, results

def save_processed_data(data_dict, output_dir):
    """
    Save all processed data and models to pickle files
    """
    print(f"Saving processed data to {output_dir}...")
    
    for name, data in data_dict.items():
        file_path = os.path.join(output_dir, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {name} to {file_path}")

def prepare_movie_data_for_app(movies_df, movie_id_map, ratings_df):
    """
    Prepare movie data in a format suitable for the Streamlit app
    """
    # Create a dataframe with the needed columns
    app_movies = movies_df.copy()
    
    # Keep only the necessary columns
    app_movies = app_movies[['movieId', 'title', 'year', 'genre_names', 'movie_idx']]
    
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
    
    return app_movies[['movie_id', 'title', 'year', 'genre_names', 'avg_rating', 'rating_count']]

def main():
    """
    Main function to execute the preprocessing pipeline
    """
    print("="*50)
    print("MOVIELENS 100K PREPROCESSING AND MODEL TRAINING")
    print("="*50)
    
    try:
        # Load the dataset
        ratings_df, movies_df = load_movielens_100k(args.data_path)
        
        # Preprocess ratings
        filtered_ratings, user_id_map, movie_id_map, active_users, popular_movies = preprocess_ratings(
            ratings_df, 
            min_ratings=args.min_ratings
        )
        
        # Preprocess movies
        processed_movies = preprocess_movies(movies_df, movie_id_map)
        
        # Create data splits
        train_data, val_data, test_data = create_data_splits(
            filtered_ratings,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        # Create sparse matrices
        n_users = len(user_id_map)
        n_movies = len(movie_id_map)
        train_matrix, val_matrix, test_matrix = create_sparse_matrices(
            train_data, val_data, test_data, n_users, n_movies
        )
        
        # Prepare TensorFlow datasets
        train_dataset, val_dataset, test_dataset = prepare_tf_datasets(
            train_data, val_data, test_data
        )
        
        # Create and train Neural Collaborative Filtering model
        ncf_model = create_ncf_model(n_users, n_movies)
        trained_model, history, training_time = train_ncf_model(
            ncf_model, train_dataset, val_dataset
        )
        
        # Evaluate the model
        rmse, evaluation_results = evaluate_model(trained_model, test_dataset)
        
        # Prepare data for the app
        app_movies = prepare_movie_data_for_app(processed_movies, movie_id_map, ratings_df)
        
        # Create model info dictionary
        model_info = {
            'model_type': 'Neural Collaborative Filtering',
            'rmse': rmse,
            'n_users': n_users,
            'n_movies': n_movies,
            'training_time': training_time,
            'embedding_size': 32
        }
        
        # Save all data
        save_data = {
            'movies_data': app_movies,
            'user_id_map': user_id_map,
            'movie_id_map': movie_id_map,
            'model_info': model_info,
            'best_tf_model_Neural Collaborative Filtering': trained_model
        }
        
        save_processed_data(save_data, args.output_dir)
        
        print("="*50)
        print("PREPROCESSING AND TRAINING COMPLETE")
        print("="*50)
        print(f"Saved all data and models to {args.output_dir}")
        print(f"Best model: Neural Collaborative Filtering (RMSE: {rmse:.4f})")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()
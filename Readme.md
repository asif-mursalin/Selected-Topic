# MovieLens Recommender System

A Streamlit web application that provides personalized movie recommendations based on the MovieLens 100K dataset. This recommender system uses collaborative filtering and neural network techniques to suggest movies that users might enjoy.

## Prerequisites

- Python 3.8 or higher
- Internet connection (to download the dataset)
- At least 1GB of free disk space

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/asif-mursalin/Selected-Topic.git
cd Selected-Topic
```

### 2. Download the MovieLens Dataset

The application requires the MovieLens 100K dataset. You can download it using curl:

```bash
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

For Windows:

- Visit https://files.grouplens.org/datasets/movielens/ml-100k.zip in your browser
- Save the file to your project directory
- Right-click the zip file and select "Extract All"

This will create a directory called `ml-100k` containing the dataset files.

### 3. Create a Virtual Environment (Optional but Recommended)

```bash
# Or using conda
conda create -n movie-recommender python=3.8
conda activate movie-recommender
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your default web browser. If it doesn't open automatically, the terminal will display a URL that you can copy and paste into your browser.

## Application Structure

The app includes:

- Personalized movie recommendations
- Popular movie suggestions
- Movie browsing by genre
- Movie search functionality
- Detailed movie information pages
- User preference settings

## Model Files (Not recommended due to library conflicts)

When you first run the app, it will look for pre-trained model files in the `models` directory. If you've run the `FINAL_TRAIN.ipynb`, these files should already be available. If not, the app will fall back to using basic recommendation methods.


## Additional Notes

- The first time you run the app, it may take a few moments to load as it processes the dataset
- User authentication in the app is simulated for demonstration purposes
- Any ratings or preferences you set will not be permanently saved when you restart the app
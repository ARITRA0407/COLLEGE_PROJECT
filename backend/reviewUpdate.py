import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import os

# --- 0. CONFIGURATION AND INITIAL SETUP ---

# Define the absolute path to the CSV file based on your directory structure
# WARNING: Ensure this path is correct on the machine running the script.

# Dynamically get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build a relative path to the CSV file (assuming "csv/reviews.csv" is relative to your project root)
CSV_FILE_PATH = os.path.join(BASE_DIR, '..', 'csv', 'reviews.csv')
CSV_FILE_PATH = os.path.normpath(CSV_FILE_PATH)

ASPECT_MAX_SCORE = 10.0
SCORE_SCALING_FACTOR = 0.5  # Controls how quickly aspect scores rise (adjust as needed)

# Download necessary NLTK resources (VADER and stopwords) if not already present
try:
    # Check if resources are downloaded
    _ = SentimentIntensityAnalyzer()
    _ = stopwords.words('english')
except LookupError:
    print("Downloading required NLTK resources (vader_lexicon, stopwords)...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("Download complete.")

# Initialize NLP tools
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# --- 1. ASPECT DICTIONARIES (Custom Keyword Scoring) ---
# Each dictionary maps keywords to a score weight. The model will sum these weights.

ASPECT_KEYWORDS = {
    'professor': {
        'pos': {'helpful': 4, 'knowledgeable': 4, 'experienced': 3, 'supportive': 3, 'excellent teaching': 5, 'friendly': 2},
        'neg': {'unresponsive': 5, 'rude': 4, 'inexperienced': 3, 'unhelpful': 3, 'strict': 2, 'slow': 2},
    },
    'campus': {
        'pos': {'beautiful': 5, 'clean': 4, 'vast': 3, 'modern': 3, 'spacious': 2, 'green': 3, 'safe': 4},
        'neg': {'crowded': 5, 'old': 4, 'small': 3, 'dirty': 3, 'messy': 2, 'unsafe': 4},
    },
    'mess': {
        'pos': {'tasty': 5, 'hygienic': 4, 'variety': 4, 'good food': 3, 'cheap': 2, 'delicious': 4},
        'neg': {'unhygienic': 5, 'bad food': 4, 'tasteless': 3, 'monotonous': 3, 'expensive': 2, 'poor quality': 5},
    },
    'placements': {
        'pos': {'strong': 5, 'high package': 4, 'good companies': 4, '100%': 3, 'supportive cell': 3, 'great opportunities': 5},
        'neg': {'poor': 5, 'no placement': 4, 'less package': 3, 'fake': 2, 'few companies': 2, 'low success': 5},
    },
    'infrastructure': {
        'pos': {'modern labs': 5, 'updated': 4, 'fast wifi': 3, 'good library': 3, 'ample equipment': 2, 'new buildings': 4},
        'neg': {'outdated': 5, 'broken': 4, 'slow wifi': 3, 'insufficient': 2, 'old equipment': 4, 'poor maintenance': 5},
    }
}

# --- 2. CORE PROCESSING FUNCTION ---

def analyze_review(review_text):
    """
    Performs text cleaning and aspect-based scoring on a single review text.
    """
    if pd.isna(review_text) or review_text == "":
        # Return default zero scores for missing or empty reviews
        return {
            'word_count': 0,
            'sentiment_score': 0.0,
            'mess_score': 0.0,
            'professor_score': 0.0,
            'campus_score': 0.0,
            'placements_score': 0.0,
            'infrastructure_score': 0.0,
            'overall_aspect_score': 0.0,
            'rating': 3  # Neutral default rating
        }

    # 2.1 Text Cleaning
    text = str(review_text).lower()
    # Remove punctuation for basic tokenization, but keep it for VADER's sentiment analysis if needed
    clean_text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in clean_text.split() if word not in stop_words and len(word) > 2]
    word_count = len(tokens)

    # 2.2 General Sentiment (VADER)
    sentiment_score = sid.polarity_scores(review_text)['compound']

    # 2.3 Aspect-Based Scoring
    aspect_scores = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        total_pos_weight = 0
        total_neg_weight = 0

        # Check for multi-word phrases first (e.g., 'good food')
        for weight_type, weight_map in keywords.items():
            for phrase, weight in weight_map.items():
                if phrase in text:
                    if weight_type == 'pos':
                        total_pos_weight += weight
                    else:
                        total_neg_weight += weight

        # Check for single-word matches in tokens
        for token in tokens:
            for weight_type, weight_map in keywords.items():
                for keyword, weight in weight_map.items():
                    # Only check single words now
                    if ' ' not in keyword and token == keyword:
                        if weight_type == 'pos':
                            total_pos_weight += weight
                        else:
                            total_neg_weight += weight

        # Calculate raw difference and scale it to a 0-10 score
        raw_score = total_pos_weight - total_neg_weight
        
        # Scale the score: A simple way to map raw_score to 0-10 range is to add a base value (5)
        # and limit the influence of the raw score via the scaling factor.
        # This prevents wildly fluctuating scores while reflecting the sentiment.
        final_score = 5 + (raw_score * SCORE_SCALING_FACTOR)
        
        # Clamp the score between 0 and 10
        final_score = np.clip(final_score, 1, ASPECT_MAX_SCORE)

        aspect_scores[f'{aspect}_score'] = round(final_score, 1)

    # 2.4 Overall Score Calculation
    scores_list = [v for k, v in aspect_scores.items() if 'score' in k]
    if scores_list:
        overall_aspect_score = round(np.mean(scores_list), 1)
    else:
        overall_aspect_score = 0.0

    # 2.5 Rating Calculation (Based on Overall Score)
    if overall_aspect_score >= 8.5:
        rating = 5
    elif overall_aspect_score >= 6.5:
        rating = 4
    elif overall_aspect_score >= 4.5:
        rating = 3
    elif overall_aspect_score >= 2.5:
        rating = 2
    else:
        rating = 1

    return {
        'word_count': word_count,
        'sentiment_score': round(sentiment_score, 2),
        'mess_score': aspect_scores.get('mess_score', 0.0),
        'professor_score': aspect_scores.get('professor_score', 0.0),
        'campus_score': aspect_scores.get('campus_score', 0.0),
        'placements_score': aspect_scores.get('placements_score', 0.0),
        'infrastructure_score': aspect_scores.get('infrastructure_score', 0.0),
        'overall_aspect_score': overall_aspect_score,
        'rating': rating
    }

# --- 3. MAIN EXECUTION ---

def main():
    print(f"--- Starting Review Update Script ---")
    print(f"Target CSV Path: {CSV_FILE_PATH}")

    # 3.1 Load Data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        initial_rows = len(df)
        print(f"Successfully loaded {initial_rows} reviews.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please verify the path and file name.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # 3.2 Apply Analysis to Reviews
    print("Applying NLP analysis and calculating scores...")
    
    # Apply the analyze_review function to the 'review_text' column
    # The result is a Series of dictionaries
    analysis_results = df['review_text'].apply(analyze_review)

    # Convert the Series of dictionaries into a DataFrame of new columns
    df_new_scores = pd.json_normalize(analysis_results)

    # 3.3 Update Existing Columns in the Original DataFrame
    
    # List of columns to update
    update_cols = [
        'rating', 'sentiment_score', 'word_count', 'mess_score', 
        'professor_score', 'campus_score', 'placements_score', 
        'infrastructure_score', 'overall_aspect_score'
    ]

    for col in update_cols:
        if col in df_new_scores.columns:
            df[col] = df_new_scores[col]
        else:
            print(f"Warning: Calculated column '{col}' not found in results.")

    print("Review scores updated successfully.")

    # 3.4 Save the Updated Data
    try:
        df.to_csv(CSV_FILE_PATH, index=False)
        print(f"Successfully saved {len(df)} updated rows back to: {CSV_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred while saving the CSV. Check permissions: {e}")

if __name__ == '__main__':
    main()

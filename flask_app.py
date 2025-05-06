from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# Load and clean data
try:
    df = pd.read_csv('Books.csv')
except FileNotFoundError:
    df = pd.DataFrame()  # Empty DataFrame if file not found

def clean_number(x):
    if pd.isnull(x):
        return 0
    if isinstance(x, str):
        return int(x.replace(',', '').strip())
    return int(x)

if not df.empty:
    df['Num_Ratings'] = df['Num_Ratings'].apply(clean_number)
    df['Avg_Rating'] = pd.to_numeric(df['Avg_Rating'], errors='coerce').fillna(0)

    def parse_genres(x):
        try:
            return ast.literal_eval(x)
        except:
            return []
    df['Genres'] = df['Genres'].apply(parse_genres)
    df = df.fillna("")

    # Add new columns for analytics
    df['title_length'] = df['Book'].apply(len)
    df['sentiment'] = df['Description'].apply(lambda x: 'positive' if TextBlob(str(x)).sentiment.polarity > 0 else ('negative' if TextBlob(str(x)).sentiment.polarity < 0 else 'neutral'))

    # Compute the minimum number of ratings for the top 10 most rated books
    top10_min_ratings = df.sort_values(by='Num_Ratings', ascending=False).head(10)['Num_Ratings'].min()

    # Vectorize genres for cosine similarity
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['Genres'])

# Helper: filter books
def filter_books(search=None):
    if search:
        mask = df['Book'].str.lower().str.contains(search.lower()) | df['Author'].str.lower().str.contains(search.lower())
        return df[mask]
    return df

# Helper: recommend similar books
def recommend_books_cosine(title, n=5):
    try:
        idx = df.index[df['Book'] == title][0]
    except IndexError:
        return pd.DataFrame()
    sim_scores = cosine_similarity([genre_matrix[idx]], genre_matrix)[0]
    sim_scores[idx] = -1  # exclude itself
    top_idx = np.argsort(sim_scores)[-n:][::-1]
    return df.iloc[top_idx]

# === API ROUTES ===

@app.route('/api/books')
def get_books():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    search = request.args.get('search')
    books = filter_books(search)
    return jsonify(books.to_dict(orient='records'))

@app.route('/api/book/<title>')
def get_book(title):
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    book = df[df['Book'] == title]
    if book.empty:
        return jsonify({'error': 'Book not found'}), 404
    book_data = book.iloc[0].to_dict()
    recs = recommend_books_cosine(title, n=5)
    book_data['recommendations'] = recs.fillna("").to_dict(orient='records')
    return jsonify(book_data)

@app.route('/api/top-rated')
def top_rated():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    top = df.sort_values(by='Avg_Rating', ascending=False).head(20)
    return jsonify(top.to_dict(orient='records'))

@app.route('/api/genre-distribution')
def genre_distribution():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    exploded = df.explode('Genres')
    genre_counts = exploded['Genres'].value_counts().to_dict()
    return jsonify(genre_counts)

@app.route('/api/top-10-most-rated')
def top_10_most_rated():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    top_10 = df.sort_values(by='Num_Ratings', ascending=False).head(10)[['Book', 'Num_Ratings']]
    if top_10.empty:
        return jsonify({'error': 'No books with ratings available'}), 404
    return jsonify(top_10.to_dict(orient='records'))

# Existing analytics endpoints

@app.route('/api/top-authors-by-books')
def top_authors_by_books():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    author_stats = df.groupby('Author').agg({'Book': 'size', 'Avg_Rating': 'mean'}).reset_index()
    author_stats.columns = ['Author', 'book_count', 'avg_rating']
    top_by_books = author_stats.sort_values(by='book_count', ascending=False).head(10)
    return jsonify(top_by_books.to_dict(orient='records'))

@app.route('/api/top-authors-by-rating')
def top_authors_by_rating():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    author_stats = df.groupby('Author').agg({'Book': 'size', 'Avg_Rating': 'mean'}).reset_index()
    author_stats.columns = ['Author', 'book_count', 'avg_rating']
    filtered = author_stats[author_stats['book_count'] >= 5]
    top_by_rating = filtered.sort_values(by='avg_rating', ascending=False).head(10)
    return jsonify(top_by_rating.to_dict(orient='records'))

@app.route('/api/ratings-histogram')
def ratings_histogram():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    hist, bin_edges = np.histogram(df['Avg_Rating'], bins=bins)
    return jsonify({'hist': hist.tolist(), 'bin_edges': bin_edges.tolist()})

# New endpoint for top 10 rated books
@app.route('/api/top-10-rated')
def top_10_rated():
    if df.empty:
        return jsonify({'error': 'Books data not available'}), 500
    top_10 = df.sort_values(by='Avg_Rating', ascending=False).head(10)[['Book', 'Avg_Rating', 'Num_Ratings']]
    return jsonify(top_10.to_dict(orient='records'))
if __name__ == '__main__':
    app.run(debug=True)


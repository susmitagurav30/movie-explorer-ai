import pandas as pd
import requests
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

API_KEY = "76eed494b29bdbc790c5abf90a5da97d"

# -------- Fetch Poster --------
def get_movie_data(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return "https://via.placeholder.com/300x450"

        data = response.json()

        if data.get('results'):
            movie = data['results'][0]
            if movie.get('poster_path'):
                return f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"

    except Exception as e:
        print("API ERROR:", e)

    return "https://via.placeholder.com/300x450"


# -------- Load Dataset --------
data = pd.read_csv("Movie_Review.csv")
data.columns = data.columns.str.strip()

data['IMDb Rating'] = pd.to_numeric(data['IMDb Rating'], errors='coerce')
data['Movie Name'] = data['Movie Name'].fillna('')
data['Genre'] = data['Genre'].fillna('')
data['Language'] = data['Language'].fillna('')


# -------- Recommendation Model --------
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(data['Genre'])
similarity = cosine_similarity(genre_matrix)


def recommend_movies(movie_name):
    idx = data[data['Movie Name'] == movie_name].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rec_movies = []
    for i in scores[1:6]:
        rec_movies.append(data.iloc[i[0]].to_dict())

    return rec_movies


# -------- Home --------
@app.route('/')
def home():
    movies = data.drop_duplicates(subset=['Movie Name']).to_dict(orient='records')

    for m in movies[:12]:
        m['poster'] = get_movie_data(m['Movie Name'])
        
    movies_with_poster = [m for m in movies if m.get('poster')]
    movies_without_poster = [m for m in movies if not m.get('poster')]

    movies = movies_with_poster + movies_without_poster



    return render_template('index.html', movies=movies, query="")


# -------- Search + Filters --------
@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('movie', '').strip()
    genre = request.form.get('genre')
    language = request.form.get('language')
    rating = request.form.get('rating')

    results = data.copy()

    # 🔍 Search
    if query:
        results = results[results['Movie Name'].str.lower().str.contains(query.lower(), na=False)]

    # 🎭 Genre
    if genre:
        results = results[results['Genre'].str.contains(genre, case=False, na=False)]

    # 🌐 Language
    if language:
        results = results[results['Language'].str.contains(language, case=False, na=False)]

    # ⭐ Rating
    if rating:
        results = results[results['IMDb Rating'] >= float(rating)]

    results = results.drop_duplicates(subset=['Movie Name'])

    movies = results.to_dict(orient='records')

    for m in movies[:12]:
        m['poster'] = get_movie_data(m['Movie Name'])

    return render_template('index.html', movies=movies, query=query,  genre=genre, language=language, rating=rating)


# -------- Run --------
if __name__ == "__main__":
    app.run(debug=True)
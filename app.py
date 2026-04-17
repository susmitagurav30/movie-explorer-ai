import requests

API_KEY = "76eed494b29bdbc790c5abf90a5da97d"

def get_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
    data = requests.get(url).json()

    for vid in data['results']:
        if vid['type'] == 'Trailer':
            return f"https://www.youtube.com/watch?v={vid['key']}"
    return ""

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

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Movie_Review.csv")
data['Movie Name'] = data['Movie Name'].fillna('')
data['Genre'] = data['Genre'].fillna('')

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(data['Genre'])

similarity = cosine_similarity(genre_matrix)

def recommend(movie_name):
    idx = data[data['Movie Name'] == movie_name].index

    if len(idx) == 0:
        return []

    idx = idx[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rec_movies = []
    for i in scores[1:5]:
        rec_movies.append(data.iloc[i[0]].to_dict())

    return rec_movies

# home
@app.route('/')
def home():
   movies = data.drop_duplicates(subset=['Movie Name']).to_dict(orient='records')
   for m in movies[:6]:
        m['poster'] = get_movie_data(m['Movie Name'])
        
   return render_template('index.html', movies=movies)     

@app.route('/search', methods=['POST'])
def search():
    query = request.form['movie'].strip()

    if query == "":
        results = data.copy()
    else:
        results = data[data['Movie Name'].str.contains(query, case=False, na=False)]

    results = results.drop_duplicates(subset=['Movie Name'])

    movies = results.to_dict(orient='records')
    for m in movies[:6]:
        poster = get_movie_data(m['Movie Name'])
        m['poster'] = poster
        
    return render_template('index.html', movies=movies, query=query)


@app.route('/recommend', methods=['POST'])
def recommend(movie_name):
    # 🔥 find closest match (partial search)
    matches = data[data['Movie Name'].str.contains(movie_name, case=False, na=False)]

    if len(matches) == 0:
        return []

    # take first matched movie
    movie_name = matches.iloc[0]['Movie Name']

    idx = data[data['Movie Name'] == movie_name].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    rec_movies = []
    for i in scores[1:6]:
        rec_movies.append(data.iloc[i[0]].to_dict())

    return rec_movies
if __name__ == "__main__":
    app.run(debug=True)




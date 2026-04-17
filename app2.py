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
    results = data.drop_duplicates(subset=['Movie Name'])

    return render_template(
        'index.html',
        table=results.to_html(classes='table table-bordered', index=False)
    )


@app.route('/search', methods=['POST'])
def search():
    query = request.form['movie'].strip()

    if query == "":
        # ✅ show all movies
        results = data.copy()
    else:
        results = data[data['Movie Name'].str.lower().str.contains(query.lower(), na=False)]

    # remove duplicates
    results = results.drop_duplicates(subset=['Movie Name'])

    return render_template(
        'index.html',
        table=results.to_html(classes='table table-bordered', index=False)
    )

@app.route('/recommend', methods=['POST'])
def recommend_route():
    movie = request.form['movie']
    recs = recommend(movie)

    return render_template('index.html', movies=recs)

if __name__ == "__main__":
    app.run(debug=True)

# @app.route('/recommend', methods=['POST'])
# def recommend_route():
#     movie = request.form['movie']
#     recs = recommend(movie)

#     for m in recs:
#         poster = get_movie_data(m['Movie Name'])
#         m['poster'] = poster
       

#     return render_template('index.html', movies=recs)
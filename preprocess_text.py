import pandas as pd

data = pd.read_csv("Movie_Review.csv")

print(data.head())
print(data.columns)
data['Genre'] = data['Genre'].fillna('')
data['Movie Name'] = data['Movie Name'].fillna('')

# Create Search Feature

def search_movie(query):
    results = data[data['Movie Name'].str.contains(query, case=False)]
    return results[['Movie Name', 'Genre', 'IMDb Rating']]

#Create Recommendation System

#Convert Genre → Numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(data['Genre'])

similarity = cosine_similarity(genre_matrix)

#Recommendation Function
def recommend(movie_name):
    index = data[data['Movie Name'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = [data.iloc[i[0]]['Movie Name'] for i in scores[1:6]]

    return top_movies


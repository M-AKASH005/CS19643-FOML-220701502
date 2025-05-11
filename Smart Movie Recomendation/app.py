from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv('movies.csv')
df.fillna('', inplace=True)
df['Series_Title_lower'] = df['Series_Title'].str.lower()

# Feature combination
df['features'] = df['Overview']+' '+df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']
vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['features'])
similarity_matrix = cosine_similarity(feature_matrix)

def recommend(movie_name):
    movie_name = movie_name.strip().lower()
    if movie_name not in df['Series_Title_lower'].values:
        return [], "Movie not found in the dataset."

    idx = df[df['Series_Title_lower'] == movie_name].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:7]
    movie_indices = [i[0] for i in sim_scores]

    results = df.iloc[movie_indices][[
        'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre',
        'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
        'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'
    ]].fillna('N/A')

    results['No_of_Votes'] = results['No_of_Votes'].astype(str)
    results['Gross'] = results['Gross'].astype(str)

    return results.to_dict(orient='records'), None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_route():
    movie_name = request.form['movie_name']
    recommendations, error = recommend(movie_name)

    # Default to None if movie not found
    searched_movie = None

    # If movie exists in the dataset, extract its full info
    if error is None:
        searched_movie_row = df[df['Series_Title_lower'] == movie_name.strip().lower()]
        if not searched_movie_row.empty:
            searched_movie = searched_movie_row.iloc[0][[
                'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre',
                'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
                'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'
            ]].fillna('N/A').to_dict()
            searched_movie['No_of_Votes'] = str(searched_movie['No_of_Votes'])
            searched_movie['Gross'] = str(searched_movie['Gross'])

    return render_template(
        'result.html',
        movie_name=movie_name,
        searched_movie=searched_movie,
        recommendations=recommendations,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)

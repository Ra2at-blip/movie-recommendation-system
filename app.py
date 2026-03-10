import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load and merge datasets 
movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
credits = pd.read_csv("datasets/tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

#  Safe extractor function 
def safe_extract(data, key=None, first_only=True):
    """
    Safely extract 'name' from list of dicts or string representation of list of dicts.
    If key is provided (tuple like ('job','Director')), filters by that key-value.
    """
    # Convert string to list of dicts if needed
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except:
            return "" if first_only else []

    if not isinstance(data, list):
        return "" if first_only else []

    # Filter by key if given
    if key:
        k, v = key
        data = [item['name'] for item in data if item.get(k) == v]
    else:
        data = [item['name'] for item in data if 'name' in item]

    if first_only:
        return data[0] if data else ""
    return data

#  Apply extraction to columns 
movies['director'] = movies['crew'].apply(lambda x: safe_extract(x, key=('job','Director')))
movies['cast'] = movies['cast'].apply(lambda x: safe_extract(x, first_only=False)[:3])  # top 3 cast
movies['genres'] = movies['genres'].apply(lambda x: safe_extract(x, first_only=False))
movies['keywords'] = movies['keywords'].apply(lambda x: safe_extract(x, first_only=False))

#  Preprocess columns 
def list_to_string(lst):
    if isinstance(lst, list):
        return " ".join(lst)
    return ""

movies['cast'] = movies['cast'].apply(list_to_string)
movies['genres'] = movies['genres'].apply(list_to_string)
movies['keywords'] = movies['keywords'].apply(list_to_string)
movies['director'] = movies['director'].apply(str)

# Create a combined 'tags' column
movies['overview'] = movies['overview'].fillna("")  # make sure no NaN
movies['tags'] = (movies['overview'] + " " +
                  movies['genres'] + " " +
                  movies['keywords'] + " " +
                  movies['cast'] + " " +
                  movies['director'])
movies['tags'] = movies['tags'].str.lower()

#  Vectorization & similarity 
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

#  Recommendation function 
def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found!"]
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended = [movies.iloc[i[0]].title for i in movies_list]
    return recommended

#  Streamlit UI 
st.title("Movie Recommender System")
movie_input = st.text_input("Enter a movie name:")
if st.button("Recommend"):
    recommendations = recommend(movie_input)
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
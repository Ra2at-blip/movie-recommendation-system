import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading datasets
movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
credits = pd.read_csv("datasets/tmdb_5000_credits.csv")

# merging datasets
movies = movies.merge(credits, on='title')

# selecting important features
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# removing missing values
movies.dropna(inplace=True)

# combine text features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# convert text to vector
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# building similarity matrix
similarity = cosine_similarity(vectors)

def recommend(movie):

    if movie not in movies['title'].values:
        print("Movie not found in dataset.")
        return

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(movies.iloc[i[0]].title) 
        
#taking inputs        
movie_name = input("Enter a movie name: ")
recommend(movie_name)
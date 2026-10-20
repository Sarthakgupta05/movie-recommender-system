import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="CineMatch | Movie Recommender", page_icon="🎬", layout="wide")

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "/tmp/ml-latest-small"
ZIP_PATH = "/tmp/movielens.zip"


@st.cache_data
def load_data():
    if not (os.path.exists(f"{DATA_DIR}/movies.csv") and os.path.exists(f"{DATA_DIR}/ratings.csv")):
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall("/tmp")
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies = movies.drop_duplicates("movieId").dropna(subset=["title"]).copy()
    movies["genres"] = movies["genres"].fillna("")
    return movies, ratings


@st.cache_resource
def build_models(movies, ratings):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    content_matrix = vectorizer.fit_transform(movies["genres"])
    user_item = ratings.pivot_table(index="movieId", columns="userId", values="rating", aggfunc="mean")
    cf_movie_ids = user_item.index.to_numpy()
    rating_matrix = csr_matrix(user_item.fillna(0).to_numpy(dtype=np.float32))
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(rating_matrix)
    movie_to_row = {int(mid): row for row, mid in enumerate(movies["movieId"])}
    cf_movie_to_row = {int(mid): row for row, mid in enumerate(cf_movie_ids)}
    return content_matrix, knn, rating_matrix, cf_movie_ids, movie_to_row, cf_movie_to_row


def normalize(scores):
    scores = np.asarray(scores, dtype=float)
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span > 1e-12 else np.zeros_like(scores)


def recommend(title, count, movies, models):
    content_matrix, knn, rating_matrix, cf_ids, movie_to_row, cf_to_row = models
    match = movies[movies["title"].str.casefold() == title.casefold()]
    if match.empty:
        return pd.DataFrame()
    movie_id = int(match.iloc[0]["movieId"])
    row = movie_to_row[movie_id]
    content_scores = cosine_similarity(content_matrix[row], content_matrix).ravel()
    collaborative_scores = np.zeros(len(movies), dtype=float)
    if movie_id in cf_to_row:
        distances, indices = knn.kneighbors(
            rating_matrix[cf_to_row[movie_id]], n_neighbors=min(51, len(cf_ids))
        )
        for distance, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_id = int(cf_ids[neighbor_idx])
            if neighbor_id in movie_to_row:
                target = movie_to_row[neighbor_id]
                collaborative_scores[target] = max(collaborative_scores[target], 1.0 - float(distance))
    scores = 0.6 * normalize(content_scores) + 0.4 * normalize(collaborative_scores)
    scores[row] = -1
    result = movies.copy()
    result["score"] = scores
    result["year"] = result["title"].str.extract(r"\((\d{4})\)\s*$")
    result["display_title"] = result["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return result.nlargest(count, "score")[["display_title", "year", "genres", "score"]].reset_index(drop=True)


def popular_movies(movies, ratings, minimum=20, count=10):
    stats = ratings.groupby("movieId")["rating"].agg(average_rating="mean", rating_count="count").reset_index()
    result = movies.merge(stats, on="movieId", how="left")
    result["rating_count"] = result["rating_count"].fillna(0)
    result["average_rating"] = result["average_rating"].fillna(0)
    result = result[result["rating_count"] >= minimum].copy()
    global_mean = ratings["rating"].mean()
    m = 20
    result["weighted_rating"] = (
        result["rating_count"] / (result["rating_count"] + m) * result["average_rating"]
        + m / (result["rating_count"] + m) * global_mean
    )
    result["year"] = result["title"].str.extract(r"\((\d{4})\)\s*$")
    result["display_title"] = result["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    return result.nlargest(count, "weighted_rating")[
        ["display_title", "year", "genres", "average_rating", "rating_count"]
    ].reset_index(drop=True)


st.title("🎬 CineMatch")
st.caption("Hybrid Movie Recommendation System")
st.write("Find movies using genre similarity and user-rating patterns. Dataset: MovieLens Latest Small (GroupLens Research).")

try:
    with st.spinner("Loading MovieLens data and preparing models..."):
        movies, ratings = load_data()
        models = build_models(movies, ratings)
except Exception as exc:
    st.error("The dataset or ML model could not be loaded.")
    st.exception(exc)
    st.stop()

tab1, tab2 = st.tabs(["🎥 Find Similar Movies", "🔥 Popular Movies"])

with tab1:
    st.subheader("Recommend movies based on a movie you like")
    titles = sorted(movies["title"].tolist())
    default_index = titles.index("Toy Story (1995)") if "Toy Story (1995)" in titles else 0
    selected_movie = st.selectbox("Choose a movie", titles, index=default_index)
    count = st.slider("Number of recommendations", 5, 20, 10)
    if st.button("Recommend Movies", type="primary"):
        with st.spinner("Finding recommendations..."):
            result = recommend(selected_movie, count, movies, models)
        if result.empty:
            st.warning("Movie not found. Please select another title.")
        else:
            st.success(f"Movies similar to {selected_movie}")
            result["score"] = result["score"].round(3)
            st.dataframe(result, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Popular movies with reliable ratings")
    minimum = st.slider("Minimum number of ratings", 5, 100, 20)
    result = popular_movies(movies, ratings, minimum=minimum)
    result["average_rating"] = result["average_rating"].round(2)
    st.dataframe(result, use_container_width=True, hide_index=True)

st.divider()
st.caption("Content-based filtering uses TF-IDF and cosine similarity. Collaborative filtering uses K-nearest neighbors on user-rating patterns. Hybrid score: 60% content + 40% collaborative.")

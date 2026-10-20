# CineMatch — Hybrid Movie Recommendation System

A Streamlit ML project using the MovieLens Latest Small dataset.

## Features
- Content-based recommendations using TF-IDF and cosine similarity
- Collaborative filtering using K-nearest neighbors
- Hybrid recommendation score (60% content + 40% collaborative)
- Popular movies ranked using average ratings and rating counts

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Go to https://share.streamlit.io/
3. Connect GitHub and choose this repository, branch `main`, and main file `app.py`.
4. Click Deploy.

The app downloads MovieLens Latest Small automatically on startup; do not upload the dataset files.

Dataset: MovieLens Latest Small, GroupLens Research:
https://grouplens.org/datasets/movielens/latest/

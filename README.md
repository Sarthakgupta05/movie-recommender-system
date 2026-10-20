# 🎬 CineMatch — Hybrid Movie Recommendation System

CineMatch is a Machine Learning project that recommends movies by combining **content-based filtering** and **collaborative filtering**. It uses the MovieLens Latest Small dataset and provides an interactive web interface built with Streamlit.

## ✨ Features

- **Content-based filtering:** Uses TF-IDF to represent movie genres and cosine similarity to find similar movies.
- **Collaborative filtering:** Uses K-nearest neighbors (KNN) to identify movies with similar user-rating patterns.
- **Hybrid recommendations:** Combines normalized content and collaborative scores using a 60:40 weighting.
- **Popular movies:** Displays highly rated movies while considering the number of ratings.
- **Interactive interface:** Select a movie, choose the number of recommendations, and explore results in Streamlit.
- **Automatic dataset download:** Downloads MovieLens data when the app starts; dataset files do not need to be committed to GitHub.

## 🧰 Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Streamlit

## 🗂️ Project Structure

```text
movie-recommender-system/
├── app.py
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/movie-recommender-system.git
cd movie-recommender-system
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

Streamlit will show a local URL in your terminal, usually `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and choose **Create app**.
4. Select your repository, the `main` branch, and `app.py` as the main file.
5. Click **Deploy**.

The app downloads the dataset automatically at startup. Do **not** upload the dataset ZIP or commit API keys, passwords, or other secrets.

## 🧠 How It Works

1. Movie genres are converted into TF-IDF vectors.
2. Cosine similarity compares the genre vectors of movies.
3. KNN compares movies using user-rating patterns.
4. The two normalized scores are combined:

   **Hybrid Score = 0.6 × Content Similarity + 0.4 × Collaborative Similarity**

5. The selected movie is excluded, and the highest-scoring alternatives are shown.

## 📊 Dataset

This project uses **MovieLens Latest Small**, published by GroupLens Research at the University of Minnesota.

- [MovieLens dataset page](https://grouplens.org/datasets/movielens/latest/)
- `movies.csv`: movie IDs, titles, and genres.
- `ratings.csv`: user IDs, movie IDs, and ratings.

The dataset is downloaded by the app; it is not stored in this repository.

## ⚠️ Notes and Limitations

- Recommendations depend on the available movie genres and user ratings.
- New movies and users with little or no data may receive less relevant recommendations.
- Missing ratings are filled with zero in the collaborative-filtering matrix, which is a simplified baseline and may affect similarity scores.
- The 60:40 weighting is a project choice, not a guarantee of optimal recommendation quality.

## 📄 License and Attribution

The code in this repository is intended for educational and demonstration purposes. Review the MovieLens dataset terms on the [official GroupLens website](https://grouplens.org/datasets/movielens/) before redistributing the dataset.

---

**Built with Python, Scikit-learn, and Streamlit.**

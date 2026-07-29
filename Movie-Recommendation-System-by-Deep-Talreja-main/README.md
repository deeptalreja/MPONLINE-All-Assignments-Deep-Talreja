# 🎬 Movie Recommendation System

A content-based movie recommendation web application built with **Python**, **Flask**, **Pandas**, and **Scikit-learn**. Recommends similar movies based on genre similarity using **TF-IDF Vectorization** and **Cosine Similarity**, presented through a modern cinema-themed interface.

---

## 👤 Developer Profile

- **Name:** Deep Talreja
- **Application Number:** IN26010914
- **Registration Number:** 23BCE11003
- **Internship:** MP Online AI/ML Internship

---

## 🔗 Project Links

- **Live Web Application:** _add your Render URL here after deployment_
- **GitHub Repository:** _add your repo URL here_

---

## 📌 Project Overview

The Movie Recommendation System helps users discover movies similar to a title they already enjoy. It uses a content-based filtering approach — analyzing genre overlap between movies and calculating similarity scores to surface the closest matches.

---

## 🚀 Features

- Genre-based content recommendation engine
- Cinema/theatrical themed web interface with animated marquee header
- Match-percentage scoring shown per recommendation
- Fast, in-memory similarity computation — no external API calls
- Ready for deployment on Render (Procfile + Gunicorn)

---

## 🛠️ Tech Stack

- Python 3
- Flask
- Pandas
- NumPy
- Scikit-learn (TF-IDF, cosine similarity)
- HTML5 / CSS3 (embedded, cinema theme)
- Gunicorn
- Render

---

## 📂 Project Structure

```
movie-recommender/
├── templates/
│   └── index.html         # Cinema-themed frontend (CSS embedded inline)
├── app.py                 # Flask backend & /recommend endpoint
├── recommender.py         # TF-IDF + cosine similarity recommendation engine
├── movies.csv              # Curated movie dataset (title, genres)
├── requirements.txt
├── Procfile                 # Render/Heroku start command
├── run.sh / run.bat          # One-command local setup scripts
└── .gitignore
```

> **Note on the dataset:** this project uses a curated dataset of 100 well-known movies with their real genres, formatted the same way as the MovieLens dataset (`movieId, title, genres`). If you'd like a larger dataset, you can download the MovieLens Latest Small Dataset from [GroupLens](https://grouplens.org/datasets/movielens/) and swap in `movies.csv` with matching column names — the recommender code will work unchanged.

---

## 🧠 Recommendation Algorithm

This project uses **Content-Based Filtering**.

### Workflow

1. Load the movie dataset (`movies.csv`)
2. Extract each movie's genre list
3. Convert genres into TF-IDF vectors
4. Compute cosine similarity between every pair of movies
5. For a selected movie, return the top 5 most similar titles by similarity score

---

## ⚙️ Installation & Setup

Clone the repository, then:

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux/macOS**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Quick Start (Automated)

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```bat
run.bat
```

### Manual

```bash
python app.py
```

Open your browser and visit `http://127.0.0.1:5004`.

---

## 🚀 API Documentation

### `GET /`
Renders the frontend with a movie selection dropdown populated from the dataset.

### `POST /recommend`

**Headers:** `Content-Type: application/json`

**Request body:**
```json
{
  "title": "The Dark Knight (2008)"
}
```

**Response (200 OK):**
```json
{
  "title": "The Dark Knight (2008)",
  "recommendations": [
    { "title": "The Departed (2006)", "genres": "Crime, Drama, Thriller", "similarity": 0.902 },
    { "title": "No Country for Old Men (2007)", "genres": "Crime, Drama, Thriller", "similarity": 0.902 }
  ]
}
```

---

## 🌐 Deployment (Render)

1. Push this repository to GitHub
2. Create a new **Web Service** on [Render](https://render.com), connect your repo
3. Render detects the `Procfile` automatically, or set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Deploy — Render builds and serves the app publicly

---

## 📈 Future Improvements

- Movie poster images (via TMDb API integration)
- Search autocomplete instead of a dropdown
- User rating-based collaborative filtering
- Personalized watchlists
- Responsive mobile layout refinements

---

## 📄 License

This project is developed for educational and learning purposes as part of the MP Online AI/ML Internship.

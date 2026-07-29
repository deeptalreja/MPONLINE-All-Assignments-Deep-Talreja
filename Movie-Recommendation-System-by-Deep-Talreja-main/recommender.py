"""
recommender.py
Content-based movie recommendation engine. Builds a TF-IDF vector
space over movie genres and recommends similar titles using cosine
similarity.

Author: Deep Talreja
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "movies.csv")


class MovieRecommender:
    def __init__(self, data_path: str = DATA_PATH):
        self.movies = pd.read_csv(data_path)

        # Genres are pipe-separated (e.g. "Action|Adventure|Sci-Fi") —
        # convert to space-separated text so TF-IDF treats each genre
        # as its own token.
        genre_text = self.movies["genres"].str.replace("|", " ", regex=False)

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf_matrix = self.vectorizer.fit_transform(genre_text)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"]
        )

    def get_all_titles(self) -> list:
        """Returns every movie title, sorted alphabetically, for populating
        a selection dropdown in the UI."""
        return sorted(self.movies["title"].tolist())

    def recommend(self, title: str, top_n: int = 5) -> list:
        """Returns the top_n movies most similar to the given title,
        based on genre overlap."""
        if title not in self.title_to_index:
            raise ValueError(f"Movie '{title}' not found in the dataset.")

        idx = self.title_to_index[title]
        scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity score, descending, excluding the movie itself
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]

        recommendations = []
        for movie_idx, score in scores:
            row = self.movies.iloc[movie_idx]
            recommendations.append({
                "title": row["title"],
                "genres": row["genres"].replace("|", ", "),
                "similarity": round(float(score), 3),
            })

        return recommendations


if __name__ == "__main__":
    # Quick manual sanity check when run directly
    engine = MovieRecommender()
    sample_title = "The Dark Knight (2008)"
    print(f"Recommendations for '{sample_title}':")
    for rec in engine.recommend(sample_title):
        print(f"  - {rec['title']} ({rec['genres']}) — similarity: {rec['similarity']}")

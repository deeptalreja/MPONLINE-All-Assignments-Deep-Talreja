"""
app.py
Flask web API + UI server for the Movie Recommendation System.

Author: Deep Talreja
"""

from flask import Flask, request, jsonify, render_template
from recommender import MovieRecommender

app = Flask(__name__)
engine = MovieRecommender()


@app.route("/", methods=["GET"])
def home():
    """Renders the front-end dashboard with a movie selection dropdown."""
    titles = engine.get_all_titles()
    return render_template("index.html", titles=titles)


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Accepts a JSON payload with a movie title and returns the top
    similar movies based on genre content.

    Expected payload:
    {
        "title": "The Dark Knight (2008)"
    }
    """
    data = request.get_json(force=True, silent=True)

    if not data or "title" not in data:
        return jsonify({"error": "Missing 'title' field"}), 400

    title = data["title"]

    try:
        recommendations = engine.recommend(title, top_n=5)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    return jsonify({"title": title, "recommendations": recommendations})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5004))
    app.run(host="0.0.0.0", port=port, debug=False)

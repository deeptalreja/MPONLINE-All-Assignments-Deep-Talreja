"""
app.py
Flask web API + UI server for the Car Price Predictor.

Author: Deep Talreja
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")
model = joblib.load(MODEL_PATH)

FUEL_MAP = {"Petrol": 0, "Diesel": 1, "CNG": 2}
SELLER_MAP = {"Dealer": 0, "Individual": 1}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}


@app.route("/", methods=["GET"])
def home():
    """Renders the front-end dashboard for interactive predictions."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON payload describing a used car and returns the
    model's predicted resale price (in Lakhs).

    Expected payload:
    {
        "present_price": 8.5,
        "kms_driven": 45000,
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": 0,
        "car_age": 4
    }
    """
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    required_fields = [
        "present_price", "kms_driven", "fuel_type",
        "seller_type", "transmission", "owner", "car_age",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    fuel_type = FUEL_MAP.get(data["fuel_type"])
    seller_type = SELLER_MAP.get(data["seller_type"])
    transmission = TRANSMISSION_MAP.get(data["transmission"])

    if fuel_type is None:
        return jsonify({"error": f"fuel_type must be one of {list(FUEL_MAP)}"}), 400
    if seller_type is None:
        return jsonify({"error": f"seller_type must be one of {list(SELLER_MAP)}"}), 400
    if transmission is None:
        return jsonify({"error": f"transmission must be one of {list(TRANSMISSION_MAP)}"}), 400

    try:
        features = np.array([[
            float(data["present_price"]),
            float(data["kms_driven"]),
            fuel_type,
            seller_type,
            transmission,
            int(data["owner"]),
            int(data["car_age"]),
        ]])
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric fields must contain valid numbers"}), 400

    predicted_price = float(model.predict(features)[0])
    predicted_price = max(predicted_price, 0.1)  # guard against negative predictions

    return jsonify({"predicted_price_lakhs": round(predicted_price, 2)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)

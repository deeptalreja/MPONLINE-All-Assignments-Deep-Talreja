"""
train.py
Trains a Random Forest Regressor to predict used-car resale prices.

Primary data source: CarDekho Vehicle dataset (place car_data.csv in a
`data/` folder next to this script — download it from Kaggle:
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

If no CSV is found, a synthetic dataset with the same schema and
realistic price relationships is generated instead, so the pipeline
still runs end-to-end out of the box.

Author: Deep Talreja
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "car_data.csv")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")

FUEL_MAP = {"Petrol": 0, "Diesel": 1, "CNG": 2}
SELLER_MAP = {"Dealer": 0, "Individual": 1}
TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}

FEATURE_COLUMNS = [
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Car_Age",
]


def _generate_synthetic_dataset(n_rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic dataset mirroring the CarDekho schema, with
    realistic relationships between features and resale price, for use
    when the real Kaggle CSV isn't available."""
    rng = np.random.default_rng(seed)

    present_price = rng.uniform(2.0, 25.0, n_rows)          # Lakhs
    kms_driven = rng.integers(500, 150000, n_rows)
    fuel_type = rng.choice(list(FUEL_MAP.values()), n_rows, p=[0.55, 0.40, 0.05])
    seller_type = rng.choice(list(SELLER_MAP.values()), n_rows, p=[0.6, 0.4])
    transmission = rng.choice(list(TRANSMISSION_MAP.values()), n_rows, p=[0.85, 0.15])
    owner = rng.choice([0, 1, 2, 3], n_rows, p=[0.7, 0.2, 0.07, 0.03])
    car_age = rng.integers(0, 16, n_rows)

    # Depreciation-style formula with noise, roughly matching real-world trends:
    # newer, less-driven, diesel/automatic, dealer-sold, fewer-owner cars hold value better.
    depreciation = 0.92 ** car_age
    km_penalty = 1 - (kms_driven / 300000)
    fuel_bonus = np.where(fuel_type == 1, 1.05, np.where(fuel_type == 0, 1.0, 0.9))
    seller_bonus = np.where(seller_type == 0, 1.05, 0.95)
    transmission_bonus = np.where(transmission == 1, 1.08, 1.0)
    owner_penalty = 1 - (owner * 0.05)

    noise = rng.normal(1.0, 0.05, n_rows)

    selling_price = (
        present_price
        * depreciation
        * km_penalty
        * fuel_bonus
        * seller_bonus
        * transmission_bonus
        * owner_penalty
        * noise
    )
    selling_price = np.clip(selling_price, 0.2, None)

    return pd.DataFrame({
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age,
        "Selling_Price": selling_price,
    })


def _load_and_prepare_real_dataset(path: str) -> pd.DataFrame:
    """Loads the CarDekho CSV and engineers the Car_Age feature the same
    way the model expects (current year minus manufacture year)."""
    df = pd.read_csv(path)

    current_year = pd.Timestamp.now().year
    df["Car_Age"] = current_year - df["Year"]

    df["Fuel_Type"] = df["Fuel_Type"].map(FUEL_MAP)
    df["Seller_Type"] = df["Seller_Type"].map(SELLER_MAP)
    df["Transmission"] = df["Transmission"].map(TRANSMISSION_MAP)

    df = df.dropna(subset=FEATURE_COLUMNS + ["Selling_Price"])
    return df


def train_and_save_model() -> None:
    if os.path.exists(DATA_PATH):
        print(f"Found real dataset at {DATA_PATH}, using it for training.")
        df = _load_and_prepare_real_dataset(DATA_PATH)
    else:
        print("No dataset found at data/car_data.csv — generating a synthetic "
              "dataset with the same schema instead.")
        df = _generate_synthetic_dataset()

    X = df[FEATURE_COLUMNS]
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model trained. MAE: {mae:.3f} Lakhs | R2 Score: {r2:.4f}")

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_and_save_model()

# 🚗 Car Price Prediction System (Dockerized)

An end-to-end Machine Learning web application that predicts the resale price of a used car based on its specifications. The entire system — data pipeline, trained Random Forest Regressor, and Flask backend — is packaged inside a Docker container for consistent, portable deployment.

---

## 👤 Student Details

- **Name:** Deep Talreja
- **Application Number:** IN26010914
- **Registration Number:** 23BCE11003
- **Internship:** MP Online AI/ML Internship

---

## 🔗 Project Links

- **Live Web Application:** https://car-price-predictor-app-docker-by-deep.onrender.com
- **GitHub Repository:** https://github.com/deeptalreja/Car-Price-Predictor-App-Docker-by-Deep-Talreja

---

## 🛠️ Tech Stack

- **Core Language:** Python 3.10
- **Containerization:** Docker
- **ML & Data Processing:** scikit-learn, pandas, NumPy
- **Backend Framework:** Flask
- **Production Server:** Gunicorn (WSGI)
- **Model Serialization:** joblib
- **Cloud Deployment:** Render (Docker runtime)

---

## 📂 Project Structure

```
car-price-predictor/
├── static/
│   └── style.css          # Forest-green UI theme
├── templates/
│   └── index.html         # Frontend prediction form
├── app.py                 # Flask backend & inference endpoint
├── train.py                # Model training pipeline
├── car_price_model.pkl    # Serialized Random Forest Regressor
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container build configuration
├── run.sh / run.bat        # One-command local setup scripts
└── .gitignore
```

---

## 📊 Dataset & Features

The model is designed around the schema of the **Vehicle Dataset from CarDekho** (available on Kaggle). To train on the real dataset:

1. Download `car_data.csv` from [Kaggle: Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
2. Place it at `data/car_data.csv` in this project
3. Run `python train.py` — it will automatically detect and use the real data

If no dataset is found, `train.py` automatically generates a **synthetic dataset** with the same schema and realistic price relationships, so the project runs end-to-end without any manual setup.

| Feature            | Type        | Description                                  |
| ------------------- | ----------- | --------------------------------------------- |
| **Present Price**   | Continuous  | Current showroom price of the car (in Lakhs)  |
| **Kms Driven**      | Discrete    | Total mileage on the odometer                 |
| **Fuel Type**       | Categorical | `Petrol: 0`, `Diesel: 1`, `CNG: 2`             |
| **Seller Type**     | Categorical | `Dealer: 0`, `Individual: 1`                   |
| **Transmission**    | Categorical | `Manual: 0`, `Automatic: 1`                    |
| **Owner**           | Discrete    | Number of previous owners                      |
| **Car Age**         | Continuous  | Years since manufacture                        |

**Target variable:** Selling Price (in Lakhs)

---

## ⚙️ How to Run Locally

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

### Manual Setup

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Navigate to `http://127.0.0.1:5002` in your browser.

### Run with Docker

```bash
docker build -t car-price-docker-app .
docker run -p 10000:10000 car-price-docker-app
```

Navigate to `http://localhost:10000`.

---

## 🚀 API Documentation

### `GET /`
Renders the frontend prediction dashboard.

### `POST /predict`

**Headers:** `Content-Type: application/json`

**Request body:**
```json
{
  "present_price": 8.5,
  "kms_driven": 45000,
  "fuel_type": "Petrol",
  "seller_type": "Dealer",
  "transmission": "Manual",
  "owner": 0,
  "car_age": 4
}
```

**Response (200 OK):**
```json
{
  "predicted_price_lakhs": 6.42
}
```

---

## ☁️ Deployment (Render)

1. Push this repository to GitHub
2. Create a new **Web Service** on [Render](https://render.com), connect your repo
3. Render auto-detects the `Dockerfile` — set environment to Docker, instance type Free
4. Deploy — Render builds the image (training the model at build time) and serves it publicly

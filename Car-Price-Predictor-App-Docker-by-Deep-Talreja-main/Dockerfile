FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model at build time so the image is self-contained.
# Uses data/car_data.csv if present, otherwise falls back to a
# synthetic dataset with the same schema (see train.py).
RUN python train.py

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]

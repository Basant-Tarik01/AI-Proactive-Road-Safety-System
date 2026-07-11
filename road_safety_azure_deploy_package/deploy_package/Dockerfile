FROM python:3.10-slim

WORKDIR /workspace

# Install minimal OS system graphics dependencies required by Headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY best_xgboost_model.json .
COPY yolov8n.pt .

EXPOSE 8000

ENV PYTHONPATH=/workspace

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

import os
import pickle
import logging
import numpy as np
import mlflow
from flask import Flask, render_template, request
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# MLflow & DagsHub setup
# ---------------------------
mlflow.set_tracking_uri("https://dagshub.com/kumarashutoshbtech2023/mlops-credit-card.mlflow")

MODEL_NAME = "my_model"
PREPROCESSOR_PATH = "models/power_transformer.pkl"

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions", ["prediction"], registry=registry)

# ---------------------------
# Load Model & Preprocessor
# ---------------------------
def get_latest_model_version(model_name):
    """Fetch the latest model version from MLflow."""
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.error(f"No versions found for model: {model_name}")
            return None
        latest_version = max(versions, key=lambda v: int(v.version)).version
        return latest_version
    except Exception as e:
        logger.error(f"Error fetching model versions: {e}")
        return None

def load_model(model_name):
    """Load the latest model from MLflow."""
    version = get_latest_model_version(model_name)
    if version is None:
        return None
    model_uri = f"models:/{model_name}/{version}"
    logger.info(f"Loading model from URI: {model_uri}")
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def load_preprocessor(preprocessor_path):
    """Load PowerTransformer from pickle file."""
    if not os.path.exists(preprocessor_path):
        logger.error(f"Preprocessor file not found: {preprocessor_path}")
        return None
    try:
        with open(preprocessor_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load preprocessor: {e}")
        return None

# Load ML components
model = load_model(MODEL_NAME)
power_transformer = load_preprocessor(PREPROCESSOR_PATH)

FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------
# Helper Functions
# ---------------------------
def preprocess_input(data):
    """Preprocess user input before prediction."""
    if power_transformer is None:
        logger.error("Preprocessor not loaded")
        return None
    try:
        arr = np.array(data).reshape(1, -1)
        return power_transformer.transform(arr)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    prediction = None
    input_values = [""] * len(FEATURE_NAMES)

    if request.method == "POST":
        csv_input = request.form.get("csv_input", "").strip()
        if csv_input:
            try:
                values = list(map(float, csv_input.split(",")))
                if len(values) != len(FEATURE_NAMES):
                    raise ValueError(f"Expected {len(FEATURE_NAMES)} values, got {len(values)}")
                input_values = values
                transformed = preprocess_input(input_values)
                if transformed is not None and model is not None:
                    result = model.predict(transformed)
                    prediction = "Fraud" if result[0] == 1 else "Non-Fraud"
                else:
                    prediction = "Error: Model or Preprocessor not loaded"
            except Exception as e:
                prediction = f"Error processing input: {e}"
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return render_template("index.html", result=prediction, csv_input=",".join(map(str, input_values)))

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    csv_input = request.form.get("csv_input", "").strip()
    if not csv_input:
        return "Error: No input provided."

    try:
        values = list(map(float, csv_input.split(",")))
        if len(values) != len(FEATURE_NAMES):
            return f"Error: Expected {len(FEATURE_NAMES)} values, got {len(values)}"
        transformed = preprocess_input(values)
        if transformed is not None and model is not None:
            result = model.predict(transformed)
            return "Fraud" if result[0] == 1 else "Non-Fraud"
            PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
            REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return "Error: Model or Preprocessor not loaded"
    except Exception as e:
        return f"Error processing input: {e}"
@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

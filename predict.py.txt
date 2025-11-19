from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/churn_model.pkl")

app = Flask(__name__)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python train.py first."
        )
    return joblib.load(MODEL_PATH)

model = load_model()


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "message": "Churn prediction API online"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON payload received"}), 400

    # If single item → convert to list
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return jsonify({
        "churn_probability": prob.tolist(),
        "churn_prediction": pred.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)

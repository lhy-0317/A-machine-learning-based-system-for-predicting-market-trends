import json
from pathlib import Path

import joblib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from torch import nn

# 资源路径
project_dir = Path(__file__).parent
model_path = project_dir / "models" / "market_trend_model.pth"
scaler_path = project_dir / "models" / "feature_scaler.pkl"
metadata_path = project_dir / "models" / "model_metadata.json"

# 载入元数据与预处理器
metadata = json.loads(metadata_path.read_text())
feature_columns = metadata["feature_columns"]
label_map = metadata["label_map"]
reverse_label_map = {v: k for k, v in label_map.items()}
input_dim = metadata["input_dim"]

scaler = joblib.load(scaler_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建并加载模型
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, len(label_map)),
).to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

app = Flask(__name__)


default_payload = metadata.get("default_payload")
if default_payload is None:
    default_payload = {
        "open": 96696.5969,
        "high": 100873.3438,
        "low": 96554.4786,
        "close": 100725.3043,
        "volume": 2120035,
        "macro_index": 54.2642,
        "sector_growth": 18.6984,
        "consumer_sentiment": 74.4038,
        "inflation_expectation": 2.2959,
        "search_trend_index": 72.4347,
        "news_sentiment": -0.1717,
        "global_supply_risk": 30.0116,
        "future_7d_return": 0.348,
    }

history_path = project_dir / "models" / "training_history.json"


@app.route("/")
def index():
    training_history = _load_history()
    return render_template(
        "index.html",
        feature_columns=feature_columns,
        default_payload=default_payload,
        has_history=bool(training_history),
    )


@app.route("/status")
def status():
    return jsonify(
        {
            "message": "Market prediction API is running.",
            "endpoints": {"health": "/health", "predict": "/predict", "ui": "/"},
            "port": 5001,
        }
    )


def _load_history():
    if history_path.exists():
        return json.loads(history_path.read_text())
    return []


@app.route("/metrics")
def metrics():
    return jsonify({"history": _load_history()})


def _prepare_payload(payload):
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("JSON body must be an object or a list of objects")
    matrix = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Each item must be an object")
        try:
            values = [float(row[col]) for col in feature_columns]
        except KeyError as err:
            raise ValueError(f"Missing field: {err.args[0]}")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid value: {err}")
        matrix.append(values)
    array = np.asarray(matrix, dtype=np.float32)
    return payload, array


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "features": feature_columns,
            "classes": list(label_map.keys()),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    payload = request.get_json()
    try:
        raw_rows, batch = _prepare_payload(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scaled = scaler.transform(batch)
    tensor = torch.from_numpy(scaled).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)

    outputs = []
    for original, pred_idx, probs in zip(raw_rows, predictions, probabilities):
        outputs.append(
            {
                "input": original,
                "predicted_trend": reverse_label_map[int(pred_idx)],
                "probabilities": {
                    reverse_label_map[i]: float(prob) for i, prob in enumerate(probs)
                },
            }
        )

    if isinstance(payload, dict):
        return jsonify(outputs[0])
    return jsonify(outputs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

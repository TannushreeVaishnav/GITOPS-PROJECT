from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "artifacts/models/xgboost_model.pkl"
SCALER_PATH = "artifacts/processed/scaler.pkl"
FEATURES_PATH = "artifacts/processed/features.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

LABELS = {0: "High", 1: "Low", 2: "Medium"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    table_data = None

    if request.method == "POST":
        # Case 1: File upload
        if "test_file" in request.files and request.files["test_file"].filename != "":
            file = request.files["test_file"]
            df = pd.read_csv(file)

            # Ensure columns match FEATURES (except Hour_sin/cos)
            if "Hour" in df.columns:
                df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
                df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

            # Align with training features
            X = df[[f for f in FEATURES]]
            X_scaled = scaler.transform(X)

            preds = model.predict(X_scaled)
            df["Predicted"] = [LABELS.get(p, "Unknown") for p in preds]

            table_data = df.to_dict(orient="records")

        else:
            # Case 2: Manual entry
            try:
                input_data = []
                for feature in FEATURES:
                    if feature in ["Hour_sin", "Hour_cos"]:
                        continue
                    input_data.append(float(request.form[feature]))

                hour = int(request.form["Hour"])
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)

                input_data.append(hour_sin)
                input_data.append(hour_cos)

                input_array = np.array(input_data).reshape(1, -1)
                scaled_array = scaler.transform(input_array)

                pred = model.predict(scaled_array)[0]
                prediction = LABELS.get(pred, "Unknown")
            except Exception as e:
                prediction = f"Error: {e}"

    display_features = [f for f in FEATURES if f not in ["Hour_sin", "Hour_cos"]]
    display_features.append("Hour")

    return render_template("index.html",
                           prediction=prediction,
                           features=display_features,
                           table_data=table_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

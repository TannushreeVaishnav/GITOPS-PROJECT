import os
import joblib
import logging
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, processed_data_path, model_output_path, metrics_path="artifacts/metrics.json"):
        self.processed_path = processed_data_path
        self.model_path = model_output_path
        self.metrics_path = metrics_path
        self.clf = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.scaler = None
        self.metrics = {}

        logger.info("Model Training Initialized")
        os.makedirs(self.model_path, exist_ok=True)

    def load_data(self):
        try:
            self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = joblib.load(
                os.path.join(self.processed_path, "train_test_split.pkl")
            )
            self.features = joblib.load(os.path.join(self.processed_path, "features.pkl"))
            self.scaler = joblib.load(os.path.join(self.processed_path, "scaler.pkl"))
            logger.info("Processed train/test data loaded successfully.")
        except Exception as e:
            logger.error(f"Error while loading processed data: {e}")
            raise CustomException("Failed to load the data", e)

    def evaluate_and_log(self, model_name, preds):
        acc = accuracy_score(self.y_test, preds)
        f1_weighted = f1_score(self.y_test, preds, average="weighted")
        f1_macro = f1_score(self.y_test, preds, average="macro")

        # Log metrics
        logger.info(f"{model_name} Results:")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Weighted F1: {f1_weighted:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")

        # Save metrics in dictionary for DVC
        self.metrics[model_name] = {
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro
        }

    def train_logistic_regression(self):
        self.clf = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            C=10.0,
            solver='saga'
        )
        self.clf.fit(self.X_train_scaled, self.y_train)
        preds = self.clf.predict(self.X_test_scaled)
        self.evaluate_and_log("Logistic Regression", preds)
        joblib.dump(self.clf, os.path.join(self.model_path, "logistic_regression_model.pkl"))

    def train_random_forest(self):
        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced",
            random_state=42
        )
        self.clf.fit(self.X_train_scaled, self.y_train)
        preds = self.clf.predict(self.X_test_scaled)
        self.evaluate_and_log("Random Forest", preds)
        joblib.dump(self.clf, os.path.join(self.model_path, "random_forest_model.pkl"))

    def train_xgboost(self):
        self.clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.clf.fit(self.X_train_scaled, self.y_train)
        preds = self.clf.predict(self.X_test_scaled)
        self.evaluate_and_log("XGBoost", preds)
        joblib.dump(self.clf, os.path.join(self.model_path, "xgboost_model.pkl"))

    def save_metrics(self):
        # Write metrics to JSON file for DVC
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Metrics saved to {self.metrics_path}")

    def run(self):
        self.load_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.save_metrics()
        logger.info("All models trained, evaluated, and saved successfully.")


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed", "artifacts/models")
    trainer.run()


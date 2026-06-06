import pandas as pd
import yaml
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["model"]["n_estimators"]
random_state = params["model"]["random_state"]

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Start MLflow experiment
mlflow.set_experiment("Loan-Risk-Prediction")

with mlflow.start_run():

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train.values.ravel())

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/loan_model.pkl")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print("Model trained successfully.")
    print("Model saved as models/loan_model.pkl")
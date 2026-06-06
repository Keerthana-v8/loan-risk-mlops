import pandas as pd
import yaml
import joblib
import os

from sklearn.ensemble import RandomForestClassifier

# Load parameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

n_estimators = params["model"]["n_estimators"]
random_state = params["model"]["random_state"]

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Train model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=random_state
)

model.fit(X_train, y_train.values.ravel())

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/loan_model.pkl")

print("Model trained successfully.")
print("Model saved as models/loan_model.pkl")
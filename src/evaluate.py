import pandas as pd
import joblib
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load trained model
model = joblib.load("models/loan_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

# Print results
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save metrics
with open("reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(classification_report(y_test, y_pred))

print("\nMetrics saved to reports/metrics.txt")
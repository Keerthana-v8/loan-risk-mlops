import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load trained model
model = joblib.load("models/loan_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save metrics
with open("reports/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(classification_report(y_test, y_pred))

print("\nMetrics saved to reports/metrics.txt")
import pandas as pd
import os

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load training data
reference_data = pd.read_csv("data/processed/X_train.csv")

# Load testing data
current_data = pd.read_csv("data/processed/X_test.csv")

# Create drift report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Run drift detection
report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Create reports folder if not present
os.makedirs("reports", exist_ok=True)

# Save report
report.save_html("reports/drift_report.html")

print("Drift report generated successfully.")
print("Saved to reports/drift_report.html")
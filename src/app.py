from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import Counter, generate_latest

import joblib
import pandas as pd
import logging
import os

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Prometheus Counter
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API Requests"
)

app = FastAPI()

# Load model
model = joblib.load("models/loan_model.pkl")


@app.get("/")
def home():
    return {"message": "Loan Risk Prediction API Running"}


# Prometheus Metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


@app.post("/predict")
async def predict(
    request: Request,
    Gender: int,
    Married: int,
    Dependents: int,
    Education: int,
    Self_Employed: int,
    ApplicantIncome: float,
    CoapplicantIncome: float,
    LoanAmount: float,
    Loan_Amount_Term: float,
    Credit_History: float,
    Property_Area: int
):

    data = pd.DataFrame(
        [[
            Gender,
            Married,
            Dependents,
            Education,
            Self_Employed,
            ApplicantIncome,
            CoapplicantIncome,
            LoanAmount,
            Loan_Amount_Term,
            Credit_History,
            Property_Area
        ]]
    )

    # Increase Prometheus Counter
    REQUEST_COUNT.inc()

    prediction = model.predict(data)

    # Log request and prediction
    logging.info(
        f"Client={request.client.host}, Prediction={int(prediction[0])}"
    )

    return {
        "prediction": int(prediction[0])
    }
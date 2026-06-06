from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/loan_model.pkl")

@app.get("/")
def home():
    return {"message": "Loan Risk Prediction API Running"}

@app.post("/predict")
def predict(
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

    prediction = model.predict(data)

    return {
        "prediction": int(prediction[0])
    }
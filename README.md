# Loan Risk MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for Loan Risk Prediction using Machine Learning and modern MLOps practices.

The system predicts whether a loan application is likely to be approved based on applicant information.

---

## Technologies Used

- Python
- Scikit-Learn
- DVC
- MLflow
- FastAPI
- Docker
- GitHub Actions
- Prometheus
- Evidently AI
- Trivy
- Git

---

## Project Structure

```text
loan-risk-mlops
│
├── data/
├── models/
├── reports/
├── src/
├── mlruns/
├── .github/
├── .dvc/
│
├── dvc.yaml
├── Dockerfile
├── requirements.txt
├── README.md
```

## Pipeline

Raw Data

↓

Preprocessing

↓

Model Training

↓

Evaluation

↓

Model Storage

↓

FastAPI Deployment

↓

Docker Containerization

↓

GitHub Actions CI/CD

↓

Monitoring & Drift Detection

---

## Model

Algorithm:

```text
Random Forest Classifier
```

Accuracy:

```text
75.6%
```

---

## DVC Pipeline

Run:

```bash
dvc repro
```

Check status:

```bash
dvc status
```

---

## MLflow Tracking

Run:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

---

## FastAPI

Start API:

```bash
uvicorn src.app:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

## Docker

Build:

```bash
docker build -t loan-risk-api .
```

Run:

```bash
docker run -p 8000:8000 loan-risk-api
```

---

## Monitoring

### Prometheus

Metrics endpoint:

```text
/metrics
```

### Evidently AI

Generate drift report:

```bash
python src/drift_detection.py
```

Output:

```text
reports/drift_report.html
```

---

## Security

- Trivy Security Scan
- GitHub Secrets
- .env Exclusion
- Secure Credential Handling

---

## CI/CD

Implemented using GitHub Actions.

Workflow:

```text
.github/workflows/ci.yml
```

---

## Author

Keerthana V

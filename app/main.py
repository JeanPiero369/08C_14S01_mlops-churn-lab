
import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

app = FastAPI(
    title="Churn Predictor API",
    description="API de predicción de abandono de clientes para Fintech",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

model = joblib.load(os.path.join(DATA_DIR, "production_model.pkl"))
scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
feature_names = pd.read_csv(
    os.path.join(DATA_DIR, "feature_names.csv"), header=None
)[0].tolist()

with open(os.path.join(DATA_DIR, "production_model_info.json")) as f:
    model_info = json.load(f)


class CustomerFeatures(BaseModel):
    """Features del cliente para predicción de churn."""
    gender: str = Field(..., example="Male", description="Male o Female")
    senior_citizen: int = Field(..., example=0, description="0 o 1")
    tenure_months: int = Field(..., example=12, description="Meses como cliente")
    contract_type: str = Field(..., example="Month-to-Month", description="Month-to-Month, One Year, Two Year")
    internet_service: Optional[str] = Field(None, example="Fiber", description="Fiber, DSL o null")
    monthly_charges: float = Field(..., example=85.5)
    support_tickets_last_6m: int = Field(..., example=3, description="Tickets de soporte en últimos 6 meses")


class PredictionResponse(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_label: str
    churn_probability: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]
    customer_ids: Optional[List[str]] = None


def preprocess_customer(data: dict) -> pd.DataFrame:
    """Aplica mismo preprocesamiento que en entrenamiento."""
    df = pd.DataFrame([{
        "gender":          1 if data["gender"] == "Male" else 0,
        "SeniorCitizen":   data["senior_citizen"],
        "tenure":          data["tenure_months"],
        "MonthlyCharges":  data["monthly_charges"],
        "support_tickets": data["support_tickets_last_6m"],
        "Contract":        data["contract_type"],
        "InternetService": data["internet_service"] if data["internet_service"] else "No",
    }])

    # One-hot encoding igual que en entrenamiento
    df = pd.get_dummies(df, columns=["Contract", "InternetService"], drop_first=True)

    # Alinear con features de entrenamiento
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    df = df[feature_names]

    # Scaling
    num_features = ["tenure", "MonthlyCharges", "support_tickets"]
    df[num_features] = scaler.transform(df[num_features])

    return df


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "BAJO"
    elif probability < 0.6:
        return "MEDIO"
    else:
        return "ALTO"


def get_recommendation(data: dict, risk: str) -> str:
    if risk == "BAJO":
        return "Sin acción requerida"
    if risk == "MEDIO":
        return "Incluir en campaña de retención"
    # ALTO
    if data.get("contract_type") == "Month-to-Month" and data.get("monthly_charges", 0) > 70:
        return "Ofrecer contrato anual con descuento"
    if data.get("support_tickets_last_6m", 0) >= 5:
        return "Soporte proactivo urgente"
    if data.get("tenure_months", 99) <= 12:
        return "Programa de fidelización temprana"
    return "Contactar con oferta de retención personalizada"


@app.get("/")
def root():
    return {
        "service": "Churn Predictor API",
        "version": model_info.get("version", "1.0"),
        "model_type": model_info.get("model_type", "unknown"),
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info_endpoint():
    return model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, customer_id: Optional[str] = None):
    try:
        data = customer.dict()
        X = preprocess_customer(data)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        risk = get_risk_level(probability)
        return PredictionResponse(
            customer_id=customer_id,
            churn_prediction=prediction,
            churn_label="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(probability, 4),
            risk_level=risk,
            recommendation=get_recommendation(data, risk),
            model_version=model_info.get("version", "1"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    results = []
    for i, customer in enumerate(request.customers):
        data = customer.dict()
        X = preprocess_customer(data)
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        cid = request.customer_ids[i] if request.customer_ids else f"CUST-{i:04d}"
        risk = get_risk_level(probability)
        results.append({
            "customer_id": cid,
            "churn_prediction": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "risk_level": risk,
            "recommendation": get_recommendation(data, risk),
        })
    return {
        "total": len(results),
        "churn_count": sum(r["churn_prediction"] for r in results),
        "predictions": results,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load model and feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Load the label encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

app = FastAPI()

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: CustomerInput):
    try:
        # Convert incoming data to DataFrame
        df = pd.DataFrame([data.dict()])
        print("Raw input:\n", df)

        # Apply label encoders to categorical features
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))

        # Reorder columns to match training
        df = df.reindex(columns=feature_names)

        # Make prediction
        prediction = loaded_model.predict(df)[0]
        prob = loaded_model.predict_proba(df)[0][1]

        return {
            "churn": bool(prediction),
            "probability": float(round(prob, 4)),
            "message": "Customer is likely to churn" if prediction else "Customer is likely to stay"
        }

    except Exception as e:
        print("Prediction error:", str(e))
        return {"error": str(e)}


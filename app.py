import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any
import openai
import os
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = FastAPI(title="ML Churn Prediction API", version="1.0.0")

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")

# Define input data model


class CustomerData(BaseModel):
    """Customer data for churn prediction"""
    # Add your specific features here based on your model
    # Example features (customize based on your actual model):
    age: float
    tenure: float
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 45.0,
                "tenure": 24.0,
                "monthly_charges": 79.85,
                "total_charges": 1800.0,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check"
            }
        }
    )

def load_artifacts():
    """Load only trusted local artifacts matching the declared request schema."""
    root = Path(__file__).resolve().parent
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            loaded_model = joblib.load(root / "logistic_model.pkl")
            loaded_scaler = joblib.load(root / "scaler.pkl")
        expected = list(CustomerData.model_fields)
        if list(getattr(loaded_scaler, "feature_names_in_", [])) != expected:
            return None, None
        if list(getattr(loaded_model, "classes_", [])) != [0, 1]:
            return None, None
        return loaded_model, loaded_scaler
    except Exception:
        # Invalid, missing or incompatible artifacts must not make readiness green.
        return None, None


model, scaler = load_artifacts()


# Response model


class PredictionResponse(BaseModel):
    churn_probability: float
    explanation: str
    input_data: Dict[str, Any]


@app.post('/predict', response_model=PredictionResponse)
def predict(data: CustomerData):
    # Check if models are loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Compatible model artifacts are not available."
        )

    try:
        # Convert Pydantic model to dict for DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        scaled = scaler.transform(df)
        prob = model.predict_proba(scaled)[:, 1][0]

        # Call OpenAI for summarisation
        summary = summarise_prediction(prob, input_dict)

        return PredictionResponse(
            churn_probability=prob,
            explanation=summary,
            input_data=input_dict
        )
    except Exception:
        raise HTTPException(
            status_code=503, detail="Prediction is currently unavailable.") from None


@app.get('/health')
def health_check():
    """Health check endpoint"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Compatible model artifacts are not available.")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "api_version": "1.0.0"
    }


def summarise_prediction(probability, input_data):
    try:
        prompt = f"""
You are an AI assistant. The model predicted a churn probability of {probability:.2f} for this customer with the following features:

{input_data}

Write a short, clear summary explaining this result to a business stakeholder in simple words.
"""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

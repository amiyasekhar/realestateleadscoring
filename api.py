"""
FastAPI REST API for Lead Scoring
Deploy with: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import pickle
import pandas as pd

app = FastAPI(
    title="Lead Scoring API",
    description="ML-powered lead scoring for real estate",
    version="1.0.0"
)

# Load model
with open('lead_scoring_model_xgb.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
label_encoder = artifacts['label_encoder']
feature_names = artifacts['feature_names']


class LeadInput(BaseModel):
    """Lead data schema"""
    age: int = Field(..., ge=18, le=100)
    income: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=300, le=850)
    website_visits: int = Field(..., ge=0)
    email_opens: int = Field(..., ge=0)
    property_views: int = Field(..., ge=0)
    contact_attempts: int = Field(..., ge=0)
    time_on_site_minutes: int = Field(..., ge=0)
    inquiry_submitted: int = Field(..., ge=0, le=1)
    financing_pre_approved: int = Field(..., ge=0, le=1)
    budget_disclosed: int = Field(..., ge=0, le=1)
    location_preference_set: int = Field(..., ge=0, le=1)


class LeadOutput(BaseModel):
    """Output schema"""
    lead_score: str
    confidence: float
    probabilities: Dict[str, float]
    recommendation: str


@app.get("/")
def root():
    return {
        "message": "Lead Scoring API",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=LeadOutput)
def predict(lead: LeadInput):
    """Predict lead score"""
    try:
        lead_dict = lead.dict()
        lead_df = pd.DataFrame([lead_dict])[feature_names]
        
        lead_scaled = scaler.transform(lead_df)
        prediction = model.predict(lead_scaled)[0]
        probabilities = model.predict_proba(lead_scaled)[0]
        
        predicted_class = label_encoder.classes_[prediction]
        prob_dict = {
            label_encoder.classes_[i]: float(probabilities[i])
            for i in range(len(label_encoder.classes_))
        }
        
        recommendations = {
            'High': 'Priority follow-up within 24 hours.',
            'Medium': 'Follow-up within 3-5 days.',
            'Low': 'Add to nurture campaign.'
        }
        
        return LeadOutput(
            lead_score=predicted_class,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            recommendation=recommendations.get(predicted_class, 'Review manually')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
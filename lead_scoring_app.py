"""
Lead Scoring Model - Production Ready
"""

import pickle
import pandas as pd
from typing import Dict, List

class LeadScoringSystem:
    """Production-ready lead scoring system"""
    
    def __init__(self, model_path='lead_scoring_model_xgb.pkl'):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoder = artifacts['label_encoder']
        self.feature_names = artifacts['feature_names']
    
    def predict_single(self, lead_data: Dict) -> Dict:
        """Predict lead score for a single lead"""
        
        # Validate input
        for feature in self.feature_names:
            if feature not in lead_data:
                raise ValueError(f"Missing feature: {feature}")
        
        # Convert to DataFrame
        lead_df = pd.DataFrame([lead_data])[self.feature_names]
        
        # Scale and predict
        lead_scaled = self.scaler.transform(lead_df)
        prediction = self.model.predict(lead_scaled)[0]
        probabilities = self.model.predict_proba(lead_scaled)[0]
        
        # Format results
        predicted_class = self.label_encoder.classes_[prediction]
        prob_dict = {
            self.label_encoder.classes_[i]: float(probabilities[i]) 
            for i in range(len(self.label_encoder.classes_))
        }
        
        return {
            'lead_score': predicted_class,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'recommendation': self._get_recommendation(predicted_class)
        }
    
    def predict_batch(self, leads: List[Dict]) -> List[Dict]:
        """Predict scores for multiple leads"""
        return [self.predict_single(lead) for lead in leads]
    
    def _get_recommendation(self, score: str) -> str:
        """Get action recommendation"""
        recommendations = {
            'High': 'Priority follow-up within 24 hours. Assign to senior sales team.',
            'Medium': 'Follow-up within 3-5 days. Send targeted property recommendations.',
            'Low': 'Add to nurture campaign. Send educational content monthly.'
        }
        return recommendations.get(score, 'Review manually')


# Example usage
if __name__ == "__main__":
    scorer = LeadScoringSystem()
    
    sample_lead = {
        'age': 42,
        'income': 185000,
        'credit_score': 740,
        'website_visits': 18,
        'email_opens': 12,
        'property_views': 14,
        'contact_attempts': 3,
        'time_on_site_minutes': 65,
        'inquiry_submitted': 1,
        'financing_pre_approved': 1,
        'budget_disclosed': 1,
        'location_preference_set': 1
    }
    
    result = scorer.predict_single(sample_lead)
    
    print("\nLead Scoring Result:")
    print(f"Score: {result['lead_score']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Recommendation: {result['recommendation']}")

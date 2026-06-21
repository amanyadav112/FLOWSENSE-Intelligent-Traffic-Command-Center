import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

class FlowSenseEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.is_trained = False
        
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.clf_cause = RandomForestClassifier(n_estimators=100, random_state=42)
        self.reg_duration = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
        self.encoders = {}
        self.cause_medians = {}
        self._prepare_data()

    def _prepare_data(self):
        print("Loading Data and Training Domain-Expert Hybrid AI...")
        df = pd.read_csv(self.data_path)
        
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
        df['duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60
        
        # Outlier removal: Remove forgotten tickets that lasted more than 3 days
        df = df[(df['duration_mins'] > 0) & (df['duration_mins'] <= 4320)]
        
        nlp_df = df.dropna(subset=['description']).copy()
        nlp_df['description'] = nlp_df['description'].astype(str)
        nlp_df['priority'].fillna('Low', inplace=True)
        nlp_df['requires_road_closure'] = nlp_df['requires_road_closure'].astype(int)
        
        # Safe median calculation to prevent NaN errors
        medians = nlp_df.groupby('event_cause')['duration_mins'].median()
        self.cause_medians = medians.fillna(60).to_dict()
        
        # Train NLP strictly for Cause Detection
        X_text = self.tfidf.fit_transform(nlp_df['description'])
        self.clf_cause.fit(X_text, nlp_df['event_cause'])
        
        # Train Duration Regressor
        nlp_df['start_hour'] = nlp_df['start_datetime'].dt.hour
        nlp_df['is_peak_hour'] = nlp_df['start_hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)
        
        for col in ['event_cause', 'priority']:
            le = LabelEncoder()
            nlp_df[col] = le.fit_transform(nlp_df[col])
            self.encoders[col] = le
            
        features = ['event_cause', 'priority', 'requires_road_closure', 'is_peak_hour']
        self.reg_duration.fit(nlp_df[features], nlp_df['duration_mins'])
        
        self.is_trained = True
        print("Training Complete! Server is Ready.")

    def process_incident(self, description, start_hour):
        if not self.is_trained: return {"error": "Model training..."}
        
        desc_lower = description.lower()
        
        # 1. AI NLP Inference for Cause
        desc_vec = self.tfidf.transform([description])
        pred_cause = self.clf_cause.predict(desc_vec)[0]
        
        # Bulletproof Demo Guardrails
        if "water" in desc_lower or "flood" in desc_lower: pred_cause = "water_logging"
        elif "broke" in desc_lower or "stuck" in desc_lower: pred_cause = "vehicle_breakdown"
        elif "crater" in desc_lower or "pothole" in desc_lower: pred_cause = "pot_holes"
        elif "construct" in desc_lower or "digging" in desc_lower: pred_cause = "construction"
        
        # 2. EXPERT LOGIC LAYER (Fixes the identical forecast bug)
        pred_priority = "Low"
        pred_closure = False
        
        if pred_cause == "water_logging":
            pred_priority = "High"
            pred_closure = True
        elif pred_cause == "pot_holes":
            pred_priority = "High"
            pred_closure = False
        elif pred_cause == "construction":
            pred_priority = "Low"
            pred_closure = True
        elif pred_cause == "vehicle_breakdown":
            pred_closure = False
            # Only heavy vehicles trigger high priority breakdowns
            if "bus" in desc_lower or "truck" in desc_lower or "heavy" in desc_lower:
                pred_priority = "High"
            else:
                pred_priority = "Low"

        # 3. Predict Duration
        is_peak_hour = 1 if (8 <= start_hour <= 11) or (17 <= start_hour <= 20) else 0
        
        try: cause_enc = self.encoders['event_cause'].transform([pred_cause])[0]
        except: cause_enc = 0
        try: priority_enc = self.encoders['priority'].transform([pred_priority])[0]
        except: priority_enc = 0
            
        # FIX: Construct a DataFrame with the exact feature names used during training
        features = ['event_cause', 'priority', 'requires_road_closure', 'is_peak_hour']
        input_df = pd.DataFrame(
            [[cause_enc, priority_enc, int(pred_closure), is_peak_hour]], 
            columns=features
        )
        
        pred_duration = self.reg_duration.predict(input_df)[0]
        
        # Statistical Safety Net
        baseline_median = self.cause_medians.get(pred_cause, 60)
        max_allowed = baseline_median * 3
        if pred_duration > max_allowed and pred_cause != 'construction':
            pred_duration = max_allowed

        # 4. Calculate Dynamic Score
        score = 10
        if pred_priority == 'High': score += 35
        if pred_closure: score += 35
        if is_peak_hour: score += 20
        
        # 5. Prescriptive Tactics
        if pred_closure or score >= 75: 
            manpower = "6-10 Officers (Heavy Deployment)"
        elif score >= 45: 
            manpower = "3-5 Officers (Standard Deployment)"
        else: 
            manpower = "1-2 Officers (Light Local Routing)"
            
        heavy_barricade_causes = ['construction', 'water_logging', 'pot_holes', 'procession']
        if pred_closure or pred_cause in heavy_barricade_causes:
            barricades = "Heavy Barricading (20+ units, reflective gear)"
        else: 
            barricades = "Light Cones Only (Hazard marking)"
            
        if pred_closure or score >= 75: 
            div = "Full Upstream Diversion Required. Activate VMS."
        else: 
            div = "Local Lane Bypass. Keep main corridor flowing."

        # Force standard Python integers and booleans for clean JSON serialization
        if hasattr(pred_duration, 'item'):
            final_duration = int(pred_duration.item())
        else:
            final_duration = int(pred_duration)
            
        final_score = int(min(score, 100))
        is_closure = bool(pred_closure)
        is_peak = bool(is_peak_hour)

        return {
            "predicted_inputs": {
                "cause": str(pred_cause).replace('_', ' ').title(),
                "priority": str(pred_priority),
                "closure_required": is_closure
            },
            "impact": {
                "duration_mins": final_duration,
                "score": final_score,
                "is_peak": is_peak
            },
            "prescription": {
                "manpower": str(manpower),
                "barricades": str(barricades),
                "diversion": str(div)
            }
        }
    
flowsense = FlowSenseEngine("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    desc = data.get('description', '')
    
    # BACKEND VALIDATION: Force hour to be between 0 and 23
    raw_hour = int(data.get('hour', 12))
    safe_hour = max(0, min(23, raw_hour)) 
    
    result = flowsense.process_incident(desc, safe_hour)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
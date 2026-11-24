"""
Inference & Risk Simulation Engine for Customer Churn Prediction.
Uses robust native JSON models (XGBoost + preprocessor parameters) to avoid pickle/sklearn version mismatch issues.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import get_project_root


# Preset Personas for interactive demonstration in Streamlit
SAMPLE_PERSONAS = {
    "Select a persona or enter custom details": None,
    "🇩🇪 High-Risk German VIP (Age 52, High Balance, Inactive)": {
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "CreditScore": 640,
        "Tenure": 3,
        "Balance": 145000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 115000.0
    },
    "🇫🇷 Loyal French Youth (Age 24, Moderate Balance, Active)": {
        "Geography": "France",
        "Gender": "Male",
        "Age": 24,
        "CreditScore": 710,
        "Tenure": 2,
        "Balance": 45000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 62000.0
    },
    "🇪🇸 Moderate-Risk Spanish Professional (Age 44, Inactive, Multi-Product)": {
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 44,
        "CreditScore": 590,
        "Tenure": 6,
        "Balance": 88000.0,
        "NumOfProducts": 3,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 92000.0
    },
    "🇫🇷 High-Balance Zero-Product Risk (Age 48, Inactive, Single Product)": {
        "Geography": "France",
        "Gender": "Female",
        "Age": 48,
        "CreditScore": 610,
        "Tenure": 4,
        "Balance": 160000.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 130000.0
    }
}


class ChurnPredictor:
    """Predictor wrapper using native XGBoost JSON and preprocessor parameters."""

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = get_project_root() / "models"
        self.models_dir = Path(models_dir)
        self.xgb_model_path = self.models_dir / "xgb_model.json"
        self.params_path = self.models_dir / "preprocessor_params.json"
        
        self.model = None
        self.params = None
        self._load_resources()

    def _load_resources(self):
        """Load native JSON model and scaling parameters with robust fallbacks."""
        # 1. Load preprocessor params
        if self.params_path.exists():
            with open(self.params_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
        else:
            # Fallback default parameters if file not found
            self.params = {
                'num_cols': [
                    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'EstimatedSalary', 'BalanceSalaryRatio', 'TenureAgeRatio',
                    'CreditScoreAgeRatio', 'ProductsPerTenure'
                ],
                'means': [650.75, 38.95, 5.02, 76381.21, 1.53, 99730.81, 4.11, 0.138, 17.87, 0.366],
                'scales': [96.59, 10.53, 2.90, 62298.16, 0.58, 57331.53, 111.60, 0.089, 5.38, 0.335]
            }

        # 2. Load XGBoost model
        if self.xgb_model_path.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.xgb_model_path))
        else:
            # Attempt joblib pipeline fallback if JSON model not found
            pipeline_path = self.models_dir / "churn_pipeline.joblib"
            if pipeline_path.exists():
                try:
                    import joblib
                    pipe = joblib.load(pipeline_path)
                    # Extract and save native model for future zero-error loads
                    self.model = pipe.named_steps['classifier']
                    self.model.save_model(str(self.xgb_model_path))
                except Exception as e:
                    raise RuntimeError(f"Could not load fallback pipeline: {e}")
            else:
                raise FileNotFoundError(f"Model files not found in {self.models_dir}. Please run training.")

    def transform_features(self, input_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute ratios, standardize numericals, and encode categoricals into feature vector."""
        credit_score = float(input_data.get('CreditScore', 650))
        age = float(input_data.get('Age', 40))
        tenure = float(input_data.get('Tenure', 3))
        balance = float(input_data.get('Balance', 0.0))
        num_products = int(input_data.get('NumOfProducts', 1))
        has_card = int(input_data.get('HasCrCard', 1))
        is_active = int(input_data.get('IsActiveMember', 1))
        salary = float(input_data.get('EstimatedSalary', 50000.0))
        geography = str(input_data.get('Geography', 'France'))
        gender = str(input_data.get('Gender', 'Male'))

        balance_salary_ratio = round(balance / (salary + 1.0), 4)
        tenure_age_ratio = round(tenure / max(age, 1.0), 4)
        credit_score_age_ratio = round(credit_score / max(age, 1.0), 4)
        products_per_tenure = round(num_products / (tenure + 1.0), 4)
        is_zero_balance = 1 if balance == 0 else 0

        raw_features = {
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'EstimatedSalary': salary,
            'BalanceSalaryRatio': balance_salary_ratio,
            'TenureAgeRatio': tenure_age_ratio,
            'CreditScoreAgeRatio': credit_score_age_ratio,
            'ProductsPerTenure': products_per_tenure,
            'Geography': geography,
            'Gender': gender,
            'HasCrCard': has_card,
            'IsActiveMember': is_active,
            'IsZeroBalance': is_zero_balance
        }

        # Scale numerical features
        num_cols = self.params['num_cols']
        means = np.array(self.params['means'])
        scales = np.array(self.params['scales'])
        
        num_vals = np.array([raw_features[c] for c in num_cols], dtype=float)
        num_scaled = (num_vals - means) / scales

        # One-hot encode categorical features (drop='first')
        is_germany = 1.0 if geography == 'Germany' else 0.0
        is_spain = 1.0 if geography == 'Spain' else 0.0
        is_male = 1.0 if gender == 'Male' else 0.0
        has_card_val = float(has_card)
        is_active_val = float(is_active)
        is_zero_val = float(is_zero_balance)

        cat_vals = np.array([is_germany, is_spain, is_male, has_card_val, is_active_val, is_zero_val], dtype=float)

        feature_vector = np.concatenate([num_scaled, cat_vals]).reshape(1, -1)
        return feature_vector, raw_features

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate churn probability, risk tier, diagnostic factors, and retention playbook."""
        feature_vector, raw_features = self.transform_features(input_data)
        
        proba = float(self.model.predict_proba(feature_vector)[0, 1])
        prediction = int(proba >= 0.45)

        if proba < 0.30:
            risk_tier = "Low Risk"
            risk_color = "#10B981"
            badge = "🟢 LOW RISK"
        elif proba < 0.60:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B"
            badge = "🟡 MODERATE RISK"
        else:
            risk_tier = "Critical Risk"
            risk_color = "#EF4444"
            badge = "🔴 CRITICAL RISK"

        # Key risk factors diagnostics
        risk_drivers = []
        if input_data.get('Geography') == 'Germany':
            risk_drivers.append("Regional Market Risk (Germany attrition rate is 2x baseline)")
        if float(input_data.get('Age', 0)) >= 45:
            risk_drivers.append(f"Demographic Vulnerability (Age {int(input_data['Age'])} falls in highest churn segment)")
        if int(input_data.get('IsActiveMember', 1)) == 0:
            risk_drivers.append("Engagement Deficit (Inactive member status strongly correlates with account closure)")
        if int(input_data.get('NumOfProducts', 1)) >= 3:
            risk_drivers.append("Product Friction (Holding 3+ products shows historical attrition spike)")
        elif int(input_data.get('NumOfProducts', 1)) == 1:
            risk_drivers.append("Low Relationship Breadth (Single product account lacks sticky retention hooks)")
        if float(input_data.get('Balance', 0)) >= 100000:
            risk_drivers.append("High Capital Exposure (Significant balance at risk if attrition occurs)")

        # Action Playbook
        actions = []
        if int(input_data.get('IsActiveMember', 1)) == 0:
            actions.append("Initiate dedicated Relationship Manager outreach with high-yield term deposit offer")
        if int(input_data.get('NumOfProducts', 1)) == 1:
            actions.append("Promote cross-sell incentive: Free fee waivers on secondary investment or checking account")
        if input_data.get('Geography') == 'Germany':
            actions.append("Provide customized German branch banking advisory and premium rewards tier")
        if float(input_data.get('Balance', 0)) > 100000:
            actions.append("Assign dedicated Wealth Management specialist to review portfolio allocation")
        if not actions:
            actions.append("Maintain quarterly relationship touchpoint; customer indicates strong loyalty stability")

        return {
            "churn_probability": round(proba, 4),
            "churn_probability_pct": round(proba * 100, 1),
            "predicted_exit": prediction,
            "risk_tier": risk_tier,
            "risk_color": risk_color,
            "badge": badge,
            "risk_drivers": risk_drivers,
            "recommended_actions": actions,
            "engineered_features": raw_features
        }

    def simulate_what_if(self, base_data: Dict[str, Any], variations: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate change in probability when key engagement levers are toggled."""
        base_res = self.predict(base_data)
        modified_data = base_data.copy()
        modified_data.update(variations)
        new_res = self.predict(modified_data)

        delta = round(new_res["churn_probability_pct"] - base_res["churn_probability_pct"], 1)
        return {
            "base_pct": base_res["churn_probability_pct"],
            "new_pct": new_res["churn_probability_pct"],
            "delta_pct": delta,
            "reduction_achieved": delta < 0
        }

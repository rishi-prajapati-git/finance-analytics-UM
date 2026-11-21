"""
Inference & Risk Simulation Engine for Customer Churn Prediction.
Loads serialized pipeline and performs real-time scoring, explainability, and what-if simulation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import get_project_root
from src.model_trainer import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


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
    """Predictor wrapper around trained pipeline."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = get_project_root() / "models" / "churn_pipeline.joblib"
        self.model_path = Path(model_path)
        self.pipeline = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train model first.")
        return joblib.load(self.model_path)

    @staticmethod
    def prepare_features(input_data: Dict[str, Any]) -> pd.DataFrame:
        """Calculate engineered features from customer attributes."""
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

        row = {
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
        return pd.DataFrame([row])

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction, risk tier, risk drivers, and retention playbooks."""
        df_feat = self.prepare_features(input_data)
        proba = float(self.pipeline.predict_proba(df_feat)[0, 1])
        prediction = int(proba >= 0.45)

        if proba < 0.30:
            risk_tier = "Low Risk"
            risk_color = "#10B981"  # Emerald
            badge = "🟢 LOW RISK"
        elif proba < 0.60:
            risk_tier = "Moderate Risk"
            risk_color = "#F59E0B"  # Amber
            badge = "🟡 MODERATE RISK"
        else:
            risk_tier = "Critical Risk"
            risk_color = "#EF4444"  # Crimson
            badge = "🔴 CRITICAL RISK"

        # Identify key risk factors
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

        # Recommended Action Playbook
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
            "engineered_features": df_feat.to_dict(orient='records')[0]
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

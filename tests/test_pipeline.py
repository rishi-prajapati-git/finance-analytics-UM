"""
Unit and integration tests for the Bank Churn Analytics pipeline.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_dataset
from src.features import add_segments, engineer_financial_ratios
from src.inference import ChurnPredictor, SAMPLE_PERSONAS


def test_data_loading():
    """Verify preprocessed dataset loads correctly with expected records."""
    df = load_dataset()
    assert df is not None
    assert len(df) == 10000
    assert 'Exited' in df.columns
    assert 'CreditScore' in df.columns


def test_feature_engineering():
    """Verify segment and ratio columns are properly computed."""
    sample_raw = pd.DataFrame([{
        'CreditScore': 650,
        'Geography': 'Germany',
        'Gender': 'Female',
        'Age': 45,
        'Tenure': 5,
        'Balance': 120000.0,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 0,
        'EstimatedSalary': 80000.0,
        'Exited': 0
    }])
    
    df_seg = add_segments(sample_raw)
    assert 'age_segment' in df_seg.columns
    assert df_seg['age_segment'].iloc[0] == '30-45'

    df_ratios = engineer_financial_ratios(df_seg)
    assert 'BalanceSalaryRatio' in df_ratios.columns
    assert 'ProductsPerTenure' in df_ratios.columns
    assert df_ratios['BalanceSalaryRatio'].iloc[0] == 1.5


def test_inference_and_what_if():
    """Verify predictor can load pipeline and generate valid probabilities."""
    predictor = ChurnPredictor()
    persona = SAMPLE_PERSONAS['🇫🇷 Loyal French Youth (Age 24, Moderate Balance, Active)']
    res = predictor.predict(persona)

    assert 'churn_probability' in res
    assert 0.0 <= res['churn_probability'] <= 1.0
    assert res['risk_tier'] in ['Low Risk', 'Moderate Risk', 'Critical Risk']
    assert isinstance(res['recommended_actions'], list)

    # Test what-if
    sim = predictor.simulate_what_if(persona, {'IsActiveMember': 0})
    assert 'delta_pct' in sim

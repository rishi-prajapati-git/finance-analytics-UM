"""
Feature Engineering module for Bank Customer Churn Analytics.
Constructs domain-specific financial features, behavioral segments, and risk indicators.
"""

import numpy as np
import pandas as pd


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Add business segments based on demographic and financial criteria."""
    df = df.copy()

    # Age Segment
    age_bins = [-np.inf, 29, 45, 60, np.inf]
    age_labels = ['Under 30', '30-45', '46-60', 'Over 60']
    df['age_segment'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    # Credit Band
    credit_bins = [-np.inf, 579, 720, np.inf]
    credit_labels = ['Low', 'Medium', 'High']
    df['credit_band'] = pd.cut(df['CreditScore'], bins=credit_bins, labels=credit_labels)

    # Tenure Group
    tenure_bins = [-np.inf, 2, 6, np.inf]
    tenure_labels = ['New (0-2 yrs)', 'Mid-term (3-6 yrs)', 'Long-term (7+ yrs)']
    df['tenure_group'] = pd.cut(df['Tenure'], bins=tenure_bins, labels=tenure_labels)

    # Balance Segment
    df['balance_segment'] = 'Zero Balance'
    df.loc[(df['Balance'] > 0) & (df['Balance'] < 100000), 'balance_segment'] = 'Low Balance'
    df.loc[df['Balance'] >= 100000, 'balance_segment'] = 'High Balance'
    df['balance_segment'] = pd.Categorical(
        df['balance_segment'],
        categories=['Zero Balance', 'Low Balance', 'High Balance'],
        ordered=True
    )

    # Products Segment
    df['products_segment'] = df['NumOfProducts'].map({
        1: '1 Product',
        2: '2 Products',
        3: '3+ Products',
        4: '3+ Products'
    })

    return df


def engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer advanced banking domain ratios and behavioral interaction features."""
    df = df.copy()

    # 1. Balance to Estimated Salary Ratio
    df['BalanceSalaryRatio'] = (df['Balance'] / (df['EstimatedSalary'] + 1.0)).round(4)

    # 2. Tenure to Age Ratio (relative loyalty span)
    df['TenureAgeRatio'] = (df['Tenure'] / df['Age']).round(4)

    # 3. Credit Score to Age Ratio
    df['CreditScoreAgeRatio'] = (df['CreditScore'] / df['Age']).round(4)

    # 4. Products per Year of Tenure
    df['ProductsPerTenure'] = (df['NumOfProducts'] / (df['Tenure'] + 1.0)).round(4)

    # 5. Is Zero Balance flag
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)

    # 6. High Value Client Flag (top 25% balance)
    hv_threshold = df['Balance'].quantile(0.75)
    df['high_value_flag'] = (df['Balance'] >= hv_threshold).astype(int)

    # 7. Retention Opportunity (High value yet inactive member)
    df['retention_opportunity'] = (
        (df['high_value_flag'] == 1) & (df['IsActiveMember'] == 0)
    ).astype(int)

    # 8. High Churn Risk Flag (Multi-factor heuristic indicator)
    df['high_churn_risk'] = (
        (df['age_segment'] == '46-60') &
        (df['IsActiveMember'] == 0) &
        (df['NumOfProducts'] >= 2)
    ).astype(int)

    # 9. Engagement drop indicator
    df['engagement_drop'] = (
        (df['IsActiveMember'] == 0) & (df['HasCrCard'] == 0)
    ).astype(int)

    # 10. Churn Risk Composite Flag
    df['churn_risk_flag'] = (
        (df['Geography'] == 'Germany') |
        (df['age_segment'] == '46-60') |
        (df['NumOfProducts'] >= 3)
    ).astype(int)

    return df


def build_full_processed_dataset(raw_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """Process raw dataset, drop identifiers, apply segmentation and ratios, and save."""
    df = pd.read_csv(raw_csv_path)

    # Drop identifiers if present
    cols_to_drop = [c for c in ['RowNumber', 'CustomerId', 'Surname'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = add_segments(df)
    df = engineer_financial_ratios(df)

    df.to_csv(output_csv_path, index=False)
    return df

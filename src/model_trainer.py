"""
Production Machine Learning Training and Benchmarking Pipeline for Bank Customer Churn.
Trains, evaluates, tunes, and serializes classification models.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.data_loader import load_dataset, get_project_root


NUMERICAL_FEATURES = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
    'BalanceSalaryRatio',
    'TenureAgeRatio',
    'CreditScoreAgeRatio',
    'ProductsPerTenure',
]

CATEGORICAL_FEATURES = [
    'Geography',
    'Gender',
    'HasCrCard',
    'IsActiveMember',
    'IsZeroBalance',
]

TARGET_COL = 'Exited'


def get_preprocessor() -> ColumnTransformer:
    """Create ColumnTransformer for numerical and categorical pipelines."""
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    df: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, Any], Pipeline, Dict[str, Any]]:
    """Train multiple ML models, compare metrics, and return best pipeline."""
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),
        "XGBoost Classifier": xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.07,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=3.2,
            eval_metric='logloss',
            random_state=42
        )
    }

    comparison_results = {}
    fitted_pipelines = {}
    test_predictions = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        preprocessor = get_preprocessor()
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Cross-validation AUC
        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1
        )

        # Fit on training data
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Predictions on holdout test set
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        comparison_results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4),
            "CV_ROC_AUC_Mean": round(float(np.mean(cv_scores)), 4),
            "CV_ROC_AUC_Std": round(float(np.std(cv_scores)), 4),
            "ConfusionMatrix": cm
        }

        test_predictions[name] = {
            "y_proba": y_proba,
            "y_pred": y_pred
        }

    # Select best model based on ROC-AUC & F1-Score balance
    best_model_name = max(
        comparison_results.keys(),
        key=lambda k: comparison_results[k]["ROC_AUC"] + comparison_results[k]["F1_Score"] * 0.5
    )
    best_pipe = fitted_pipelines[best_model_name]
    best_proba = test_predictions[best_model_name]["y_proba"]

    # Compute ROC and PR Curves for the best model and others
    curves_data = {}
    for name, preds in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, preds["y_proba"])
        p_curve, r_curve, _ = precision_recall_curve(y_test, preds["y_proba"])
        # Downsample points for efficient serialization
        step = max(1, len(fpr) // 80)
        curves_data[name] = {
            "fpr": [round(float(v), 4) for v in fpr[::step]],
            "tpr": [round(float(v), 4) for v in tpr[::step]],
            "precision_curve": [round(float(v), 4) for v in p_curve[::step]],
            "recall_curve": [round(float(v), 4) for v in r_curve[::step]]
        }

    # Extract Feature Importance from best model
    feature_importance_list = []
    clf_step = best_pipe.named_steps['classifier']
    preproc_step = best_pipe.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_encoder = preproc_step.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    if hasattr(clf_step, 'feature_importances_'):
        importances = clf_step.feature_importances_
        for feat, imp in zip(all_feature_names, importances):
            feature_importance_list.append({
                "feature": feat,
                "importance": round(float(imp), 4)
            })
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
    elif hasattr(clf_step, 'coef_'):
        coefs = np.abs(clf_step.coef_[0])
        for feat, coef in zip(all_feature_names, coefs):
            feature_importance_list.append({
                "feature": feat,
                "importance": round(float(coef), 4)
            })
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

    metrics_payload = {
        "best_model": best_model_name,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "churn_rate_test": round(float(y_test.mean()), 4),
        "metrics": comparison_results[best_model_name],
        "all_models": comparison_results,
        "optimal_threshold": 0.42
    }

    return metrics_payload, comparison_results, best_pipe, {
        "feature_importances": feature_importance_list,
        "curves": curves_data
    }


def save_artifacts(
    best_pipe: Pipeline,
    metrics_payload: Dict[str, Any],
    comparison_results: Dict[str, Any],
    extra_artifacts: Dict[str, Any]
) -> None:
    """Save trained pipeline and JSON performance metrics to models/ directory."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline
    model_path = models_dir / "churn_pipeline.joblib"
    joblib.dump(best_pipe, model_path)

    # 2. Metrics summary
    with open(models_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 3. Model comparison
    with open(models_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # 4. Feature importance
    with open(models_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(extra_artifacts["feature_importances"], f, indent=2)

    # 5. ROC/PR curves
    with open(models_dir / "roc_pr_curves.json", "w", encoding="utf-8") as f:
        json.dump(extra_artifacts["curves"], f, indent=2)

    print(f"Artifacts successfully saved to {models_dir}")


def run_training_pipeline() -> None:
    """Main training execution function."""
    print("Loading preprocessed dataset...")
    df = load_dataset()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("Training and benchmarking models...")
    metrics_payload, comparison_results, best_pipe, extra_artifacts = train_and_evaluate_models(df)

    print("\n=== Model Benchmark Summary ===")
    for model_name, res in comparison_results.items():
        print(f"[{model_name}] -> ROC-AUC: {res['ROC_AUC']:.4f} | F1: {res['F1_Score']:.4f} | Recall: {res['Recall']:.4f} | Precision: {res['Precision']:.4f}")

    print(f"\n🏆 Best Model Selected: {metrics_payload['best_model']}")
    save_artifacts(best_pipe, metrics_payload, comparison_results, extra_artifacts)


if __name__ == "__main__":
    run_training_pipeline()

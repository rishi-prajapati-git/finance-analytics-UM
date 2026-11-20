"""
Data loader utility with robust fallback paths for local development and Streamlit Cloud.
"""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_path(filename: str = "churn_final_scored.csv") -> Path:
    """Resolve absolute path to data file."""
    root = get_project_root()
    processed_path = root / "data" / "processed" / filename
    if processed_path.exists():
        return processed_path
    
    # Fallback to raw if processed does not exist
    raw_path = root / "data" / "raw" / "Churn_Modelling.csv"
    if raw_path.exists():
        return raw_path
        
    raise FileNotFoundError(f"Data file {filename} not found in {root / 'data'}")


def load_dataset(filename: str = "churn_final_scored.csv") -> pd.DataFrame:
    """Load dataset with correct dtypes."""
    path = get_data_path(filename)
    return pd.read_csv(path)

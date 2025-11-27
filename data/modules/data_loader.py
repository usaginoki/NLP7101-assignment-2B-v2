"""
Data loading utilities for SemEval 2026 Task 13 Subtask B
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a parquet file and return as DataFrame

    Args:
        file_path: Path to the parquet file

    Returns:
        DataFrame with the loaded data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Loading {path.name}...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all datasets (train, validation, test)

    Returns:
        Dictionary with keys 'train', 'validation', 'test' and DataFrame values
    """
    datasets = {}

    for split in ['train', 'validation', 'test']:
        file_path = f"{split}.parquet"
        try:
            datasets[split] = load_data(file_path)
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, skipping...")
            continue

    return datasets


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Get basic information about a dataset

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with dataset statistics
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }

    # Try to identify label column
    possible_label_cols = ['label', 'target', 'class', 'model', 'family']
    for col in possible_label_cols:
        if col in df.columns:
            info['label_column'] = col
            info['class_distribution'] = df[col].value_counts().to_dict()
            break

    return info

"""
Utility functions for the EDA pipeline
"""

from pathlib import Path
import os


def create_output_dir(dir_path: str) -> Path:
    """
    Create output directory if it doesn't exist

    Args:
        dir_path: Path to directory

    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, filename: str, output_dir: str = "figures"):
    """
    Save a matplotlib or plotly figure

    Args:
        fig: Figure object (matplotlib or plotly)
        filename: Name of the output file
        output_dir: Directory to save the figure
    """
    output_path = create_output_dir(output_dir)
    filepath = output_path / filename

    # Detect figure type and save accordingly
    if hasattr(fig, 'savefig'):  # Matplotlib
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {filepath}")
    elif hasattr(fig, 'write_html'):  # Plotly
        if filename.endswith('.html'):
            fig.write_html(filepath)
        elif filename.endswith('.png') or filename.endswith('.jpg'):
            fig.write_image(filepath)
        else:
            fig.write_html(str(filepath) + '.html')
        print(f"Saved figure to: {filepath}")
    else:
        print(f"Warning: Unknown figure type, cannot save")


def get_code_column(df) -> str:
    """
    Identify the column containing code

    Args:
        df: DataFrame to check

    Returns:
        Name of the code column
    """
    possible_code_cols = ['code', 'text', 'content', 'source', 'snippet']

    for col in possible_code_cols:
        if col in df.columns:
            return col

    # If not found, assume first text column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col

    raise ValueError("Could not identify code column in dataset")


def get_label_column(df) -> str:
    """
    Identify the column containing labels

    Args:
        df: DataFrame to check

    Returns:
        Name of the label column
    """
    possible_label_cols = ['label', 'target', 'class', 'model', 'family']

    for col in possible_label_cols:
        if col in df.columns:
            return col

    raise ValueError("Could not identify label column in dataset")

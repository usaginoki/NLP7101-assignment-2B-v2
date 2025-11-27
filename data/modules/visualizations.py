"""
Visualization utilities for EDA
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional
import numpy as np


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_class_distribution(df: pd.DataFrame, label_col: str, title: str = "Class Distribution"):
    """
    Plot class distribution using plotly

    Args:
        df: DataFrame with data
        label_col: Name of the label column
        title: Plot title

    Returns:
        Plotly figure
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = [label_col, 'count']

    fig = px.bar(
        counts,
        x=label_col,
        y='count',
        title=title,
        labels={label_col: 'AI Model Family', 'count': 'Number of Samples'},
        color='count',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )

    return fig


def plot_feature_boxplots(df: pd.DataFrame, feature: str, label_col: str, title: Optional[str] = None):
    """
    Plot boxplots of a feature across different classes using seaborn

    Args:
        df: DataFrame with data
        feature: Feature column name
        label_col: Label column name
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.boxplot(
        data=df,
        x=label_col,
        y=feature,
        ax=ax,
        palette='Set2'
    )

    ax.set_xlabel('AI Model Family', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.set_title(title or f'{feature} Distribution by Model Family', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None, title: str = "Feature Correlation"):
    """
    Plot correlation heatmap using seaborn

    Args:
        df: DataFrame with features
        features: List of feature columns (if None, use all numeric columns)
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[features].corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title(title, fontsize=16, pad=20)
    plt.tight_layout()

    return fig


def plot_distribution_histogram(df: pd.DataFrame, feature: str, label_col: str, title: Optional[str] = None):
    """
    Plot histogram of a feature using plotly

    Args:
        df: DataFrame with data
        feature: Feature column name
        label_col: Label column name
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = px.histogram(
        df,
        x=feature,
        color=label_col,
        marginal='box',
        title=title or f'{feature} Distribution',
        labels={feature: feature, label_col: 'Model Family'},
        opacity=0.7,
        barmode='overlay'
    )

    fig.update_layout(height=500)

    return fig


def plot_violin(df: pd.DataFrame, feature: str, label_col: str, title: Optional[str] = None):
    """
    Plot violin plot using seaborn

    Args:
        df: DataFrame with data
        feature: Feature column name
        label_col: Label column name
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.violinplot(
        data=df,
        x=label_col,
        y=feature,
        ax=ax,
        palette='muted'
    )

    ax.set_xlabel('AI Model Family', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.set_title(title or f'{feature} Distribution by Model Family', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_scatter(df: pd.DataFrame, x_feature: str, y_feature: str, label_col: str, title: Optional[str] = None):
    """
    Plot scatter plot using plotly

    Args:
        df: DataFrame with data
        x_feature: X-axis feature
        y_feature: Y-axis feature
        label_col: Label column for colors
        title: Plot title

    Returns:
        Plotly figure
    """
    fig = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color=label_col,
        title=title or f'{x_feature} vs {y_feature}',
        labels={x_feature: x_feature, y_feature: y_feature, label_col: 'Model Family'},
        opacity=0.6,
        hover_data=[label_col]
    )

    fig.update_layout(height=600)

    return fig


def plot_pairplot(df: pd.DataFrame, features: List[str], label_col: str):
    """
    Create a pairplot using seaborn

    Args:
        df: DataFrame with data
        features: List of features to plot
        label_col: Label column for hue

    Returns:
        Seaborn PairGrid
    """
    # Sample if dataset is too large
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42)
    else:
        df_sample = df

    g = sns.pairplot(
        df_sample[features + [label_col]],
        hue=label_col,
        palette='Set1',
        diag_kind='kde',
        plot_kws={'alpha': 0.6},
        height=3
    )

    g.fig.suptitle('Feature Pairplot', y=1.02, fontsize=16)

    return g


def plot_parallel_coordinates(df: pd.DataFrame, features: List[str], label_col: str, title: str = "Parallel Coordinates"):
    """
    Create parallel coordinates plot using plotly

    Args:
        df: DataFrame with data
        features: List of features to include
        label_col: Label column
        title: Plot title

    Returns:
        Plotly figure
    """
    # Sample if too large
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df

    fig = px.parallel_coordinates(
        df_sample,
        dimensions=features,
        color=label_col,
        title=title,
        color_continuous_scale=px.colors.diverging.Tealrose
    )

    fig.update_layout(height=600)

    return fig

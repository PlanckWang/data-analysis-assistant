"""Visualization tools for data analysis"""

import io
import base64
import asyncio
from typing import Dict, Any, List, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_json

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

def _create_histogram_sync(df: pd.DataFrame, column: str, bins: int, title: Optional[str]) -> Dict[str, Any]:
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric"}

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = ax.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)

    # Add statistics
    mean_val = df[column].mean()
    median_val = df[column].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    # Labels and title
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title(title or f'Distribution of {column}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert to base64
    image = fig_to_base64(fig)

    return {
        "type": "histogram",
        "column": column,
        "image": image,
        "statistics": {
            "mean": float(mean_val),
            "median": float(median_val),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max())
        }
    }

async def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Create histogram for a numeric column"""
    try:
        return await asyncio.to_thread(_create_histogram_sync, df, column, bins, title)
    except Exception as e:
        return {"error": str(e)}

def _create_scatter_plot_sync(df: pd.DataFrame, x_column: str, y_column: str, color_column: Optional[str], title: Optional[str]) -> Dict[str, Any]:
    # Validate columns
    for col in [x_column, y_column]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found"}
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter
    if color_column and color_column in df.columns:
        # Use color coding
        if pd.api.types.is_numeric_dtype(df[color_column]):
            scatter = ax.scatter(df[x_column], df[y_column], c=df[color_column],
                               cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_column)
        else:
            # Categorical color
            for category in df[color_column].unique():
                mask = df[color_column] == category
                ax.scatter(df[mask][x_column], df[mask][y_column],
                         label=str(category), alpha=0.6, s=50)
            ax.legend()
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.6, s=50)

    # Add trend line
    z = np.polyfit(df[x_column].dropna(), df[y_column].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(df[x_column].sort_values(), p(df[x_column].sort_values()),
            "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')

    # Labels and title
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title or f'{y_column} vs {x_column}')
    ax.grid(True, alpha=0.3)
    if not color_column:
        ax.legend()

    # Calculate correlation
    correlation = df[[x_column, y_column]].corr().iloc[0, 1]

    # Convert to base64
    image = fig_to_base64(fig)

    return {
        "type": "scatter",
        "x_column": x_column,
        "y_column": y_column,
        "color_column": color_column,
        "image": image,
        "correlation": float(correlation)
    }

async def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Create scatter plot for two numeric columns"""
    try:
        return await asyncio.to_thread(_create_scatter_plot_sync, df, x_column, y_column, color_column, title)
    except Exception as e:
        return {"error": str(e)}

def _create_box_plot_sync(df: pd.DataFrame, columns: List[str], title: Optional[str]) -> Dict[str, Any]:
    # Validate columns
    numeric_cols = []
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        return {"error": "No valid numeric columns provided"}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    data_to_plot = [df[col].dropna().values for col in numeric_cols]

    # Create box plot
    box_plot = ax.boxplot(data_to_plot, labels=numeric_cols, patch_artist=True)

    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(numeric_cols)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    # Labels and title
    ax.set_ylabel('Values')
    ax.set_title(title or 'Box Plot Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x labels if many columns
    if len(numeric_cols) > 5:
        plt.xticks(rotation=45, ha='right')

    # Convert to base64
    image = fig_to_base64(fig)

    # Calculate statistics
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "q1": float(df[col].quantile(0.25)),
            "median": float(df[col].median()),
            "q3": float(df[col].quantile(0.75)),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "outliers": int((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) |
                           (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))).sum())
        }

    return {
        "type": "box_plot",
        "columns": numeric_cols,
        "image": image,
        "statistics": stats
    }

async def create_box_plot(
    df: pd.DataFrame,
    columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Create box plot for numeric columns"""
    try:
        return await asyncio.to_thread(_create_box_plot_sync, df, columns, title)
    except Exception as e:
        return {"error": str(e)}

def _create_correlation_heatmap_sync(df: pd.DataFrame, columns: Optional[List[str]], method: str) -> Dict[str, Any]:
    # Select numeric columns
    if columns:
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation heatmap"}

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

    # Title
    ax.set_title(f'Correlation Heatmap ({method.capitalize()} method)')

    # Convert to base64
    image = fig_to_base64(fig)

    return {
        "type": "correlation_heatmap",
        "method": method,
        "columns": numeric_cols,
        "image": image,
        "correlation_matrix": corr_matrix.round(3).to_dict()
    }

async def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Dict[str, Any]:
    """Create correlation heatmap"""
    try:
        return await asyncio.to_thread(_create_correlation_heatmap_sync, df, columns, method)
    except Exception as e:
        return {"error": str(e)}

def _create_time_series_plot_sync(df: pd.DataFrame, date_column: str, value_columns: List[str], title: Optional[str]) -> Dict[str, Any]:
    # Validate date column
    if date_column not in df.columns:
        return {"error": f"Column '{date_column}' not found"}

    # Convert to datetime if needed
    df_copy = df.copy()
    try:
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    except:
        return {"error": f"Cannot convert '{date_column}' to datetime"}

    # Sort by date
    df_copy = df_copy.sort_values(date_column)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each value column
    for col in value_columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            ax.plot(df_copy[date_column], df_copy[col], marker='o',
                   markersize=4, label=col, alpha=0.8)

    # Labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Values')
    ax.set_title(title or 'Time Series Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')

    # Format dates on x-axis
    fig.autofmt_xdate()

    # Convert to base64
    image = fig_to_base64(fig)

    return {
        "type": "time_series",
        "date_column": date_column,
        "value_columns": value_columns,
        "image": image,
        "date_range": {
            "start": str(df_copy[date_column].min()),
            "end": str(df_copy[date_column].max())
        }
    }

async def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Create time series plot"""
    try:
        return await asyncio.to_thread(_create_time_series_plot_sync, df, date_column, value_columns, title)
    except Exception as e:
        return {"error": str(e)}

def _create_interactive_plot_sync(df: pd.DataFrame, plot_type: str, **kwargs) -> Dict[str, Any]:
    if plot_type == "scatter":
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color')
        size = kwargs.get('size')

        fig = px.scatter(df, x=x, y=y, color=color, size=size,
                       hover_data=df.columns)

    elif plot_type == "line":
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color')

        fig = px.line(df, x=x, y=y, color=color)

    elif plot_type == "bar":
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color')

        fig = px.bar(df, x=x, y=y, color=color)

    elif plot_type == "histogram":
        x = kwargs.get('x')
        nbins = kwargs.get('nbins', 30)

        fig = px.histogram(df, x=x, nbins=nbins)

    elif plot_type == "box":
        y = kwargs.get('y')
        x = kwargs.get('x')

        fig = px.box(df, x=x, y=y)

    elif plot_type == "violin":
        y = kwargs.get('y')
        x = kwargs.get('x')

        fig = px.violin(df, x=x, y=y)

    elif plot_type == "heatmap":
        # For correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0
        ))

    else:
        return {"error": f"Unsupported plot type: {plot_type}"}

    # Update layout
    fig.update_layout(
        title=kwargs.get('title', f'{plot_type.capitalize()} Plot'),
        height=600,
        template='plotly_white'
    )

    # Convert to JSON for web display
    plot_json = to_json(fig)

    return {
        "type": f"interactive_{plot_type}",
        "plot_json": plot_json,
        "plot_type": plot_type
    }

async def create_interactive_plot(
    df: pd.DataFrame,
    plot_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Create interactive plots using Plotly"""
    try:
        return await asyncio.to_thread(_create_interactive_plot_sync, df, plot_type, **kwargs)
    except Exception as e:
        return {"error": str(e)}
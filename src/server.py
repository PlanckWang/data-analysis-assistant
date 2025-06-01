"""Data Analysis Assistant MCP Server"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP

# Import visualization functions
from .visualization import (
    create_histogram,
    create_scatter_plot,
    create_box_plot,
    create_correlation_heatmap,
    create_time_series_plot,
    create_interactive_plot
)

# Import statistical testing functions
from .statistical_tests import (
    normality_test,
    t_test,
    anova_test,
    chi_square_test,
    correlation_test
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Data Analysis Assistant")

# Configuration
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
EXPORT_DIR = Path(__file__).parent.parent / "exports"
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

# In-memory storage for current session data
class DataSession:
    def __init__(self):
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.current_df: Optional[str] = None
        self.analysis_history: List[Dict[str, Any]] = []
    
    def add_dataframe(self, name: str, df: pd.DataFrame):
        self.dataframes[name] = df
        self.current_df = name
    
    def get_current_df(self) -> Optional[pd.DataFrame]:
        if self.current_df and self.current_df in self.dataframes:
            return self.dataframes[self.current_df]
        return None
    
    def add_to_history(self, analysis_type: str, result: Any):
        self.analysis_history.append({
            "type": analysis_type,
            "result": result,
            "timestamp": pd.Timestamp.now().isoformat()
        })

# Global session instance
session = DataSession()

@mcp.tool()
async def upload_file(file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a CSV or Excel file for analysis.
    
    Args:
        file_path: Path to the file to upload
        file_name: Optional custom name for the dataset
    
    Returns:
        Information about the uploaded file
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return {"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}
        
        # Read the file
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        # Store in session
        dataset_name = file_name or file_path.stem
        session.add_dataframe(dataset_name, df)
        
        # Basic info
        info = {
            "success": True,
            "dataset_name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        logger.info(f"Successfully uploaded file: {dataset_name}")
        return info
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {"error": str(e)}

@mcp.tool()
async def data_overview() -> Dict[str, Any]:
    """
    Get an overview of the current dataset including shape, types, and missing values.
    
    Returns:
        Comprehensive overview of the dataset
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    try:
        overview = {
            "shape": {
                "rows": len(df),
                "columns": len(df.columns)
            },
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": {
                "by_column": df.isnull().sum().to_dict(),
                "total": df.isnull().sum().sum(),
                "percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict()
            },
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "duplicated_rows": int(df.duplicated().sum())
        }
        
        # Add first few rows as preview
        overview["preview"] = df.head().to_dict(orient='records')
        
        session.add_to_history("data_overview", overview)
        return overview
        
    except Exception as e:
        logger.error(f"Error in data overview: {e}")
        return {"error": str(e)}

@mcp.tool()
async def descriptive_statistics(columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for numeric columns.
    
    Args:
        columns: Optional list of column names. If None, analyzes all numeric columns.
    
    Returns:
        Descriptive statistics including mean, std, min, max, quartiles
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    try:
        if columns:
            # Validate columns exist and are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            valid_cols = [col for col in columns if col in numeric_cols]
            if not valid_cols:
                return {"error": "No valid numeric columns specified"}
            analysis_df = df[valid_cols]
        else:
            # Use all numeric columns
            analysis_df = df.select_dtypes(include=[np.number])
        
        if analysis_df.empty:
            return {"error": "No numeric columns found in the dataset"}
        
        # Calculate statistics
        stats = {
            "basic_stats": analysis_df.describe().to_dict(),
            "additional_stats": {
                "skewness": analysis_df.skew().to_dict(),
                "kurtosis": analysis_df.kurtosis().to_dict(),
                "variance": analysis_df.var().to_dict(),
                "coefficient_of_variation": (analysis_df.std() / analysis_df.mean()).to_dict()
            },
            "columns_analyzed": list(analysis_df.columns)
        }
        
        session.add_to_history("descriptive_statistics", stats)
        return stats
        
    except Exception as e:
        logger.error(f"Error in descriptive statistics: {e}")
        return {"error": str(e)}

@mcp.tool()
async def correlation_analysis(
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        columns: Optional list of column names. If None, uses all numeric columns.
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation matrix and significant correlations
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    try:
        if columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            valid_cols = [col for col in columns if col in numeric_cols]
            if not valid_cols:
                return {"error": "No valid numeric columns specified"}
            analysis_df = df[valid_cols]
        else:
            analysis_df = df.select_dtypes(include=[np.number])
        
        if analysis_df.empty:
            return {"error": "No numeric columns found in the dataset"}
        
        if len(analysis_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = analysis_df.corr(method=method)
        
        # Find significant correlations (excluding diagonal)
        significant_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for significance
                    significant_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(corr_value, 4),
                        "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                    })
        
        result = {
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "method": method,
            "significant_correlations": sorted(
                significant_corr, 
                key=lambda x: abs(x['correlation']), 
                reverse=True
            ),
            "columns_analyzed": list(analysis_df.columns)
        }
        
        session.add_to_history("correlation_analysis", result)
        return result
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        return {"error": str(e)}

@mcp.tool()
async def list_datasets() -> Dict[str, Any]:
    """
    List all currently loaded datasets.
    
    Returns:
        Information about all loaded datasets
    """
    datasets = []
    for name, df in session.dataframes.items():
        datasets.append({
            "name": name,
            "shape": df.shape,
            "columns": len(df.columns),
            "is_current": name == session.current_df
        })
    
    return {
        "datasets": datasets,
        "current_dataset": session.current_df,
        "total_datasets": len(datasets)
    }

@mcp.tool()
async def switch_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Switch to a different loaded dataset.
    
    Args:
        dataset_name: Name of the dataset to switch to
    
    Returns:
        Confirmation of the switch
    """
    if dataset_name not in session.dataframes:
        return {"error": f"Dataset '{dataset_name}' not found"}
    
    session.current_df = dataset_name
    return {
        "success": True,
        "current_dataset": dataset_name,
        "shape": session.dataframes[dataset_name].shape
    }

@mcp.tool()
async def get_column_info(column_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific column.
    
    Args:
        column_name: Name of the column to analyze
    
    Returns:
        Detailed column information
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found in dataset"}
    
    try:
        col = df[column_name]
        info = {
            "name": column_name,
            "dtype": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "null_percentage": round(col.isnull().sum() / len(col) * 100, 2),
            "unique_count": int(col.nunique()),
            "memory_usage_bytes": int(col.memory_usage(deep=True))
        }
        
        # Add type-specific information
        if pd.api.types.is_numeric_dtype(col):
            info.update({
                "type": "numeric",
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "25%": float(col.quantile(0.25)),
                "50%": float(col.quantile(0.50)),
                "75%": float(col.quantile(0.75))
            })
        else:
            # Categorical/object column
            value_counts = col.value_counts()
            info.update({
                "type": "categorical",
                "top_values": value_counts.head(10).to_dict(),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "frequency_of_top": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting column info: {e}")
        return {"error": str(e)}

@mcp.tool()
async def plot_histogram(
    column: str,
    bins: int = 30,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a histogram for a numeric column.
    
    Args:
        column: Name of the column to plot
        bins: Number of bins for the histogram
        title: Optional title for the plot
    
    Returns:
        Base64 encoded image and statistics
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_histogram(df, column, bins, title)
    if "error" not in result:
        session.add_to_history("histogram", result)
    return result

@mcp.tool()
async def plot_scatter(
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a scatter plot for two numeric columns.
    
    Args:
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column for color coding
        title: Optional title for the plot
    
    Returns:
        Base64 encoded image and correlation
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_scatter_plot(df, x_column, y_column, color_column, title)
    if "error" not in result:
        session.add_to_history("scatter_plot", result)
    return result

@mcp.tool()
async def plot_box(
    columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create box plots for numeric columns.
    
    Args:
        columns: List of column names to include
        title: Optional title for the plot
    
    Returns:
        Base64 encoded image and statistics
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_box_plot(df, columns, title)
    if "error" not in result:
        session.add_to_history("box_plot", result)
    return result

@mcp.tool()
async def plot_correlation_heatmap(
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Create a correlation heatmap.
    
    Args:
        columns: Optional list of columns. If None, uses all numeric columns.
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Base64 encoded heatmap image
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_correlation_heatmap(df, columns, method)
    if "error" not in result:
        session.add_to_history("correlation_heatmap", result)
    return result

@mcp.tool()
async def plot_time_series(
    date_column: str,
    value_columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a time series plot.
    
    Args:
        date_column: Column containing dates
        value_columns: List of value columns to plot
        title: Optional title for the plot
    
    Returns:
        Base64 encoded image
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_time_series_plot(df, date_column, value_columns, title)
    if "error" not in result:
        session.add_to_history("time_series_plot", result)
    return result

@mcp.tool()
async def create_interactive_chart(
    plot_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create interactive charts using Plotly.
    
    Args:
        plot_type: Type of plot - 'scatter', 'line', 'bar', 'histogram', 'box', 'violin', 'heatmap'
        **kwargs: Additional parameters specific to each plot type
    
    Returns:
        JSON representation of the interactive plot
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await create_interactive_plot(df, plot_type, **kwargs)
    if "error" not in result:
        session.add_to_history("interactive_plot", result)
    return result

@mcp.tool()
async def test_normality(
    columns: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for normality of numeric columns using multiple statistical tests.
    
    Args:
        columns: List of column names to test
        alpha: Significance level (default 0.05)
    
    Returns:
        Normality test results for each column
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await normality_test(df, columns, alpha)
    if "error" not in result:
        session.add_to_history("normality_test", result)
    return result

@mcp.tool()
async def perform_t_test(
    column1: str,
    column2: Optional[str] = None,
    paired: bool = False,
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Perform t-test to compare means.
    
    Args:
        column1: First column (or only column for one-sample test)
        column2: Second column (optional, for two-sample test)
        paired: Whether to perform paired t-test
        alternative: Test direction - 'two-sided', 'less', or 'greater'
    
    Returns:
        T-test results including statistics and interpretation
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await t_test(df, column1, column2, paired, alternative)
    if "error" not in result:
        session.add_to_history("t_test", result)
    return result

@mcp.tool()
async def perform_anova(
    dependent_var: str,
    independent_var: str
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA to test differences between groups.
    
    Args:
        dependent_var: Dependent variable (numeric)
        independent_var: Independent variable (categorical)
    
    Returns:
        ANOVA results including F-statistic and post-hoc tests
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await anova_test(df, dependent_var, independent_var)
    if "error" not in result:
        session.add_to_history("anova_test", result)
    return result

@mcp.tool()
async def perform_chi_square(
    column1: str,
    column2: str
) -> Dict[str, Any]:
    """
    Perform chi-square test of independence for categorical variables.
    
    Args:
        column1: First categorical variable
        column2: Second categorical variable
    
    Returns:
        Chi-square test results and contingency table
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await chi_square_test(df, column1, column2)
    if "error" not in result:
        session.add_to_history("chi_square_test", result)
    return result

@mcp.tool()
async def test_correlation(
    column1: str,
    column2: str,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Test correlation significance between two variables.
    
    Args:
        column1: First variable
        column2: Second variable
        method: Correlation method - 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation coefficient, p-value, and interpretation
    """
    df = session.get_current_df()
    if df is None:
        return {"error": "No dataset loaded. Please upload a file first."}
    
    result = await correlation_test(df, column1, column2, method)
    if "error" not in result:
        session.add_to_history("correlation_test", result)
    return result

# Run the server
def serve():
    """Run the MCP server"""
    import sys
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Data Analysis Assistant MCP Server")
    
    try:
        asyncio.run(stdio_server(mcp))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    serve()
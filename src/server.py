"""
Data Analysis Assistant MCP (Machine Control Protocol) Server.

This module defines the server-side logic for data analysis tools that can be
called remotely via the MCP. It uses an in-memory `DataSession` to store
datasets and manages various data analysis operations like file uploading,
statistical calculations, and plotting.
"""

import asyncio
import os
from pathlib import Path
import structlog # Added
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

# Configure logging - This will be handled by structlog in chatbot_app.py
# logging.basicConfig(level=logging.INFO) # Removed
logger = structlog.get_logger(__name__) # Changed

# Initialize FastMCP server
mcp = FastMCP("Data Analysis Assistant")

from .config import settings

# In-memory storage for current session data
class DataSession:
    """
    Manages datasets and analysis history for a user session.

    This is a simple in-memory store. For persistence or multi-user scenarios,
    this would typically be replaced by a database or a more robust session management solution.
    """
    def __init__(self) -> None:
        """Initializes the DataSession with empty dataframes and history."""
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.current_df: Optional[str] = None
        self.analysis_history: List[Dict[str, Any]] = []
        logger.info("DataSession_initialized_in_memory", message="DataSession is using in-memory storage. All data will be lost on server restart.")
    
    def add_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Adds a Pandas DataFrame to the session.

        Args:
            name: The name to assign to the dataset.
            df: The Pandas DataFrame to add.
        """
        self.dataframes[name] = df
        self.current_df = name # Set the newly added dataframe as current
    
    def get_current_df(self) -> Optional[pd.DataFrame]:
        """
        Retrieves the currently active DataFrame.

        Returns:
            Optional[pd.DataFrame]: The current DataFrame if one is active, otherwise None.
        """
        if self.current_df and self.current_df in self.dataframes:
            return self.dataframes[self.current_df]
        return None
    
    def add_to_history(self, analysis_type: str, result: Any) -> None:
        """
        Adds an analysis step to the session's history.

        Args:
            analysis_type: A string describing the type of analysis performed.
            result: The result of the analysis (can be any type).
        """
        self.analysis_history.append({
            "type": analysis_type,
            "result": result, # Consider summarizing large results if needed
            "timestamp": pd.Timestamp.now().isoformat()
        })

# Global session instance - For single-user context as per current design.
# session: DataSession = DataSession() # Removed global session
user_sessions: Dict[str, DataSession] = {} # New dictionary for user-specific sessions

@mcp.tool()
async def upload_file(session_id: str, file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload a CSV or Excel file for analysis. The file should already be present on the server's filesystem
    at the location specified by `file_path`.

    Args:
        file_path (str): The absolute path to the CSV or Excel file on the server.
        file_name (Optional[str]): An optional custom name for the dataset. If not provided,
                                   the file's stem will be used.

    Returns:
        Dict[str, Any]: A dictionary containing information about the uploaded file,
                        including its name, shape, columns, data types, and memory usage.
                        Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    try:
        file_path_obj: Path = Path(file_path) # Create Path object for easier manipulation
        
        if not file_path_obj.exists():
            logger.warn("upload_file_not_found", file_path=str(file_path_obj), session_id=session_id)
            return {"error": f"文件未找到: {str(file_path_obj)}"}
        
        file_suffix: str = file_path_obj.suffix.lower()
        if file_suffix not in settings.ALLOWED_EXTENSIONS:
            logger.warn("upload_file_type_not_allowed", file_path=str(file_path_obj), suffix=file_suffix, allowed_extensions=settings.ALLOWED_EXTENSIONS, session_id=session_id)
            return {"error": f"不支持的文件类型 '{file_suffix}'. 允许的类型: {settings.ALLOWED_EXTENSIONS}"}
        
        # Read the file
        df: pd.DataFrame
        if file_suffix == '.csv':
            df = pd.read_csv(file_path_obj)
        else: # Handles .xlsx, .xls
            df = pd.read_excel(file_path_obj)
        
        # Store in session
        dataset_name: str = file_name or file_path_obj.stem
        session.add_dataframe(dataset_name, df)
        
        # Basic info
        info: Dict[str, Any] = {
            "success": True, # Indicates successful execution of the tool's main logic
            "dataset_name": dataset_name,
            "shape": df.shape, # Tuple (rows, columns)
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}, # Convert dtypes to string
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),  # MB
        }
        
        logger.info("file_uploaded_successfully", dataset_name=dataset_name, file_path=str(file_path_obj), resulting_shape=df.shape, session_id=session_id)
        return info
        
    except Exception as e:
        logger.error("file_upload_tool_error", provided_file_path=file_path, provided_file_name=file_name, error=str(e), exc_info=True, session_id=session_id)
        return {"error": str(e)}

@mcp.tool()
async def data_overview(session_id: str) -> Dict[str, Any]:
    """
    Get an overview of the current dataset.
    
    This includes information about the dataset's shape (rows, columns),
    column names, data types of each column, missing value counts and percentages,
    lists of numeric and categorical columns, memory usage, count of duplicated rows,
    and a preview of the first few rows.

    Returns:
        Dict[str, Any]: A dictionary containing the dataset overview.
                        Returns `{"error": "message"}` if no dataset is loaded or an error occurs.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    if df is None:
        logger.warn("data_overview_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}
    
    current_dataset_name: Optional[str] = session.current_df
    try:
        numeric_columns: List[str] = list(df.select_dtypes(include=[np.number]).columns)
        categorical_columns: List[str] = list(df.select_dtypes(include=['object', 'category']).columns)

        overview: Dict[str, Any] = {
            "dataset_name": current_dataset_name,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": {
                "by_column": {col: int(val) for col, val in df.isnull().sum().to_dict().items()}, # Ensure int for JSON
                "total": int(df.isnull().sum().sum()),
                "percentage": {col: round(val, 2) for col, val in (df.isnull().sum() / len(df) * 100).to_dict().items()}
            },
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "duplicated_rows": int(df.duplicated().sum()),
            "preview": df.head().to_dict(orient='records') # First 5 rows
        }
        
        session.add_to_history("data_overview", {"dataset_name": current_dataset_name, "shape": overview["shape"]}) # Log summary
        logger.info("data_overview_success", dataset_name=current_dataset_name, session_id=session_id)
        return overview
        
    except Exception as e:
        logger.error("data_overview_tool_error", dataset_name=current_dataset_name, error=str(e), exc_info=True, session_id=session_id)
        return {"error": str(e)}

@mcp.tool()
async def descriptive_statistics(session_id: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for specified numeric columns, or all numeric columns if none are specified.
    
    Statistics include count, mean, std, min, max, and quartiles, as well as
    skewness, kurtosis, variance, and coefficient of variation.

    Args:
        columns (Optional[List[str]]): A list of numeric column names to analyze.
                                       If None or empty, all numeric columns are used.
    
    Returns:
        Dict[str, Any]: A dictionary containing descriptive statistics.
                        Returns `{"error": "message"}` if no suitable data or an error occurs.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    if df is None:
        logger.warn("descriptive_statistics_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    current_dataset_name: Optional[str] = session.current_df
    analysis_df: pd.DataFrame
    
    try:
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warn("descriptive_statistics_no_numeric_columns", dataset_name=current_dataset_name, session_id=session_id)
            return {"error": "数据集中未找到数值列。"}

        if columns:
            valid_cols: List[str] = [col for col in columns if col in numeric_df.columns]
            if not valid_cols:
                logger.warn("descriptive_statistics_no_valid_specified_columns", dataset_name=current_dataset_name, requested_columns=columns, session_id=session_id)
                return {"error": "未指定有效的数值列，或指定的列不是数值列。"}
            analysis_df = numeric_df[valid_cols]
        else:
            analysis_df = numeric_df # Use all numeric columns
        
        if analysis_df.empty: # Should be caught by numeric_df.empty, but as a safeguard
             logger.warn("descriptive_statistics_empty_analysis_df", dataset_name=current_dataset_name, session_id=session_id)
             return {"error": "列选择后没有可供分析的数据。"}

        desc_stats: pd.DataFrame = analysis_df.describe()
        
        # Ensure all potentially NaN/Inf values are converted for JSON
        basic_stats_dict: Dict[str, Dict[str, Optional[float]]] = {
            col: {idx: (None if pd.isna(val) else float(val)) for idx, val in series.items()}
            for col, series in desc_stats.items()
        }

        additional_stats_dict: Dict[str, Dict[str, Optional[float]]] = {
            "skewness": {col: (None if pd.isna(val) else float(val)) for col, val in analysis_df.skew().items()},
            "kurtosis": {col: (None if pd.isna(val) else float(val)) for col, val in analysis_df.kurtosis().items()},
            "variance": {col: (None if pd.isna(val) else float(val)) for col, val in analysis_df.var().items()},
            "coefficient_of_variation": { # Handle potential division by zero if mean is 0
                col: (None if pd.isna(val) or (analysis_df.mean().get(col, 0) == 0 and val != 0) else float(val))
                for col, val in (analysis_df.std() / analysis_df.mean(skipna=True)).items() # skipna for mean
            }
        }
        
        stats_result: Dict[str, Any] = {
            "dataset_name": current_dataset_name,
            "columns_analyzed": list(analysis_df.columns),
            "basic_stats": basic_stats_dict,
            "additional_stats": additional_stats_dict,
        }

        session.add_to_history("descriptive_statistics", {"columns_analyzed": list(analysis_df.columns)})
        logger.info("descriptive_statistics_success", dataset_name=current_dataset_name, columns_analyzed=list(analysis_df.columns), session_id=session_id)
        return stats_result
        
    except Exception as e:
        logger.error("descriptive_statistics_tool_error", dataset_name=current_dataset_name, requested_columns=columns, error=str(e), exc_info=True, session_id=session_id)
        return {"error": str(e)}

@mcp.tool()
async def correlation_analysis(
    session_id: str,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Calculate the correlation matrix for specified or all numeric columns.
    
    Supported methods are 'pearson', 'spearman', or 'kendall'.
    Also identifies and returns pairs of variables with a correlation coefficient
    absolute value greater than 0.5.

    Args:
        columns (Optional[List[str]]): List of numeric column names. If None, all numeric columns are used.
        method (str): Correlation method to use. Defaults to "pearson".
    
    Returns:
        Dict[str, Any]: Contains the correlation matrix, method used, significant correlations,
                        and columns analyzed. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    if df is None:
        logger.warn("correlation_analysis_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    current_dataset_name: Optional[str] = session.current_df
    analysis_df: pd.DataFrame

    try:
        numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            logger.warn("correlation_analysis_no_numeric_columns", dataset_name=current_dataset_name, session_id=session_id)
            return {"error": "数据集中未找到数值列。"}

        if columns:
            valid_cols: List[str] = [col for col in columns if col in numeric_df.columns]
            if not valid_cols:
                logger.warn("correlation_analysis_no_valid_specified_columns", dataset_name=current_dataset_name, requested_columns=columns, session_id=session_id)
                return {"error": "未指定有效的数值列，或指定的列不是数值列。"}
            analysis_df = numeric_df[valid_cols]
        else:
            analysis_df = numeric_df
        
        if analysis_df.empty or len(analysis_df.columns) < 2:
            logger.warn("correlation_analysis_not_enough_columns", dataset_name=current_dataset_name, num_columns=len(analysis_df.columns), session_id=session_id)
            return {"error": "相关性分析至少需要两个数值列。"}
        
        corr_matrix: pd.DataFrame = analysis_df.corr(method=method)
        
        significant_correlations: List[Dict[str, Any]] = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)): # Ensure j > i to avoid duplicates and self-correlation
                corr_value: float = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Threshold for significance
                    significant_correlations.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": round(corr_value, 4),
                        "strength": "strong" if abs(corr_value) > 0.7 else "moderate"
                    })
        
        # Sort by absolute correlation value in descending order
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        result: Dict[str, Any] = {
            "dataset_name": current_dataset_name,
            "columns_analyzed": list(analysis_df.columns),
            "method": method,
            # Convert DataFrame to dict for JSON serialization: {column -> {index -> value}}
            "correlation_matrix": {col: series.round(4).to_dict() for col, series in corr_matrix.items()},
            "significant_correlations": significant_correlations
        }
        
        session.add_to_history("correlation_analysis", {"method": method, "columns_analyzed": list(analysis_df.columns)})
        logger.info("correlation_analysis_success", dataset_name=current_dataset_name, method=method, columns_analyzed=list(analysis_df.columns), session_id=session_id)
        return result
        
    except Exception as e:
        logger.error("correlation_analysis_tool_error", dataset_name=current_dataset_name, requested_columns=columns, method=method, error=str(e), exc_info=True, session_id=session_id)
        return {"error": str(e)}

@mcp.tool()
async def list_datasets(session_id: str) -> Dict[str, Any]:
    """
    List all currently loaded datasets in the session.

    Returns:
        Dict[str, Any]: A dictionary containing a list of dataset information
                        (name, shape, column count, if it's the current one),
                        the name of the current dataset, and the total count.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    datasets_info: List[Dict[str, Any]] = []
    for name, df_item in session.dataframes.items(): # df_item is used to avoid conflict with outer df
        datasets_info.append({
            "name": name,
            "shape": df_item.shape,
            "columns": len(df_item.columns),
            "is_current": name == session.current_df
        })
    
    logger.info("list_datasets_success", count=len(datasets_info), current_dataset=session.current_df, session_id=session_id)
    return {
        "datasets": datasets_info,
        "current_dataset": session.current_df,
        "total_datasets": len(datasets_info)
    }

@mcp.tool()
async def switch_dataset(session_id: str, dataset_name: str) -> Dict[str, Any]:
    """
    Switch the currently active dataset in the session.

    Args:
        dataset_name (str): The name of the dataset to switch to.
                            This name must exist in the loaded datasets.
    
    Returns:
        Dict[str, Any]: Confirmation of the switch, including the new current dataset's name and shape.
                        Returns `{"error": "message"}` if the dataset is not found.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    if dataset_name not in session.dataframes:
        logger.warn("switch_dataset_not_found", requested_dataset=dataset_name, available_datasets=list(session.dataframes.keys()), session_id=session_id)
        return {"error": f"数据集 '{dataset_name}' 未找到。可用数据集: {', '.join(session.dataframes.keys())}"}
    
    session.current_df = dataset_name
    switched_df_shape: tuple = session.dataframes[dataset_name].shape # Added type hint
    logger.info("switch_dataset_success", new_current_dataset=dataset_name, shape=switched_df_shape, session_id=session_id)
    return {
        "success": True,
        "current_dataset": dataset_name,
        "shape": switched_df_shape
    }

@mcp.tool()
async def get_column_info(session_id: str, column_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific column in the current dataset.

    This includes data type, count of non-null values, null count and percentage,
    number of unique values, and memory usage. For numeric columns, it also includes
    mean, std, min, max, and quartiles. For categorical/object columns, it includes
    top unique values and their frequencies.

    Args:
        column_name (str): The name of the column to analyze.
    
    Returns:
        Dict[str, Any]: A dictionary containing detailed information about the column.
                        Returns `{"error": "message"}` if no dataset/column or an error occurs.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df

    if df is None:
        logger.warn("get_column_info_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}
    
    if column_name not in df.columns:
        logger.warn("get_column_info_column_not_found", dataset_name=current_dataset_name, column_name=column_name, available_columns=list(df.columns), session_id=session_id)
        return {"error": f"列 '{column_name}' 在数据集 '{current_dataset_name}' 中未找到。可用列: {', '.join(df.columns)}"}
    
    try:
        col: pd.Series = df[column_name]
        info: Dict[str, Any] = {
            "dataset_name": current_dataset_name,
            "column_name": column_name,
            "dtype": str(col.dtype),
            "non_null_count": int(col.count()),
            "null_count": int(col.isnull().sum()),
            "null_percentage": round(col.isnull().sum() / len(col) * 100, 2) if len(col) > 0 else 0.0,
            "unique_count": int(col.nunique()),
            "memory_usage_bytes": int(col.memory_usage(deep=True))
        }
        
        if pd.api.types.is_numeric_dtype(col):
            # Check if series is empty or all NaNs before calculating stats
            is_col_valid_for_numeric_stats = not col.empty and col.notna().any()
            info.update({
                "column_type": "numeric",
                "mean": float(col.mean()) if is_col_valid_for_numeric_stats else None,
                "std_dev": float(col.std()) if is_col_valid_for_numeric_stats else None,
                "min_value": float(col.min()) if is_col_valid_for_numeric_stats else None,
                "max_value": float(col.max()) if is_col_valid_for_numeric_stats else None,
                "q1_25_percentile": float(col.quantile(0.25)) if is_col_valid_for_numeric_stats else None,
                "median_50_percentile": float(col.quantile(0.50)) if is_col_valid_for_numeric_stats else None,
                "q3_75_percentile": float(col.quantile(0.75)) if is_col_valid_for_numeric_stats else None,
            })
        else: # Categorical/object or other types
            value_counts: pd.Series = col.value_counts()
            info.update({
                "column_type": "categorical/object", # More generic term
                "top_values": {str(k): int(v) for k, v in value_counts.head(10).items()}, # Ensure keys are strings for JSON
                "most_frequent_value": str(value_counts.index[0]) if not value_counts.empty else None,
                "frequency_of_most_frequent": int(value_counts.iloc[0]) if not value_counts.empty else 0
            })
        
        logger.info("get_column_info_success", dataset_name=current_dataset_name, column_name=column_name, session_id=session_id)
        return info
        
    except Exception as e:
        logger.error("get_column_info_tool_error", dataset_name=current_dataset_name, column_name=column_name, error=str(e), exc_info=True, session_id=session_id)
        return {"error": str(e)}

@mcp.tool()
async def plot_histogram(
    session_id: str,
    column: str,
    bins: int = 30,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a histogram for a specified numeric column in the current dataset.

    Args:
        column (str): The name of the numeric column to plot.
        bins (int): The number of bins for the histogram. Defaults to 30.
        title (Optional[str]): An optional title for the plot. If None, a default title is generated.

    Returns:
        Dict[str, Any]: A dictionary containing the plot as a Base64 encoded image string
                        and potentially some statistics. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("plot_histogram_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}
    
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        logger.warn("plot_histogram_invalid_column", dataset_name=current_dataset_name, column_name=column, session_id=session_id)
        return {"error": f"列 '{column}' 在数据集 '{current_dataset_name}' 中不是一个有效的数值列。"}

    # Assuming create_histogram is an async function that returns a Dict
    result: Dict[str, Any] = await create_histogram(df, column, bins, title)

    if "error" not in result:
        session.add_to_history("histogram", {"column": column, "bins": bins, "title": title})
        logger.info("plot_histogram_success", dataset_name=current_dataset_name, column=column, bins=bins, session_id=session_id)
    else:
        logger.error("plot_histogram_tool_error", dataset_name=current_dataset_name, column=column, error=result.get("error"), exc_info=False, session_id=session_id) # No exc_info as error is from called func
    return result

@mcp.tool()
async def plot_scatter(
    session_id: str,
    x_column: str,
    y_column: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a scatter plot for two specified numeric columns in the current dataset.
    Optionally, a third column can be used for color-coding points.

    Args:
        x_column (str): The name of the column for the x-axis. Must be numeric.
        y_column (str): The name of the column for the y-axis. Must be numeric.
        color_column (Optional[str]): An optional column name to use for color-coding the points.
                                     Can be numeric or categorical.
        title (Optional[str]): An optional title for the plot.

    Returns:
        Dict[str, Any]: A dictionary containing the plot as a Base64 encoded image string
                        and potentially correlation statistics. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("plot_scatter_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    # Validate columns
    for col_name in [x_column, y_column]:
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            logger.warn("plot_scatter_invalid_column", dataset_name=current_dataset_name, column_name=col_name, type="axis", session_id=session_id)
            return {"error": f"用于坐标轴的列 '{col_name}' 在数据集 '{current_dataset_name}' 中不是一个有效的数值列。"}
    if color_column and color_column not in df.columns:
        logger.warn("plot_scatter_invalid_column", dataset_name=current_dataset_name, column_name=color_column, type="color", session_id=session_id)
        return {"error": f"颜色列 '{color_column}' 在数据集 '{current_dataset_name}' 中未找到。"}

    result: Dict[str, Any] = await create_scatter_plot(df, x_column, y_column, color_column, title)
    
    if "error" not in result:
        session.add_to_history("scatter_plot", {"x_column": x_column, "y_column": y_column, "color_column": color_column, "title": title})
        logger.info("plot_scatter_success", dataset_name=current_dataset_name, x_column=x_column, y_column=y_column, color_column=color_column, session_id=session_id)
    else:
        logger.error("plot_scatter_tool_error", dataset_name=current_dataset_name, x_column=x_column, y_column=y_column, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def plot_box(
    session_id: str,
    columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create box plots for a list of specified numeric columns in the current dataset.

    Args:
        columns (List[str]): A list of numeric column names to include in the box plot.
        title (Optional[str]): An optional title for the plot.

    Returns:
        Dict[str, Any]: A dictionary containing the plot as a Base64 encoded image string
                        and potentially summary statistics. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("plot_box_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    valid_columns: List[str] = []
    if not columns: # Check if columns list is empty
        logger.warn("plot_box_no_columns_specified", dataset_name=current_dataset_name, session_id=session_id)
        return {"error": "未指定用于箱线图的列。"}

    for col_name in columns:
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            logger.warn("plot_box_invalid_column", dataset_name=current_dataset_name, column_name=col_name, all_columns=columns, session_id=session_id)
            return {"error": f"列 '{col_name}' 在数据集 '{current_dataset_name}' 中不是一个有效的数值列。请仅提供数值列。"}
        valid_columns.append(col_name)

    if not valid_columns: # Should be caught by previous checks, but as safeguard
        logger.warn("plot_box_no_valid_columns_found", dataset_name=current_dataset_name, requested_columns=columns, session_id=session_id)
        return {"error": "未找到用于箱线图的有效数值列。"}

    result: Dict[str, Any] = await create_box_plot(df, valid_columns, title)
    
    if "error" not in result:
        session.add_to_history("box_plot", {"columns": valid_columns, "title": title})
        logger.info("plot_box_success", dataset_name=current_dataset_name, columns=valid_columns, session_id=session_id)
    else:
        logger.error("plot_box_tool_error", dataset_name=current_dataset_name, columns=columns, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def plot_correlation_heatmap(
    session_id: str,
    columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Create a correlation heatmap for specified or all numeric columns in the current dataset.

    Args:
        columns (Optional[List[str]]): A list of numeric column names. If None, all numeric columns are used.
        method (str): The correlation method ('pearson', 'spearman', 'kendall'). Defaults to 'pearson'.

    Returns:
        Dict[str, Any]: A dictionary containing the plot as a Base64 encoded image string.
                        Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("plot_correlation_heatmap_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    analysis_df: pd.DataFrame
    numeric_df: pd.DataFrame = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        logger.warn("plot_correlation_heatmap_no_numeric_cols", dataset_name=current_dataset_name, session_id=session_id)
        return {"error": "没有数值列可用于创建热力图。"}

    if columns:
        valid_cols: List[str] = [col for col in columns if col in numeric_df.columns]
        if not valid_cols or len(valid_cols) < 2:
            logger.warn("plot_correlation_heatmap_not_enough_valid_cols", dataset_name=current_dataset_name, requested_columns=columns, session_id=session_id)
            return {"error": "为热力图指定的有效数值列不足（至少需要2列）。"}
        analysis_df = numeric_df[valid_cols]
    else:
        if len(numeric_df.columns) < 2:
            logger.warn("plot_correlation_heatmap_not_enough_numeric_cols_in_dataset", dataset_name=current_dataset_name, num_numeric_cols=len(numeric_df.columns), session_id=session_id)
            return {"error": "数据集中没有足够的数值列来生成热力图（至少需要2列）。"}
        analysis_df = numeric_df

    result: Dict[str, Any] = await create_correlation_heatmap(analysis_df, columns, method) # Pass analysis_df
    
    if "error" not in result:
        session.add_to_history("correlation_heatmap", {"columns": list(analysis_df.columns), "method": method})
        logger.info("plot_correlation_heatmap_success", dataset_name=current_dataset_name, columns=list(analysis_df.columns), method=method, session_id=session_id)
    else:
        logger.error("plot_correlation_heatmap_tool_error", dataset_name=current_dataset_name, columns=columns, method=method, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def plot_time_series(
    session_id: str,
    date_column: str,
    value_columns: List[str],
    title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a time series plot for one or more value columns against a date column.

    Args:
        date_column (str): The name of the column containing date/datetime information.
                           This column will be converted to datetime objects.
        value_columns (List[str]): A list of column names for the values to plot on the y-axis.
                                   These columns must be numeric.
        title (Optional[str]): An optional title for the plot.

    Returns:
        Dict[str, Any]: A dictionary containing the plot as a Base64 encoded image string.
                        Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("plot_time_series_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    if not date_column or date_column not in df.columns:
        logger.warn("plot_time_series_invalid_date_column", dataset_name=current_dataset_name, date_column=date_column, session_id=session_id)
        return {"error": f"日期列 '{date_column}' 在数据集 '{current_dataset_name}' 中未找到。"}

    if not value_columns:
        logger.warn("plot_time_series_no_value_columns", dataset_name=current_dataset_name, session_id=session_id)
        return {"error": "未指定用于时间序列图的数值列。"}

    valid_value_columns: List[str] = []
    for vc_name in value_columns:
        if vc_name not in df.columns or not pd.api.types.is_numeric_dtype(df[vc_name]):
            logger.warn("plot_time_series_invalid_value_column", dataset_name=current_dataset_name, value_column=vc_name, session_id=session_id)
            return {"error": f"数值列 '{vc_name}' 在数据集 '{current_dataset_name}' 中不是一个有效的数值列。"}
        valid_value_columns.append(vc_name)

    if not valid_value_columns: # Should be caught by previous checks
        logger.warn("plot_time_series_no_valid_value_columns_found", dataset_name=current_dataset_name, session_id=session_id)
        return {"error": "未找到用于时间序列图的有效数值列。"}

    result: Dict[str, Any] = await create_time_series_plot(df, date_column, valid_value_columns, title)
    
    if "error" not in result:
        session.add_to_history("time_series_plot", {"date_column": date_column, "value_columns": valid_value_columns, "title": title})
        logger.info("plot_time_series_success", dataset_name=current_dataset_name, date_column=date_column, value_columns=valid_value_columns, session_id=session_id)
    else:
        logger.error("plot_time_series_tool_error", dataset_name=current_dataset_name, date_column=date_column, value_columns=value_columns, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def create_interactive_chart(
    session_id: str,
    plot_type: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Create interactive charts using Plotly based on the current dataset.

    The specific arguments required will vary based on the `plot_type`.
    Refer to the `visualization.create_interactive_plot` function for details
    on required `kwargs` for each plot type.

    Args:
        plot_type (str): Type of plot to create (e.g., 'scatter', 'line', 'bar',
                         'histogram', 'box', 'violin', 'heatmap').
        **kwargs: Additional keyword arguments specific to the chosen `plot_type`.
                  These are passed directly to the underlying plotting function.
                  Examples: `x_column='col_A'`, `y_column='col_B'` for scatter,
                            `column='col_C'` for histogram.
    
    Returns:
        Dict[str, Any]: A dictionary containing the Plotly chart specification as JSON.
                        This JSON can be rendered by Plotly.js in the frontend.
                        Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("create_interactive_chart_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}
    
    # Log the attempt with kwargs for better debugging, but be mindful of sensitive data in kwargs if any.
    logger.info("create_interactive_chart_attempt", dataset_name=current_dataset_name, plot_type=plot_type, kwargs=kwargs, session_id=session_id)

    result: Dict[str, Any] = await create_interactive_plot(df, plot_type, **kwargs)

    if "error" not in result:
        session.add_to_history("interactive_plot", {"plot_type": plot_type, "kwargs_keys": list(kwargs.keys())}) # Log summary
        logger.info("create_interactive_chart_success", dataset_name=current_dataset_name, plot_type=plot_type, session_id=session_id)
    else:
        logger.error("create_interactive_chart_tool_error", dataset_name=current_dataset_name, plot_type=plot_type, kwargs=kwargs, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def test_normality(
    session_id: str,
    columns: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform normality tests (e.g., Shapiro-Wilk, D'Agostino's K^2) on specified numeric columns.

    Args:
        columns (List[str]): A list of numeric column names to test for normality.
        alpha (float): The significance level for the tests. Defaults to 0.05.
    
    Returns:
        Dict[str, Any]: A dictionary containing the normality test results for each column.
                        Returns `{"error": "message"}` on failure or if columns are unsuitable.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("test_normality_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    if not columns:
        logger.warn("test_normality_no_columns_specified", dataset_name=current_dataset_name, session_id=session_id)
        return {"error": "未指定用于正态性检验的列。"}

    valid_columns: List[str] = []
    for col_name in columns:
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            logger.warn("test_normality_invalid_column", dataset_name=current_dataset_name, column_name=col_name, session_id=session_id)
            return {"error": f"列 '{col_name}' 不是一个有效的数值列，无法进行正态性检验。"}
        valid_columns.append(col_name)

    if not valid_columns: # Should be caught by previous checks
        logger.warn("test_normality_no_valid_columns_found", dataset_name=current_dataset_name, requested_columns=columns, session_id=session_id)
        return {"error": "未找到用于正态性检验的有效数值列。"}

    result: Dict[str, Any] = await normality_test(df, valid_columns, alpha)
    
    if "error" not in result:
        session.add_to_history("normality_test", {"columns": valid_columns, "alpha": alpha})
        logger.info("test_normality_success", dataset_name=current_dataset_name, columns=valid_columns, alpha=alpha, session_id=session_id)
    else:
        logger.error("test_normality_tool_error", dataset_name=current_dataset_name, columns=columns, alpha=alpha, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def perform_t_test(
    session_id: str,
    column1: str,
    column2: Optional[str] = None,
    paired: bool = False,
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Perform a t-test (one-sample, two-sample independent, or paired) on specified columns.

    Args:
        column1 (str): The primary numeric column for the test. For a one-sample test, this is the column to test.
                       For two-sample or paired tests, this is the first column/group.
        column2 (Optional[str]): The second numeric column, required for two-sample or paired tests.
                                 If None, a one-sample t-test is performed on `column1`.
        paired (bool): If True and `column2` is provided, a paired t-test is performed. Defaults to False.
        alternative (str): Defines the alternative hypothesis. Can be 'two-sided', 'less', or 'greater'.
                           Defaults to 'two-sided'.
    
    Returns:
        Dict[str, Any]: A dictionary containing the t-test results (statistic, p-value, interpretation).
                        Returns `{"error": "message"}` on failure or invalid input.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("perform_t_test_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    # Validate column1
    if column1 not in df.columns or not pd.api.types.is_numeric_dtype(df[column1]):
        logger.warn("perform_t_test_invalid_column1", dataset_name=current_dataset_name, column_name=column1, session_id=session_id)
        return {"error": f"列 '{column1}' 不是一个有效的数值列，无法进行 t-检验。"}

    # Validate column2 if provided
    if column2 and (column2 not in df.columns or not pd.api.types.is_numeric_dtype(df[column2])):
        logger.warn("perform_t_test_invalid_column2", dataset_name=current_dataset_name, column_name=column2, session_id=session_id)
        return {"error": f"列 '{column2}' 不是一个有效的数值列，无法进行 t-检验。"}

    # Ensure column2 is provided for paired or two-sample (non-one-sample) tests
    if (paired or column2) and not column2: # If paired is true, column2 must be there. If column2 is there, it's two-sample.
         logger.warn("perform_t_test_missing_column2", dataset_name=current_dataset_name, test_type="paired or two-sample", session_id=session_id)
         return {"error": "对于配对或双样本 t-检验，必须提供 Column2。"}


    result: Dict[str, Any] = await t_test(df, column1, column2, paired, alternative)
    
    if "error" not in result:
        log_params = {"column1": column1, "column2": column2, "paired": paired, "alternative": alternative}
        session.add_to_history("t_test", log_params)
        logger.info("perform_t_test_success", dataset_name=current_dataset_name, session_id=session_id, **log_params)
    else:
        logger.error("perform_t_test_tool_error", dataset_name=current_dataset_name, column1=column1, column2=column2, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def perform_anova(
    session_id: str,
    dependent_var: str,
    independent_var: str
) -> Dict[str, Any]:
    """
    Perform a one-way ANOVA to test for differences in means across groups defined by a categorical variable.

    Args:
        dependent_var (str): The name of the numeric dependent variable.
        independent_var (str): The name of the categorical independent variable (grouping variable).
    
    Returns:
        Dict[str, Any]: A dictionary containing ANOVA results (F-statistic, p-value)
                        and potentially post-hoc test results. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("perform_anova_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    if dependent_var not in df.columns or not pd.api.types.is_numeric_dtype(df[dependent_var]):
        logger.warn("perform_anova_invalid_dependent_var", dataset_name=current_dataset_name, column_name=dependent_var, session_id=session_id)
        return {"error": f"因变量 '{dependent_var}' 必须是数值列。"}

    if independent_var not in df.columns: # Categorical check is usually handled by statsmodels, but ensure exists
        logger.warn("perform_anova_invalid_independent_var", dataset_name=current_dataset_name, column_name=independent_var, session_id=session_id)
        return {"error": f"自变量 '{independent_var}' 在数据集中未找到。"}

    # Ensure the independent variable has at least 2 groups for ANOVA
    if df[independent_var].nunique() < 2:
        logger.warn("perform_anova_not_enough_groups", dataset_name=current_dataset_name, column_name=independent_var, num_groups=df[independent_var].nunique(), session_id=session_id)
        return {"error": f"自变量 '{independent_var}' 必须至少有两个不同的组才能进行 ANOVA 分析。"}


    result: Dict[str, Any] = await anova_test(df, dependent_var, independent_var)
    
    if "error" not in result:
        log_params = {"dependent_var": dependent_var, "independent_var": independent_var}
        session.add_to_history("anova_test", log_params)
        logger.info("perform_anova_success", dataset_name=current_dataset_name, session_id=session_id, **log_params)
    else:
        logger.error("perform_anova_tool_error", dataset_name=current_dataset_name, dependent_var=dependent_var, independent_var=independent_var, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def perform_chi_square(
    session_id: str,
    column1: str,
    column2: str
) -> Dict[str, Any]:
    """
    Perform a Chi-square test of independence between two categorical variables.

    Args:
        column1 (str): The name of the first categorical variable.
        column2 (str): The name of the second categorical variable.
    
    Returns:
        Dict[str, Any]: A dictionary containing Chi-square test results (statistic, p-value,
                        degrees of freedom) and the contingency table. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("perform_chi_square_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    for col_name in [column1, column2]:
        if col_name not in df.columns: # Basic check, type check done by chi_square_test
            logger.warn("perform_chi_square_column_not_found", dataset_name=current_dataset_name, column_name=col_name, session_id=session_id)
            return {"error": f"列 '{col_name}' 在数据集中未找到，无法进行卡方检验。"}

    result: Dict[str, Any] = await chi_square_test(df, column1, column2)
    
    if "error" not in result:
        log_params = {"column1": column1, "column2": column2}
        session.add_to_history("chi_square_test", log_params)
        logger.info("perform_chi_square_success", dataset_name=current_dataset_name, session_id=session_id, **log_params)
    else:
        logger.error("perform_chi_square_tool_error", dataset_name=current_dataset_name, column1=column1, column2=column2, error=result.get("error"), session_id=session_id)
    return result

@mcp.tool()
async def test_correlation(
    session_id: str,
    column1: str,
    column2: str,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Test the significance of the correlation between two numeric variables.

    Args:
        column1 (str): The name of the first numeric variable.
        column2 (str): The name of the second numeric variable.
        method (str): The correlation method to use ('pearson', 'spearman', 'kendall').
                      Defaults to 'pearson'.
    
    Returns:
        Dict[str, Any]: A dictionary containing the correlation coefficient, p-value,
                        and an interpretation of the test. Returns `{"error": "message"}` on failure.
    """
    if session_id not in user_sessions:
        user_sessions[session_id] = DataSession()
    session = user_sessions[session_id]

    df: Optional[pd.DataFrame] = session.get_current_df()
    current_dataset_name: Optional[str] = session.current_df
    if df is None:
        logger.warn("test_correlation_no_dataset", session_id=session_id)
        return {"error": "没有加载数据集。请先上传文件。"}

    for col_name in [column1, column2]:
        if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
            logger.warn("test_correlation_invalid_column", dataset_name=current_dataset_name, column_name=col_name, session_id=session_id)
            return {"error": f"列 '{col_name}' 必须是数值列才能进行相关性检验。"}

    result: Dict[str, Any] = await correlation_test(df, column1, column2, method)
    
    if "error" not in result:
        log_params = {"column1": column1, "column2": column2, "method": method}
        session.add_to_history("correlation_test", log_params)
        logger.info("test_correlation_success", dataset_name=current_dataset_name, session_id=session_id, **log_params)
    else:
        logger.error("test_correlation_tool_error", dataset_name=current_dataset_name, column1=column1, column2=column2, method=method, error=result.get("error"), session_id=session_id)
    return result

# Run the server
def serve() -> None:
    """
    Run the MCP (Machine Control Protocol) server.

    This function initializes and starts the FastMCP server using stdio for communication.
    It logs server start, stop (on KeyboardInterrupt), and any critical errors that might
    cause the server to exit.
    """
    import sys
    from mcp.server.stdio import stdio_server
    
    logger.info("mcp_server_starting", message="Starting Data Analysis Assistant MCP Server")
    
    try:
        asyncio.run(stdio_server(mcp))
    except KeyboardInterrupt:
        logger.info("mcp_server_stopped_by_user", reason="KeyboardInterrupt")
    except Exception as e:
        logger.error("mcp_server_error", error=str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    serve()
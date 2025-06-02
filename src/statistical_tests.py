"""Statistical testing tools for data analysis"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)


async def normality_test(
    df: pd.DataFrame,
    columns: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for normality using multiple methods.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to test
        alpha: Significance level
    
    Returns:
        Test results including p-values and interpretations
    """
    results = {}
    
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found"}
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            results[col] = {"error": f"Column '{col}' is not numeric"}
            continue
        
        data = df[col].dropna()
        
        if len(data) < 3:
            results[col] = {"error": "Insufficient data (need at least 3 values)"}
            continue
        
        col_results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            col_results["shapiro_wilk"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > alpha,
                "interpretation": "正态分布" if p_value > alpha else "非正态分布"
            }
        
        # Kolmogorov-Smirnov test
        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        col_results["kolmogorov_smirnov"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > alpha,
            "interpretation": "正态分布" if p_value > alpha else "非正态分布"
        }
        
        # Anderson-Darling test
        result = stats.anderson(data)
        col_results["anderson_darling"] = {
            "statistic": float(result.statistic),
            "critical_values": dict(zip(result.significance_level, result.critical_values)),
            "is_normal": result.statistic < result.critical_values[2],  # 5% level
            "interpretation": "正态分布" if result.statistic < result.critical_values[2] else "非正态分布"
        }
        
        # D'Agostino's K-squared test
        if len(data) >= 8:
            stat, p_value = stats.normaltest(data)
            col_results["dagostino_k2"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > alpha,
                "interpretation": "正态分布" if p_value > alpha else "非正态分布"
            }
        
        # Summary
        normal_count = sum(1 for test in col_results.values() if test.get("is_normal", False))
        col_results["summary"] = {
            "likely_normal": normal_count >= len(col_results) / 2,
            "normal_tests_passed": normal_count,
            "total_tests": len(col_results)
        }
        
        results[col] = col_results
    
    return {
        "columns_tested": columns,
        "alpha": alpha,
        "results": results
    }

def _normality_test_sync(df: pd.DataFrame, columns: List[str], alpha: float) -> Dict[str, Any]:
    results = {}

    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found"}
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            results[col] = {"error": f"Column '{col}' is not numeric"}
            continue

        data = df[col].dropna()

        if len(data) < 3:
            results[col] = {"error": "Insufficient data (need at least 3 values)"}
            continue

        col_results = {}

        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            col_results["shapiro_wilk"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > alpha,
                "interpretation": "正态分布" if p_value > alpha else "非正态分布"
            }

        # Kolmogorov-Smirnov test
        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        col_results["kolmogorov_smirnov"] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_normal": p_value > alpha,
            "interpretation": "正态分布" if p_value > alpha else "非正态分布"
        }

        # Anderson-Darling test
        result = stats.anderson(data)
        col_results["anderson_darling"] = {
            "statistic": float(result.statistic),
            "critical_values": dict(zip(result.significance_level, result.critical_values)),
            "is_normal": result.statistic < result.critical_values[2],  # 5% level
            "interpretation": "正态分布" if result.statistic < result.critical_values[2] else "非正态分布"
        }

        # D'Agostino's K-squared test
        if len(data) >= 8:
            stat, p_value = stats.normaltest(data)
            col_results["dagostino_k2"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > alpha,
                "interpretation": "正态分布" if p_value > alpha else "非正态分布"
            }

        # Summary
        normal_count = sum(1 for test in col_results.values() if test.get("is_normal", False))
        col_results["summary"] = {
            "likely_normal": normal_count >= len(col_results) / 2,
            "normal_tests_passed": normal_count,
            "total_tests": len(col_results)
        }

        results[col] = col_results

    return {
        "columns_tested": columns,
        "alpha": alpha,
        "results": results
    }

async def normality_test(
    df: pd.DataFrame,
    columns: List[str],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for normality using multiple methods.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to test
        alpha: Significance level
    
    Returns:
        Test results including p-values and interpretations
    """
    try:
        return await asyncio.to_thread(_normality_test_sync, df, columns, alpha)
    except Exception as e:
        return {"error": str(e)}

def _t_test_sync(df: pd.DataFrame, column1: str, column2: Optional[str], paired: bool, alternative: str) -> Dict[str, Any]:
    # Validate columns
    if column1 not in df.columns:
        return {"error": f"Column '{column1}' not found"}

    if not pd.api.types.is_numeric_dtype(df[column1]):
        return {"error": f"Column '{column1}' is not numeric"}

    data1 = df[column1].dropna()

    if column2:
        # Two-sample or paired t-test
        if column2 not in df.columns:
            return {"error": f"Column '{column2}' not found"}
        
        if not pd.api.types.is_numeric_dtype(df[column2]):
            return {"error": f"Column '{column2}' is not numeric"}
        
        data2 = df[column2].dropna()
        
        if paired:
            # Paired t-test
            if len(data1) != len(data2):
                return {"error": "Paired t-test requires equal sample sizes"}
            
            stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)
            test_type = "paired"
        else:
            # Independent two-sample t-test
            # First check for equal variances
            _, levene_p = stats.levene(data1, data2)
            equal_var = levene_p > 0.05
            
            stat, p_value = stats.ttest_ind(
                data1, data2,
                equal_var=equal_var,
                alternative=alternative
            )
            test_type = "independent"
    else:
        # One-sample t-test (test against mean of 0)
        stat, p_value = stats.ttest_1samp(data1, 0, alternative=alternative)
        test_type = "one-sample"
        data2 = None

    # Calculate effect size (Cohen's d)
    if data2 is not None:
        # For paired samples, Cohen's d is calculated on the differences
        if paired:
            diff = data1 - data2
            cohens_d = diff.mean() / diff.std()
        else:
            pooled_std = np.sqrt(((len(data1) - 1) * data1.std() ** 2 + 
                                 (len(data2) - 1) * data2.std() ** 2) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (data1.mean() - data2.mean()) / pooled_std
    else:
        cohens_d = data1.mean() / data1.std()

    # Interpretation
    alpha = 0.05
    significant = p_value < alpha

    if alternative == "two-sided":
        interpretation = "存在显著差异" if significant else "无显著差异"
    elif alternative == "less":
        interpretation = f"{column1} 显著小于 {column2 or '0'}" if significant else "无显著差异"
    else:
        interpretation = f"{column1} 显著大于 {column2 or '0'}" if significant else "无显著差异"

    result_dict = {
        "test_type": test_type,
        "alternative": alternative,
        "t_statistic": float(stat),
        "p_value": float(p_value),
        "degrees_of_freedom": len(data1) + (len(data2) if data2 is not None and not paired else 0) - (2 if data2 is not None and not paired else 1 if not paired else len(data1) -1) ,
        "cohens_d": float(cohens_d),
        "effect_size": "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large",
        "significant": significant,
        "interpretation": interpretation,
        "sample_stats": {
            column1: {
                "n": len(data1),
                "mean": float(data1.mean()),
                "std": float(data1.std())
            }
        }
    }

    if data2 is not None:
        result_dict["sample_stats"][column2] = {
            "n": len(data2),
            "mean": float(data2.mean()),
            "std": float(data2.std())
        }
        if not paired:
            result_dict["equal_variances"] = levene_p > 0.05

    return result_dict

async def t_test(
    df: pd.DataFrame,
    column1: str,
    column2: Optional[str] = None,
    paired: bool = False,
    alternative: str = "two-sided"
) -> Dict[str, Any]:
    """
    Perform t-test (one-sample, two-sample, or paired).

    Args:
        df: DataFrame containing the data
        column1: First column name
        column2: Second column name (for two-sample or paired test)
        paired: Whether to perform paired t-test
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Test results including t-statistic, p-value, and interpretation
    """
    try:
        return await asyncio.to_thread(_t_test_sync, df, column1, column2, paired, alternative)
    except Exception as e:
        return {"error": str(e)}

def _anova_test_sync(df: pd.DataFrame, dependent_var: str, independent_var: str) -> Dict[str, Any]:
    # Validate columns
    if dependent_var not in df.columns:
        return {"error": f"Column '{dependent_var}' not found"}

    if independent_var not in df.columns:
        return {"error": f"Column '{independent_var}' not found"}

    if not pd.api.types.is_numeric_dtype(df[dependent_var]):
        return {"error": f"Dependent variable '{dependent_var}' must be numeric"}

    # Prepare data
    clean_df = df[[dependent_var, independent_var]].dropna()
    groups = clean_df.groupby(independent_var)[dependent_var].apply(list)

    if len(groups) < 2:
        return {"error": "Need at least 2 groups for ANOVA"}

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups.values)

    # Calculate effect size (eta-squared)
    ss_between = sum(len(group) * (np.mean(group) - clean_df[dependent_var].mean()) ** 2
                    for group in groups.values)
    ss_total = sum((clean_df[dependent_var] - clean_df[dependent_var].mean()) ** 2)
    eta_squared = ss_between / ss_total

    # Group statistics
    group_stats = {}
    for name, group in groups.items():
        group_stats[str(name)] = {
            "n": len(group),
            "mean": float(np.mean(group)),
            "std": float(np.std(group)),
            "min": float(np.min(group)),
            "max": float(np.max(group))
        }

    result_dict = {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "effect_size": "small" if eta_squared < 0.06 else "medium" if eta_squared < 0.14 else "large",
        "significant": p_value < 0.05,
        "interpretation": "组间存在显著差异" if p_value < 0.05 else "组间无显著差异",
        "groups": len(groups),
        "total_observations": len(clean_df),
        "group_statistics": group_stats
    }

    # Post-hoc tests if significant
    if p_value < 0.05 and len(groups) > 2:
        # Tukey HSD test
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey = pairwise_tukeyhsd(
            clean_df[dependent_var],
            clean_df[independent_var],
            alpha=0.05
        )

        result_dict["post_hoc"] = {
            "method": "Tukey HSD",
            "results": str(tukey)
        }

    return result_dict

async def anova_test(
    df: pd.DataFrame,
    dependent_var: str,
    independent_var: str
) -> Dict[str, Any]:
    """
    Perform one-way ANOVA test.
    
    Args:
        df: DataFrame containing the data
        dependent_var: Dependent variable (numeric)
        independent_var: Independent variable (categorical)
    
    Returns:
        ANOVA results including F-statistic, p-value, and post-hoc tests
    """
    try:
        return await asyncio.to_thread(_anova_test_sync, df, dependent_var, independent_var)
    except Exception as e:
        return {"error": str(e)}

def _chi_square_test_sync(df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
    # Validate columns
    for col in [column1, column2]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found"}

    # Create contingency table
    contingency_table = pd.crosstab(df[column1], df[column2])

    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Calculate Cramér's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
    if min_dim == 0: # Avoid division by zero if one dimension has only 1 level
        cramers_v = np.nan
    else:
        cramers_v = np.sqrt(chi2 / (n * min_dim))

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(cramers_v) if not np.isnan(cramers_v) else None,
        "effect_size": "small" if cramers_v < 0.1 else "medium" if cramers_v < 0.3 else "large" if not np.isnan(cramers_v) else "N/A",
        "significant": p_value < 0.05,
        "interpretation": "变量间存在显著关联" if p_value < 0.05 else "变量间无显著关联",
        "contingency_table": contingency_table.to_dict(),
        "table_shape": contingency_table.shape
    }

async def chi_square_test(
    df: pd.DataFrame,
    column1: str,
    column2: str
) -> Dict[str, Any]:
    """
    Perform chi-square test of independence.
    
    Args:
        df: DataFrame containing the data
        column1: First categorical variable
        column2: Second categorical variable
    
    Returns:
        Chi-square test results
    """
    try:
        return await asyncio.to_thread(_chi_square_test_sync, df, column1, column2)
    except Exception as e:
        return {"error": str(e)}

def _correlation_test_sync(df: pd.DataFrame, column1: str, column2: str, method: str) -> Dict[str, Any]:
    # Validate columns
    for col in [column1, column2]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found"}

        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric"}

    # Get clean data
    clean_df = df[[column1, column2]].dropna()

    if len(clean_df) < 3:
        return {"error": "Need at least 3 observations"}

    # Calculate correlation
    if method == "pearson":
        corr, p_value = stats.pearsonr(clean_df[column1], clean_df[column2])
    elif method == "spearman":
        corr, p_value = stats.spearmanr(clean_df[column1], clean_df[column2])
    elif method == "kendall":
        corr, p_value = stats.kendalltau(clean_df[column1], clean_df[column2])
    else:
        return {"error": f"Unknown method: {method}"}

    # Confidence interval for Pearson correlation
    confidence_interval = None
    if method == "pearson" and len(clean_df) > 3 : # Need n > 3 for se calculation
        # Fisher z-transformation
        z = np.arctanh(corr)
        se = 1 / np.sqrt(len(clean_df) - 3)
        z_critical = stats.norm.ppf(0.975)  # 95% CI

        ci_lower = np.tanh(z - z_critical * se)
        ci_upper = np.tanh(z + z_critical * se)

        confidence_interval = [float(ci_lower), float(ci_upper)]

    # Interpretation
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = "weak"
    elif abs_corr < 0.7:
        strength = "moderate"
    else:
        strength = "strong"

    direction = "positive" if corr > 0 else "negative"

    result_dict = {
        "method": method,
        "correlation": float(corr),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "strength": strength,
        "direction": direction,
        "interpretation": f"{strength} {direction} 相关性" + ("（显著）" if p_value < 0.05 else "（不显著）"),
        "n_observations": len(clean_df)
    }

    if confidence_interval:
        result_dict["confidence_interval_95"] = confidence_interval

    return result_dict

async def correlation_test(
    df: pd.DataFrame,
    column1: str,
    column2: str,
    method: str = "pearson"
) -> Dict[str, Any]:
    """
    Test correlation significance between two variables.
    
    Args:
        df: DataFrame containing the data
        column1: First variable
        column2: Second variable
        method: 'pearson', 'spearman', or 'kendall'
    
    Returns:
        Correlation coefficient and significance test
    """
    try:
        return await asyncio.to_thread(_correlation_test_sync, df, column1, column2, method)
    except Exception as e:
        return {"error": str(e)}
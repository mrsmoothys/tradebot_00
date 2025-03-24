# utils/diagnostics.py
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class DataDiagnostics:
    """Provides diagnostic utilities for data quality assessment."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_dataframe(self, df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, Any]:
        """
        Perform comprehensive analysis of DataFrame quality and structure.
        
        Args:
            df: DataFrame to analyze
            name: Name to identify this DataFrame in logs
            
        Returns:
            Dictionary of diagnostic metrics
        """
        results = {
            "name": name,
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_count": len(df.columns),
            "row_count": len(df),
            "dtypes": {},
            "missing_values": {},
            "numeric_stats": {},
            "column_categories": {
                "numeric": [],
                "datetime": [],
                "categorical": [],
                "boolean": [],
                "other": []
            }
        }
        
        # Analyze data types and missing values
        for col in df.columns:
            dtype = str(df[col].dtype)
            results["dtypes"][col] = dtype
            
            # Count missing values
            missing = df[col].isna().sum()
            if missing > 0:
                results["missing_values"][col] = {
                    "count": missing,
                    "percentage": missing / len(df) * 100
                }
            
            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                results["column_categories"]["numeric"].append(col)
                
                # Basic stats for numeric columns
                if not df[col].isna().all():
                    results["numeric_stats"][col] = {
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "has_zeros": (df[col] == 0).sum(),
                        "has_negative": (df[col] < 0).sum(),
                        "has_infinite": np.isinf(df[col]).sum() if pd.api.types.is_float_dtype(df[col]) else 0
                    }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                results["column_categories"]["datetime"].append(col)
            elif pd.api.types.is_bool_dtype(df[col]):
                results["column_categories"]["boolean"].append(col)
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < len(df) * 0.1:
                results["column_categories"]["categorical"].append(col)
            else:
                results["column_categories"]["other"].append(col)
        
        # Log key findings
        self.logger.info(f"DataFrame '{name}': {results['shape'][0]} rows, {results['shape'][1]} columns")
        
        if results["missing_values"]:
            self.logger.warning(f"Found missing values in {len(results['missing_values'])} columns")
            
        # Check for potential data issues
        for col, stats in results["numeric_stats"].items():
            if stats["has_infinite"] > 0:
                self.logger.warning(f"Column '{col}' contains {stats['has_infinite']} infinite values")
        
        return results
    
    def log_full_analysis(self, df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Perform and log a full analysis of the DataFrame.
        
        Args:
            df: DataFrame to analyze
            name: Name to identify this DataFrame in logs
        """
        analysis = self.analyze_dataframe(df, name)
        
        # Log detailed analysis
        self.logger.info(f"=== Detailed Analysis for '{name}' ===")
        self.logger.info(f"Shape: {analysis['shape']}")
        self.logger.info(f"Memory Usage: {analysis['memory_usage_mb']:.2f} MB")
        
        # Log column type distribution
        type_counts = {k: len(v) for k, v in analysis["column_categories"].items() if len(v) > 0}
        self.logger.info(f"Column Types: {type_counts}")
        
        # Log missing values summary
        if analysis["missing_values"]:
            missing_summary = {col: f"{info['percentage']:.1f}%" 
                              for col, info in analysis["missing_values"].items()}
            self.logger.info(f"Missing Values: {missing_summary}")
        else:
            self.logger.info("No missing values detected")
        
        # Log problematic columns
        problem_cols = []
        for col, stats in analysis.get("numeric_stats", {}).items():
            issues = []
            if stats.get("has_infinite", 0) > 0:
                issues.append(f"{stats['has_infinite']} infinite values")
            if stats.get("has_negative", 0) > 0 and "price" in col.lower():
                issues.append(f"{stats['has_negative']} negative values")
            
            if issues:
                problem_cols.append(f"{col}: {', '.join(issues)}")
        
        if problem_cols:
            self.logger.warning(f"Problematic columns detected:\n  " + "\n  ".join(problem_cols))
        
        self.logger.info("=" * 50)
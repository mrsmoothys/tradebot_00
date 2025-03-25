import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class DataQualityMonitor:
    """Monitors and reports data quality issues."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.quality_metrics = {
            'nan_counts': {},
            'inf_counts': {},
            'type_errors': {},
            'value_range_issues': {}
        }
    
    def check_dataframe(self, df: pd.DataFrame, name: str = "unnamed") -> pd.DataFrame:
        """Perform comprehensive data quality checks and log issues."""
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            self.quality_metrics['nan_counts'][name] = nan_counts[nan_counts > 0].to_dict()
            self.logger.warning(f"NaN values detected in {name}: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        
        # Value range checks
        for col in numeric_cols:
            try:
                col_min, col_max = df[col].min(), df[col].max()
                
                # Check for concerning values
                if abs(col_max) > 1e6 or abs(col_min) > 1e6:
                    self.quality_metrics['value_range_issues'][f"{name}.{col}"] = (col_min, col_max)
                    self.logger.warning(f"Extreme values in {name}.{col}: min={col_min}, max={col_max}")
            except Exception as e:
                self.quality_metrics['type_errors'][f"{name}.{col}"] = str(e)
        
        return df
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate data quality report."""
        return {
            'metrics': self.quality_metrics,
            'recommendations': self._generate_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on observed issues."""
        recommendations = []
        
        if self.quality_metrics['nan_counts']:
            recommendations.append("Implement more robust missing value handling")
        
        if self.quality_metrics['inf_counts']:
            recommendations.append("Add explicit infinity checks in numeric calculations")
            
        if self.quality_metrics['value_range_issues']:
            recommendations.append("Consider implementing value normalization or clipping")
            
        return recommendations
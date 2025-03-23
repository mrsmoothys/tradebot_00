import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:
    """
    Implements feature selection mechanisms.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the feature selector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.selection_config = self.config.get('features.selection', {})
        self.method = self.selection_config.get('method', 'correlation')
        self.params = self.selection_config.get('params', {})
    
    def select_features(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using the configured method.
        
        Args:
            df: DataFrame with features
            target_column: Target column for selection
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not feature_cols:
            self.logger.warning("No feature columns found for selection")
            return df, []
        
        # Calculate target (e.g., next period's return)
        df['target'] = df[target_column].pct_change(periods=1).shift(-1)
        
        # Drop NaNs
        df_clean = df.dropna()
        
        # Select method
        if self.method == 'correlation':
            selected_features = self._correlation_selection(df_clean, feature_cols)
        elif self.method == 'f_regression':
            selected_features = self._f_regression_selection(df_clean, feature_cols)
        elif self.method == 'recursive_feature_elimination':
            selected_features = self._rfe_selection(df_clean, feature_cols)
        else:
            self.logger.warning(f"Unknown feature selection method: {self.method}")
            selected_features = feature_cols
        
        # Keep only selected features and OHLCV
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume'] + selected_features
        result_df = df[columns_to_keep].copy()
        
        return result_df, selected_features
    
    def _correlation_selection(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        Select features based on correlation with target.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature columns
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation with target
        correlations = df[feature_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
        
        # Select top N features
        n_features = self.params.get('n_features', min(30, len(feature_cols)))
        selected_features = correlations.head(n_features).index.tolist()
        
        return selected_features
    
    def _f_regression_selection(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        Select features using F-regression.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature columns
            
        Returns:
            List of selected feature names
        """
        # Prepare data
        X = df[feature_cols]
        y = df['target']
        
        # Select top N features
        n_features = self.params.get('n_features', min(30, len(feature_cols)))
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selector.fit(X, y)
        
        # Get selected feature mask
        feature_mask = selector.get_support()
        
        # Return selected features
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if feature_mask[i]]
        
        return selected_features
    
    def _rfe_selection(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """
        Select features using recursive feature elimination.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            
        Returns:
            List of selected features
        """
        # Make sure we only select from numeric features
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                self.logger.warning(f"Skipping non-numeric feature: {col}")
        
        if not numeric_features:
            self.logger.warning("No numeric features available for selection")
            return feature_cols
        
        # Prepare data for feature selection
        X = df[numeric_features].values
        
        # Make sure to use numeric target
        y = df['target'].values  # Use the target column already created in select_features method
        
        # Get RFE parameters
        n_features = self.config.get('features.selection.params.n_features', 30)
        step = self.config.get('features.selection.params.step', 1)
        
        # Initialize estimator (Random Forest is a good default)
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Create RFE selector
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, len(numeric_features)), step=step)
        
        # Fit RFE with error handling
        try:
            rfe.fit(X, y)
            
            # Get selected features
            selected_feature_indices = np.where(rfe.support_)[0]
            selected = [numeric_features[i] for i in selected_feature_indices]
            
            # Log selection info
            self.logger.info(f"Selected {len(selected)} features using RFE")
            
            return selected
        except Exception as e:
            self.logger.error(f"Error in RFE feature selection: {e}")
            # Fall back to the original features on error
            return numeric_features
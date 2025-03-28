import pandas as pd
import os
import numpy as np
import logging
import ta
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.profiling import profile
from utils.progress import ProgressTracker
import hashlib
from utils.data_quality import DataQualityMonitor


class FeatureEngineer:
    """
    Implements feature generation from raw OHLCV data.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the feature engineer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.indicators_config = self.config.get('features.indicators', [])


    def categorize_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize DataFrame columns by data type for optimized processing.
        
        Args:
            df: DataFrame with mixed column types
            
        Returns:
            Dictionary mapping feature types to column lists
        """
        feature_categories = {
            'numeric': [],          # Numeric features for model input
            'datetime': [],         # Date/time related features
            'categorical': [],      # Categorical features
            'ohlcv': [],            # Price and volume data
            'metadata': [],         # Non-feature columns like IDs
            'derived_targets': []   # Target variables or predictions
        }
        
        # Define standard OHLCV and metadata columns
        ohlcv_patterns = ['open', 'high', 'low', 'close', 'volume']
        metadata_patterns = ['time', 'timestamp', 'date', 'symbol', 'trades', 'quote']
        target_patterns = ['target', 'signal', 'prediction', 'confidence']
        
        # Categorize each column
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for OHLCV columns
            if any(pattern in col_lower for pattern in ohlcv_patterns):
                feature_categories['ohlcv'].append(col)
                continue
                
            # Check for metadata columns
            if any(pattern in col_lower for pattern in metadata_patterns):
                feature_categories['metadata'].append(col)
                continue
                
            # Check for target or prediction columns
            if any(pattern in col_lower for pattern in target_patterns):
                feature_categories['derived_targets'].append(col)
                continue
                
            # Categorize by data type
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_categories['numeric'].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_categories['datetime'].append(col)
            else:
                feature_categories['categorical'].append(col)
        
        # Log categorization results
        self.logger.debug(f"Feature categorization results: {', '.join([f'{k}: {len(v)}' for k, v in feature_categories.items()])}")
        
        return feature_categories
    
   # Modifications to feature_engineering.py

    @profile
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features with efficient memory utilization."""
        # Create local reference to essential columns only
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_df = df[essential_cols].copy()
        
        # Downcast datatypes for memory efficiency
        for col in ohlcv_df.columns:
            if ohlcv_df[col].dtype == 'float64':
                ohlcv_df[col] = ohlcv_df[col].astype('float32')
        
        # Process indicators in batches to prevent memory spikes
        grouped_indicators = [
            # Group 1: Basic indicators (low memory impact)
            [ind for ind in self.indicators_config if ind['name'] in ['rsi', 'sma', 'ema']],
            # Group 2: Medium complexity indicators
            [ind for ind in self.indicators_config if ind['name'] in ['macd', 'bollinger_bands', 'atr']],
            # Group 3: Complex indicators (high memory impact)
            [ind for ind in self.indicators_config if ind['name'] in ['adx', 'ichimoku', 'vwap', 'slope_momentum']]
        ]
        
        # Track problematic features for reporting
        problematic_features = set()
        
        # Process each batch with memory cleanup
        import gc
        for batch_idx, indicator_batch in enumerate(grouped_indicators):
            self.logger.info(f"Processing indicator batch {batch_idx+1}/{len(grouped_indicators)}")
            
            for indicator_config in indicator_batch:
                indicator_name = indicator_config['name']
                params = indicator_config.get('params', {})
                
                try:
                    # Generate feature
                    method = getattr(self, f"_generate_{indicator_name}")
                    method(ohlcv_df, **params)  # In-place modification
                except AttributeError:
                    self.logger.warning(f"Indicator method not found: _generate_{indicator_name}")
                except Exception as e:
                    self.logger.error(f"Error generating {indicator_name}: {e}")
                    problematic_features.add(indicator_name)
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Replace infinities and NaN values efficiently
        numeric_cols = ohlcv_df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
        ohlcv_df[numeric_cols] = ohlcv_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        ohlcv_df[numeric_cols] = ohlcv_df[numeric_cols].fillna(0)
        
        # Log feature generation metrics
        feature_count = len(ohlcv_df.columns) - len(essential_cols)
        self.logger.info(f"Generated {feature_count} features with {len(problematic_features)} issues")
        
        return ohlcv_df

    
    @profile
    def _generate_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price action pattern features."""
        # Calculate candle body and wick sizes
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100  # Body size as percentage
        df['upper_wick'] = (df['high'] - df['close'].clip(lower=df['open'])) / df['open'] * 100
        df['lower_wick'] = (df['open'].clip(upper=df['close']) - df['low']) / df['open'] * 100
        
        # Detect engulfing patterns
        df['bullish_engulfing'] = (
            (df['open'].shift(1) > df['close'].shift(1)) &  # Previous candle was bearish
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['open'] < df['close'].shift(1)) &  # Current open below previous close
            (df['close'] > df['open'].shift(1))  # Current close above previous open
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle was bullish
            (df['open'] > df['close']) &  # Current candle is bearish
            (df['close'] < df['open'].shift(1)) &  # Current close below previous open
            (df['open'] > df['close'].shift(1))  # Current open above previous close
        ).astype(int)
        
        return df

    @profile
    def _generate_slope_momentum(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Generate price slope momentum indicators more efficiently.
        
        Args:
            df: DataFrame with OHLCV data
            windows: List of lookback periods for calculating slope
            
        Returns:
            DataFrame with added slope momentum features
        """
        # Pre-allocate numpy arrays for better performance
        prices = df['close'].values
        n = len(prices)
        
        for window in windows:
            # Pre-allocate arrays
            slopes = np.full(n, np.nan)
            norm_slopes = np.full(n, np.nan)
            
            # Only process where we have enough data
            for i in range(window, n):
                # Get window of prices
                window_prices = prices[i-window:i]
                indices = np.arange(window)
                
                # Calculate slope using polyfit
                slope = np.polyfit(indices, window_prices, 1)[0]
                slopes[i] = slope
                
                # Normalize
                if prices[i] != 0:
                    norm_slopes[i] = slope / prices[i] * 1000
            
            # Add to dataframe
            df[f'price_slope_{window}'] = slopes
            df[f'norm_slope_{window}'] = norm_slopes
            df[f'slope_momentum_{window}'] = np.concatenate([[np.nan], np.diff(norm_slopes)])
            
        return df

    @profile
    def _generate_rsi(self, df: pd.DataFrame, window: int = 14) -> None:
        """Generate RSI indicator with optimized calculation."""
        # Get price series efficiently
        close = df['close'].values
        
        # Calculate price changes
        diff = np.zeros(len(close))
        diff[1:] = np.diff(close)
        
        # Separate gains and losses
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        
        # Calculate averages using numba if available
        try:
            import numba as nb
            
            @nb.njit
            def calculate_rsi(gain, loss, window):
                avg_gain = np.zeros_like(gain)
                avg_loss = np.zeros_like(loss)
                
                # First value is simple average
                avg_gain[window] = np.mean(gain[1:window+1])
                avg_loss[window] = np.mean(loss[1:window+1])
                
                # Calculate subsequent values
                for i in range(window+1, len(gain)):
                    avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
                    avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
                
                # Calculate RS and RSI
                rs = np.zeros_like(gain)
                rsi = np.zeros_like(gain)
                
                for i in range(window, len(gain)):
                    if avg_loss[i] == 0:
                        rsi[i] = 100
                    else:
                        rs[i] = avg_gain[i] / avg_loss[i]
                        rsi[i] = 100 - (100 / (1 + rs[i]))
                
                return rsi
            
            df[f'rsi_{window}'] = calculate_rsi(gain, loss, window)
            
        except ImportError:
            # Fallback to vectorized numpy implementation
            avg_gain = np.zeros_like(gain)
            avg_loss = np.zeros_like(loss)
            
            # Initialize first average
            avg_gain[window] = np.mean(gain[1:window+1])
            avg_loss[window] = np.mean(loss[1:window+1])
            
            # Use vectorized operations for subsequent values
            for i in range(window+1, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
                avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
            
            # Calculate RS and RSI
            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
            rsi = np.zeros_like(rs)
            rsi[window:] = 100 - (100 / (1 + rs[window:]))
            
            df[f'rsi_{window}'] = rsi
    
    @profile
    def _generate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> pd.DataFrame:
        """Generate MACD indicator."""
        macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df[f'macd_{fast}_{slow}'] = macd.macd()
        df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
        df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
        return df
    
    @profile
    def _generate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, 
                                 num_std: float = 2.0) -> pd.DataFrame:
        """Generate Bollinger Bands indicator."""
        bollinger = ta.volatility.BollingerBands(df['close'], window=window, window_dev=num_std)
        df[f'bollinger_high_{window}'] = bollinger.bollinger_hband()
        df[f'bollinger_low_{window}'] = bollinger.bollinger_lband()
        df[f'bollinger_mid_{window}'] = bollinger.bollinger_mavg()
        df[f'bollinger_width_{window}'] = bollinger.bollinger_wband()
        return df
    
    @profile
    def _generate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Generate ATR indicator."""
        df[f'atr_{window}'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=window
        ).average_true_range()
        return df
    
    @profile
    def _generate_sma(self, df: pd.DataFrame, window: List[int]) -> pd.DataFrame:
        """Generate SMA indicators for multiple windows."""
        for w in window:
            df[f'sma_{w}'] = ta.trend.SMAIndicator(df['close'], window=w).sma_indicator()
        return df
    
    @profile
    def _generate_ema(self, df: pd.DataFrame, window: List[int]) -> pd.DataFrame:
        """Generate EMA indicators for multiple windows."""
        for w in window:
            df[f'ema_{w}'] = ta.trend.EMAIndicator(df['close'], window=w).ema_indicator()
        return df
    
    # Add these to the FeatureEngineer class

    @profile
    def _generate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Generate Average Directional Index indicator."""
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=window)
        df[f'adx_{window}'] = adx.adx()
        df[f'adx_pos_{window}'] = adx.adx_pos()
        df[f'adx_neg_{window}'] = adx.adx_neg()
        return df

    @profile
    def _generate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Ichimoku Cloud indicators."""
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        return df

    @profile
    def _generate_vwap(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Generate Volume Weighted Average Price."""
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=window
        ).volume_weighted_average_price()
        return df


    def apply_pca(self, df: pd.DataFrame, n_components: int = 30) -> Tuple[pd.DataFrame, PCA, StandardScaler]:
        """
        Apply PCA dimensionality reduction to features.
        
        Args:
            df: DataFrame with features
            n_components: Number of PCA components
            
        Returns:
            Tuple of (DataFrame with PCA components, PCA model, StandardScaler model)
        """
        # Separate OHLCV
        ohlcv = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if not feature_cols:
            self.logger.warning("No feature columns found for PCA")
            return df, None, None
        
        # Copy features
        features = df[feature_cols].copy()
        
        # Replace inf/NaN values
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(features.mean(), inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(feature_cols)))
        components = pca.fit_transform(features_scaled)
        
        # Create component DataFrame
        component_df = pd.DataFrame(
            components,
            columns=[f'pc_{i+1}' for i in range(components.shape[1])],
            index=df.index
        )
        
        # Combine with OHLCV
        result_df = pd.concat([ohlcv, component_df], axis=1)
        
        return result_df, pca, scaler
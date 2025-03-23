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
    
    @profile
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from raw OHLCV data with caching.
        
        Args:
            df: DataFrame with OHLCV data
                
        Returns:
            DataFrame with generated features
        """
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a cache key based on data shape and first/last timestamps
        first_date = str(df.index[0]).replace(' ', '_').replace(':', '-')
        last_date = str(df.index[-1]).replace(' ', '_').replace(':', '-')
        symbol = df.get('symbol', 'unknown') if 'symbol' in df else 'unknown'
        cache_key = f"{symbol}_{len(df)}_{first_date}_{last_date}"
        
        # Define cache file path
        cache_file = os.path.join(cache_dir, f"features_{cache_key}.pkl")
        
        # Check if cache exists
        if os.path.exists(cache_file):
            try:
                # Attempt to load from cache
                cached_df = pd.read_pickle(cache_file)
                self.logger.info(f"Loaded features from cache: {cache_file}")
                return cached_df
            except Exception as e:
                self.logger.warning(f"Error loading cache: {e}. Recalculating features.")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create progress tracker
        progress = ProgressTracker(
            name=f"Feature Generation", 
            total_steps=len(self.indicators_config),
            log_interval=1,
            logger=self.logger
        )

        # Generate features based on configuration
        for indicator_config in self.indicators_config:
            indicator_name = indicator_config['name']
            params = indicator_config.get('params', {})
            
            try:
                method = getattr(self, f"_generate_{indicator_name}")
                result_df = method(result_df, **params)
                progress.update(1, indicator_name)
            except AttributeError:
                self.logger.warning(f"Indicator method not found: _generate_{indicator_name}")
            except Exception as e:
                self.logger.error(f"Error generating {indicator_name}: {e}")
        
        # Mark as complete
        progress.complete()

        # Drop NaN values
        result_df.dropna(inplace=True)
        
        # Save to cache
        try:
            result_df.to_pickle(cache_file)
            self.logger.info(f"Saved features to cache: {cache_file}")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
        
        return result_df
    
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
    def _generate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Generate RSI indicator."""
        df[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
        return df
    
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
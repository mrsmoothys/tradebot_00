"""
Utility functions for market data analysis and calculations.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for a given price dataframe.
    
    Args:
        data: DataFrame with OHLC price data
        period: Period for ATR calculation (default: 14)
        
    Returns:
        Series containing ATR values
    """
    # Calculate True Range
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    # Handle first row where previous close isn't available
    close.iloc[0] = (high.iloc[0] + low.iloc[0]) / 2
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using exponential moving average
    atr = true_range.ewm(span=period, adjust=False).mean()
    
    return atr
import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import time
import random
import string

def generate_id(prefix: str = '', length: int = 8) -> str:
    """
    Generate a random ID string.
    
    Args:
        prefix: ID prefix
        length: ID length
        
    Returns:
        Random ID string
    """
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    random_str = ''.join(random.choice(chars) for _ in range(length))
    
    if prefix:
        return f"{prefix}_{random_str}"
    
    return random_str

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """
    Convert timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp in seconds or milliseconds
        
    Returns:
        Datetime object
    """
    # Check if timestamp is in milliseconds
    if timestamp > 1e11:
        return datetime.fromtimestamp(timestamp / 1000.0)
    
    return datetime.fromtimestamp(timestamp)

def datetime_to_timestamp(dt: datetime, milliseconds: bool = True) -> Union[int, float]:
    """
    Convert datetime to timestamp.
    
    Args:
        dt: Datetime object
        milliseconds: Whether to return timestamp in milliseconds
        
    Returns:
        Unix timestamp
    """
    timestamp = dt.timestamp()
    
    if milliseconds:
        return int(timestamp * 1000)
    
    return int(timestamp)

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str, pretty: bool = True) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        pretty: Whether to format JSON with indentation
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)

def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary from YAML
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], file_path: str) -> None:
    """
    Save a dictionary to a YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save YAML file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def resample_dataframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample a DataFrame to a different timeframe.
    
    Args:
        df: DataFrame with OHLCV data and datetime index
        timeframe: Target timeframe (e.g., '1h', '4h', '1d')
        
    Returns:
        Resampled DataFrame
    """
    # Check that DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")
    
    # Map common timeframe strings to pandas offset strings
    timeframe_map = {
        '1m': '1Min',
        '3m': '3Min',
        '5m': '5Min',
        '15m': '15Min',
        '30m': '30Min',
        '1h': '1H',
        '2h': '2H',
        '4h': '4H',
        '6h': '6H',
        '8h': '8H',
        '12h': '12H',
        '1d': '1D',
        '3d': '3D',
        '1w': '1W',
        '1M': '1M'
    }
    
    # Convert timeframe to pandas offset
    if timeframe in timeframe_map:
        offset = timeframe_map[timeframe]
    else:
        # Try to use the timeframe directly
        offset = timeframe
    
    # Define aggregation functions
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Filter only OHLCV columns that are present
    agg_dict = {col: ohlc_dict[col] for col in ohlc_dict if col in df.columns}
    
    # Resample and aggregate
    resampled = df.resample(offset).agg(agg_dict)
    
    # Drop rows with NaN values
    return resampled.dropna()

def calculate_returns(prices: Union[List[float], np.ndarray, pd.Series], 
                    log_returns: bool = False) -> np.ndarray:
    """
    Calculate returns from a series of prices.
    
    Args:
        prices: Series of prices
        log_returns: Whether to calculate log returns
        
    Returns:
        Array of returns
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    elif isinstance(prices, pd.Series):
        prices = prices.values
    
    if log_returns:
        # Calculate log returns: ln(P_t / P_{t-1})
        returns = np.diff(np.log(prices))
    else:
        # Calculate simple returns: (P_t / P_{t-1}) - 1
        returns = (prices[1:] / prices[:-1]) - 1
    
    return returns

def calculate_drawdowns(equity_curve: Union[List[float], np.ndarray, pd.Series]) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdowns from an equity curve.
    
    Args:
        equity_curve: Equity curve (series of equity values)
        
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_duration)
    """
    if isinstance(equity_curve, list):
        equity_curve = np.array(equity_curve)
    elif isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown percentage
    drawdown = (equity_curve - running_max) / running_max * 100
    
    # Calculate drawdown duration
    is_drawdown = drawdown < 0
    durations = []
    current_duration = 0
    max_duration = 0
    
    for i in range(len(is_drawdown)):
        if is_drawdown[i]:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return drawdown, np.min(drawdown), max_duration

def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray, pd.Series], 
                        risk_free_rate: float = 0.0, 
                        periods_per_year: int = 252) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values
    
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate Sharpe ratio
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    return sharpe

def calculate_sortino_ratio(returns: Union[List[float], np.ndarray, pd.Series], 
                         risk_free_rate: float = 0.0, 
                         periods_per_year: int = 252) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annualized risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns = returns.values
    
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside returns
    downside_returns = excess_returns[excess_returns < 0]
    
    # Calculate Sortino ratio
    if len(excess_returns) == 0 or len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
    
    return sortino

def parallel_process(func, items, n_jobs: int = -1, timeout: Optional[float] = None, 
                   show_progress: bool = True) -> List[Any]:
    """
    Process items in parallel using multiprocessing.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        n_jobs: Number of processes to use (-1 for all cores)
        timeout: Timeout in seconds (optional)
        show_progress: Whether to show a progress bar
        
    Returns:
        List of results
    """
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    
    # Determine number of processes
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    # Create process pool
    with Pool(n_jobs) as pool:
        # Create parallel tasks
        tasks = [pool.apply_async(func, (item,)) for item in items]
        
        # Get results with optional progress bar
        if show_progress:
            results = []
            for task in tqdm(tasks, total=len(items)):
                results.append(task.get(timeout=timeout))
        else:
            results = [task.get(timeout=timeout) for task in tasks]
    
    return results

def retry_operation(func, max_attempts: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)) -> Any:
    """
    Retry an operation with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Factor to increase delay by after each attempt
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Result of the function
    """
    attempt = 0
    current_delay = delay
    
    while attempt < max_attempts:
        try:
            return func()
        except exceptions as e:
            attempt += 1
            
            if attempt >= max_attempts:
                raise e
            
            # Sleep with exponential backoff
            time.sleep(current_delay)
            current_delay *= backoff_factor

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, how: str = 'inner') -> pd.DataFrame:
    """
    Merge two DataFrames with datetime index.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        how: How to merge ('inner', 'outer', 'left', 'right')
        
    Returns:
        Merged DataFrame
    """
    # Check that DataFrames have datetime index
    if not isinstance(df1.index, pd.DatetimeIndex) or not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("Both DataFrames must have DatetimeIndex for merging")
    
    # Merge DataFrames
    return pd.merge(df1, df2, left_index=True, right_index=True, how=how, suffixes=('', '_y'))

def rolling_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create a rolling window view of data for time series feature creation.
    
    Args:
        data: Input data array (1D)
        window_size: Size of the rolling window
        
    Returns:
        2D array with rolling windows
    """
    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)
    strides = data.strides + (data.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
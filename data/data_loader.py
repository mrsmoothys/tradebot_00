import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from utils.profiling import profile


class DataLoader:
    """
    Handles loading data from various sources and initial preprocessing.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the data loader.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.data_dir = self.config.get('data.directory')
    
    @profile
    def load_data(self, symbol: str, timeframe: str, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with the loaded data
        """
        # Determine file pattern
        file_pattern = f"{symbol}_{timeframe}_data_*.csv"
        
        # Find matching files
        matching_files = []
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{symbol}_{timeframe}_data_") and filename.endswith(".csv"):
                matching_files.append(filename)
        
        if not matching_files:
            raise FileNotFoundError(f"No data files found for {symbol} {timeframe}")
        
        # Load the most recent file if multiple matches
        filename = sorted(matching_files)[-1]
        file_path = os.path.join(self.data_dir, filename)
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Timestamp already exists, just convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'Open time' in df.columns:
             # First convert column to pd.Series to avoid type issues
            open_time_series = pd.Series(df['Open time'])
            
            # Check if values are numeric (timestamps in milliseconds)
            try:
                if pd.to_numeric(open_time_series, errors='coerce').notna().all():
                    # It's a numeric timestamp in milliseconds
                    df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
                else:
                    # It's already a datetime string
                    df['timestamp'] = pd.to_datetime(df['Open time'])
            except:
                # Fall back to treating as string datetime
                df['timestamp'] = pd.to_datetime(df['Open time'])
            
            df.set_index('timestamp', inplace=True)

        
        # Rename columns if needed
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Open time': 'open_time',
            'Close time': 'close_time',
            'Quote asset volume': 'quote_volume',
            'Number of trades': 'trades',
            'Taker buy base asset volume': 'taker_buy_base',
            'Taker buy quote asset volume': 'taker_buy_quote'
        }
        
        df.rename(columns={col: col_name for col, col_name in column_mapping.items() 
                           if col in df.columns}, inplace=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        self.logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
        return df
    
    def load_multi_timeframe_data(self, symbol: str, timeframes: List[str], 
                                 start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Dictionary of DataFrames by timeframe
        """
        result = {}
        for timeframe in timeframes:
            try:
                df = self.load_data(symbol, timeframe, start_date, end_date)
                result[timeframe] = df
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol} {timeframe}: {e}")
        
        return result
    
    def load_multi_symbol_data(self, symbols: List[str], timeframe: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbols
            timeframe: Trading timeframe
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Dictionary of DataFrames by symbol
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.load_data(symbol, timeframe, start_date, end_date)
                result[symbol] = df
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol} {timeframe}: {e}")
        
        return result
    
    def get_available_symbols(self) -> List[str]:
        """
        Get a list of available symbols.
        
        Returns:
            List of available symbols
        """
        symbols = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    symbols.add(parts[0])
        
        return sorted(list(symbols))
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """
        Get a list of available timeframes for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of available timeframes
        """
        timeframes = set()
        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{symbol}_") and filename.endswith(".csv"):
                parts = filename.split("_")
                if len(parts) >= 2:
                    timeframes.add(parts[1])
        
        return sorted(list(timeframes))
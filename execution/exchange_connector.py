import logging
import time
import hmac
import hashlib
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from requests.exceptions import RequestException

class ExchangeConnector:
    """
    Base class for exchange API connections.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the exchange connector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.exchange_config = self.config.get('exchange', {})
        
        # Common parameters
        self.exchange_id = self.exchange_config.get('id', 'binance')
        self.api_key = self.exchange_config.get('api_key', '')
        self.api_secret = self.exchange_config.get('api_secret', '')
        self.testnet = self.exchange_config.get('testnet', True)
        
        # Rate limiting params
        self.rate_limit = self.exchange_config.get('rate_limit', 10)  # Requests per second
        self.last_request_time = 0
        
        # Market data cache
        self.tickers_cache = {}
        self.tickers_cache_time = 0
        self.tickers_cache_ttl = 5  # 5 seconds
        
        # Connection status
        self.is_connected = False
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle API errors.
        
        Args:
            error: Exception that occurred
        """
        self.logger.error(f"Exchange API error: {error}")
        
        # Check if connection related and update status
        if isinstance(error, (RequestException, ConnectionError, TimeoutError)):
            self.is_connected = False
    
    def connect(self) -> bool:
        """
        Connect to the exchange API and verify credentials.
        
        Returns:
            True if connection and authentication successful, False otherwise
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dictionary of exchange information
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker data
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker data for all symbols.
        
        Returns:
            Dictionary of ticker data by symbol
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to retrieve
            
        Returns:
            Dictionary with order book data
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_time: Optional[Union[str, datetime]] = None,
                          end_time: Optional[Union[str, datetime]] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with historical data
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances.
        
        Returns:
            Dictionary of currency balances
        """
        raise NotImplementedError("Implement in derived class")
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        raise NotImplementedError("Implement in derived class")
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                  price: Optional[float] = None, time_in_force: str = 'GTC',
                  stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            order_type: Order type ('LIMIT', 'MARKET', 'STOP_LOSS', etc.)
            quantity: Order quantity
            price: Order price (optional, required for LIMIT orders)
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            stop_price: Stop price (optional, required for STOP orders)
            
        Returns:
            Dictionary with order details
        """
        raise NotImplementedError("Implement in derived class")
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Dictionary with cancellation details
        """
        raise NotImplementedError("Implement in derived class")
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of cancellation details
        """
        raise NotImplementedError("Implement in derived class")


class BinanceConnector(ExchangeConnector):
    """
    Binance exchange API connector.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the Binance connector.
        
        Args:
            config_manager: Configuration manager instance
        """
        super().__init__(config_manager)
        
        # Set API URLs based on testnet/mainnet
        if self.testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.logger.info("Using Binance testnet API")
        else:
            self.base_url = "https://api.binance.com/api"
            self.logger.info("Using Binance mainnet API")
        
        # Timeframe mapping
        self.timeframe_map = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d',
            '3d': '3d',
            '1w': '1w',
            '1M': '1M'
        }
    
    def _get_signature(self, query_string: str) -> str:
        """
        Generate HMAC signature for authenticated requests.
        
        Args:
            query_string: Query string to sign
            
        Returns:
            HMAC signature
        """
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _send_public_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Send public API request.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response
        """
        self._enforce_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params)
            elif method == 'POST':
                response = requests.post(url, data=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self._handle_error(e)
            raise
    
    def _send_private_request(self, method: str, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Send private API request with authentication.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response
        """
        self._enforce_rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        # Default parameters
        params = params or {}
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Generate query string and signature
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = self._get_signature(query_string)
        
        # Add signature to parameters
        params['signature'] = signature
        
        # Set headers
        headers = {'X-MBX-APIKEY': self.api_key}
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = requests.post(url, data=params, headers=headers)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self._handle_error(e)
            raise
    
    def connect(self) -> bool:
        """
        Connect to the Binance API and verify credentials.
        
        Returns:
            True if connection and authentication successful, False otherwise
        """
        try:
            # Test public API
            self._send_public_request('GET', 'v3/ping')
            
            # Test private API if credentials are provided
            if self.api_key and self.api_secret:
                self._send_private_request('GET', 'v3/account')
            
            self.is_connected = True
            self.logger.info("Successfully connected to Binance API")
            return True
        
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"Failed to connect to Binance API: {e}")
            return False
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get exchange information.
        
        Returns:
            Dictionary of exchange information
        """
        return self._send_public_request('GET', 'v3/exchangeInfo')
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with ticker data
        """
        return self._send_public_request('GET', 'v3/ticker/24hr', {'symbol': symbol})
    
    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker data for all symbols.
        
        Returns:
            Dictionary of ticker data by symbol
        """
        current_time = time.time()
        
        # Return cached data if available and not expired
        if self.tickers_cache and current_time - self.tickers_cache_time < self.tickers_cache_ttl:
            return self.tickers_cache
        
        # Fetch new data
        response = self._send_public_request('GET', 'v3/ticker/24hr')
        
        # Process response
        tickers = {}
        for ticker in response:
            symbol = ticker['symbol']
            tickers[symbol] = ticker
        
        # Update cache
        self.tickers_cache = tickers
        self.tickers_cache_time = current_time
        
        return tickers
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels to retrieve
            
        Returns:
            Dictionary with order book data
        """
        return self._send_public_request('GET', 'v3/depth', {'symbol': symbol, 'limit': limit})
    
    def get_historical_data(self, symbol: str, timeframe: str,
                          start_time: Optional[Union[str, datetime]] = None,
                          end_time: Optional[Union[str, datetime]] = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with historical data
        """
        # Check if timeframe is supported
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        binance_interval = self.timeframe_map[timeframe]
        
        # Prepare parameters
        params = {
            'symbol': symbol,
            'interval': binance_interval,
            'limit': limit
        }
        
        # Add start_time if provided
        if start_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            
            params['startTime'] = int(start_time.timestamp() * 1000)
        
        # Add end_time if provided
        if end_time:
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        # Send request
        response = self._send_public_request('GET', 'v3/klines', params)
        
        # Convert to DataFrame
        df = pd.DataFrame(response, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        # Convert types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                         'Quote asset volume', 'Number of trades',
                         'Taker buy base asset volume', 'Taker buy quote asset volume']
        
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column])
        
        # Convert timestamps to datetime
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        # Set index
        df.set_index('Open time', inplace=True)
        
        # Rename columns to match system conventions
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Close time': 'close_time',
            'Quote asset volume': 'quote_volume',
            'Number of trades': 'trades',
            'Taker buy base asset volume': 'taker_buy_base',
            'Taker buy quote asset volume': 'taker_buy_quote'
        }, inplace=True)
        
        # Drop unnecessary columns
        df.drop(columns=['Ignore'], inplace=True, errors='ignore')
        
        return df
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balances.
        
        Returns:
            Dictionary of currency balances
        """
        response = self._send_private_request('GET', 'v3/account')
        
        balances = {}
        for asset in response['balances']:
            symbol = asset['asset']
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            
            if total > 0:
                balances[symbol] = {
                    'free': free,
                    'locked': locked,
                    'total': total
                }
        
        return balances
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._send_private_request('GET', 'v3/openOrders', params)
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                  price: Optional[float] = None, time_in_force: str = 'GTC',
                  stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            order_type: Order type ('LIMIT', 'MARKET', 'STOP_LOSS', etc.)
            quantity: Order quantity
            price: Order price (optional, required for LIMIT orders)
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            stop_price: Stop price (optional, required for STOP orders)
            
        Returns:
            Dictionary with order details
        """
        # Prepare parameters
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity
        }
        
        # Add price for LIMIT orders
        if order_type.upper() == 'LIMIT':
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        # Add stop price for STOP orders
        if order_type.upper() in ['STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']:
            if stop_price is None:
                raise ValueError("Stop price is required for STOP orders")
            
            params['stopPrice'] = stop_price
        
        # Send request
        return self._send_private_request('POST', 'v3/order', params)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Dictionary with cancellation details
        """
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        return self._send_private_request('DELETE', 'v3/order', params)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Trading symbol (optional)
            
        Returns:
            List of cancellation details
        """
        cancelled_orders = []
        
        if symbol:
            # Cancel all orders for a specific symbol
            params = {'symbol': symbol}
            response = self._send_private_request('DELETE', 'v3/openOrders', params)
            cancelled_orders.extend(response if isinstance(response, list) else [response])
        else:
            # Cancel all orders for all symbols
            open_orders = self.get_open_orders()
            symbols = set(order['symbol'] for order in open_orders)
            
            for sym in symbols:
                try:
                    params = {'symbol': sym}
                    response = self._send_private_request('DELETE', 'v3/openOrders', params)
                    cancelled_orders.extend(response if isinstance(response, list) else [response])
                except Exception as e:
                    self.logger.error(f"Error cancelling orders for {sym}: {e}")
        
        return cancelled_orders
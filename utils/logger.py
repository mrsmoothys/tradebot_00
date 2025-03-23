import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_level: int = logging.INFO, 
                log_file: Optional[str] = None, console: bool = True) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        log_level: Logging level
        log_file: Path to log file (optional)
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file provided
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def setup_trading_logger(config_manager) -> logging.Logger:
    """
    Set up a logger for the trading bot using configuration.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Configured logger instance
    """
    # Get logging configuration
    log_config = config_manager.get('logging', {})
    
    # Set up logging parameters
    log_level_str = log_config.get('level', 'INFO')
    log_level = getattr(logging, log_level_str.upper())
    
    log_to_file = log_config.get('file', True)
    log_to_console = log_config.get('console', True)
    
    log_dir = log_config.get('directory', 'logs')
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"trading_{timestamp}.log") if log_to_file else None
    
    # Set up logger
    return setup_logger('trading', log_level, log_file, log_to_console)

class TradeLogger:
    """
    Specialized logger for trading activity with additional methods for trade events.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the trade logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_trade_entry(self, symbol: str, side: str, price: float, quantity: float, 
                      order_id: str = None, reason: str = None) -> None:
        """
        Log a trade entry.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            price: Entry price
            quantity: Trade quantity
            order_id: Order ID (optional)
            reason: Entry reason (optional)
        """
        message = f"ENTRY: {side} {quantity} {symbol} @ {price}"
        
        if order_id:
            message += f" (ID: {order_id})"
        
        if reason:
            message += f" - Reason: {reason}"
        
        self.logger.info(message)
    
    def log_trade_exit(self, symbol: str, side: str, price: float, quantity: float,
                     profit_pct: float, profit_amount: float, order_id: str = None,
                     reason: str = None) -> None:
        """
        Log a trade exit.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            price: Exit price
            quantity: Trade quantity
            profit_pct: Profit percentage
            profit_amount: Profit amount
            order_id: Order ID (optional)
            reason: Exit reason (optional)
        """
        message = f"EXIT: {side} {quantity} {symbol} @ {price} - P/L: {profit_pct:.2f}% (${profit_amount:.2f})"
        
        if order_id:
            message += f" (ID: {order_id})"
        
        if reason:
            message += f" - Reason: {reason}"
        
        # Use different log levels based on profit/loss
        if profit_pct > 0:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_signal(self, symbol: str, timeframe: str, signal: int, price: float, 
                confidence: float = None, prediction: float = None) -> None:
        """
        Log a trading signal.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            signal: Signal value
            price: Current price
            confidence: Signal confidence (optional)
            prediction: Model prediction (optional)
        """
        signal_type = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
        signal_strength = abs(signal)
        
        message = f"SIGNAL: {signal_type} ({signal_strength}) for {symbol} {timeframe} @ {price}"
        
        if confidence is not None:
            message += f" - Confidence: {confidence:.2f}"
        
        if prediction is not None:
            message += f" - Prediction: {prediction:.4f}"
        
        self.logger.info(message)
    
    def log_error(self, error_type: str, message: str, exception: Exception = None) -> None:
        """
        Log an error.
        
        Args:
            error_type: Type of error
            message: Error message
            exception: Exception (optional)
        """
        error_message = f"ERROR ({error_type}): {message}"
        
        if exception:
            error_message += f" - Exception: {str(exception)}"
        
        self.logger.error(error_message)
    
    def log_performance(self, symbol: str, timeframe: str, metrics: dict) -> None:
        """
        Log performance metrics.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            metrics: Dictionary of performance metrics
        """
        message = f"PERFORMANCE: {symbol} {timeframe} - "
        
        # Add key metrics
        key_metrics = [
            f"Return: {metrics.get('total_return', 0):.2f}%",
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
            f"Drawdown: {metrics.get('max_drawdown', 0):.2f}%",
            f"Win Rate: {metrics.get('win_rate', 0):.2f}%"
        ]
        
        message += ", ".join(key_metrics)
        
        self.logger.info(message)
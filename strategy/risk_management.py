import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

class RiskManager:
    """
    Manages risk for trading strategies, including position sizing and risk allocation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Base risk parameters
        self.risk_per_trade = self.config.get('risk_per_trade', 2.0) / 100  # Default 2% risk per trade
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 5.0) / 100  # Max 5% risk
        self.min_risk_per_trade = self.config.get('min_risk_per_trade', 1.0) / 100  # Min 1% risk
        
        # Advanced risk parameters
        self.max_correlated_risk = self.config.get('max_correlated_risk', 10.0) / 100  # Max 10% for correlated assets
        self.max_open_trades = self.config.get('max_open_trades', 5)  # Maximum number of open trades
        self.max_drawdown_exit = self.config.get('max_drawdown_exit', 15.0) / 100  # Exit all if drawdown exceeds 15%
        
        # Adaptive risk parameters
        self.adaptive_risk = self.config.get('adaptive_risk', True)  # Whether to use adaptive risk
        self.volatility_factor = self.config.get('volatility_factor', 1.0)  # Factor to adjust risk based on volatility
        self.performance_factor = self.config.get('performance_factor', 0.5)  # Factor to adjust risk based on performance
        
        # Keep track of recent trades and their outcomes for performance-based risk adjustment
        self.recent_trades = []
        self.max_recent_trades = 20  # Number of recent trades to consider
        
        self.logger.info("Initialized risk manager")
    
    def calculate_position_size(self, capital: float, entry_price: float, stop_loss: float, 
                              symbol: Optional[str] = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            capital: Available capital
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Symbol being traded (optional)
            
        Returns:
            Position size in quote currency
        """
        # Calculate risk amount in quote currency
        risk_percentage = self._get_adjusted_risk_percentage(symbol)
        risk_amount = capital * risk_percentage
        
        # Calculate risk per unit
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = stop_distance / entry_price
        
        # Calculate position size
        if stop_distance_pct > 0:
            position_size = risk_amount / stop_distance_pct
        else:
            # Fallback if stop distance is too small
            self.logger.warning(f"Stop distance ({stop_distance}) is too small, using minimum risk")
            position_size = risk_amount / 0.01  # Assume 1% risk
        
        return position_size
    
    def _get_adjusted_risk_percentage(self, symbol: Optional[str] = None) -> float:
        """
        Get risk percentage adjusted for market conditions and recent performance.
        
        Args:
            symbol: Symbol being traded (optional)
            
        Returns:
            Adjusted risk percentage
        """
        risk_percentage = self.risk_per_trade
        
        if not self.adaptive_risk:
            return risk_percentage
        
        # Adjust based on recent performance
        win_rate = self._calculate_win_rate()
        if win_rate is not None:
            # Increase risk if win rate is high, decrease if low
            performance_adjustment = (win_rate - 0.5) * self.performance_factor
            risk_percentage *= (1 + performance_adjustment)
        
        # Get volatility adjustment if symbol is provided
        if symbol:
            volatility_adjustment = self._get_volatility_adjustment(symbol)
            risk_percentage *= volatility_adjustment
        
        # Ensure risk stays within min/max bounds
        risk_percentage = max(self.min_risk_per_trade, min(self.max_risk_per_trade, risk_percentage))
        
        return risk_percentage
    
    def _calculate_win_rate(self) -> Optional[float]:
        """
        Calculate win rate from recent trades.
        
        Returns:
            Win rate as a fraction, or None if no recent trades
        """
        if not self.recent_trades:
            return None
        
        wins = sum(1 for trade in self.recent_trades if trade.get('profit_pct', 0) > 0)
        return wins / len(self.recent_trades)
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """
        Calculate adjustment factor based on recent volatility.
        Higher volatility = lower position size.
        
        Args:
            symbol: Symbol being traded
            
        Returns:
            Volatility adjustment factor
        """
        # Default to no adjustment
        adjustment = 1.0
        
        # In a real implementation, this would use recent ATR or other volatility metrics
        # relative to historical average for this symbol
        
        return adjustment
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a completed trade for performance tracking.
        
        Args:
            trade: Trade information dictionary
        """
        self.recent_trades.append(trade)
        
        # Keep only the most recent trades
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)  # Remove oldest trade
    
    def check_max_open_trades(self, current_open_trades: int) -> bool:
        """
        Check if we can open more trades based on max open trades limit.
        
        Args:
            current_open_trades: Current number of open trades
            
        Returns:
            True if we can open more trades, False otherwise
        """
        return current_open_trades < self.max_open_trades
    
    def check_correlated_risk(self, symbol: str, open_positions: Dict[str, Any], 
                            correlation_matrix: Dict[str, Dict[str, float]]) -> bool:
        """
        Check if opening a position would exceed correlated risk limits.
        
        Args:
            symbol: Symbol to check
            open_positions: Dictionary of open positions by symbol
            correlation_matrix: Matrix of correlations between symbols
            
        Returns:
            True if within risk limits, False otherwise
        """
        # Skip if no correlation data
        if not correlation_matrix or symbol not in correlation_matrix:
            return True
        
        # Calculate correlated risk
        correlated_risk = 0.0
        
        for pos_symbol, position in open_positions.items():
            if pos_symbol in correlation_matrix[symbol]:
                correlation = abs(correlation_matrix[symbol][pos_symbol])
                if correlation > 0.7:  # Only consider highly correlated assets
                    # Add the risk exposure of this correlated position
                    position_risk = position.get('risk_pct', self.risk_per_trade)
                    correlated_risk += position_risk * correlation
        
        # Add the risk of the new position
        total_correlated_risk = correlated_risk + self.risk_per_trade
        
        # Check if within limits
        return total_correlated_risk <= self.max_correlated_risk
    
    def should_reduce_risk(self, current_drawdown: float) -> bool:
        """
        Check if we should reduce risk based on current drawdown.
        
        Args:
            current_drawdown: Current drawdown as a percentage
            
        Returns:
            True if we should reduce risk, False otherwise
        """
        return current_drawdown >= self.max_drawdown_exit
    
    def get_risk_adjusted_sl_tp(self, atr: float, entry_price: float, side: str) -> Tuple[float, float]:
        """
        Get risk-adjusted stop loss and take profit levels based on ATR.
        
        Args:
            atr: Average True Range value
            entry_price: Entry price
            side: 'long' or 'short'
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Get ATR multipliers
        sl_atr_multiplier = self.config.get('sl_atr_multiplier', 2.0)
        min_risk_reward = self.config.get('min_risk_reward', 1.5)
        
        # Calculate stop distance based on ATR
        stop_distance = atr * sl_atr_multiplier
        
        # Calculate SL and TP levels
        if side == 'long':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * min_risk_reward)
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * min_risk_reward)
        
        return stop_loss, take_profit
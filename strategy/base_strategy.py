import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from datetime import datetime

class Position(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    FLAT = 0

class Order:
    """Class representing a trading order."""
    
    def __init__(
        self,
        symbol: str,
        order_type: str,
        side: str,
        price: float,
        quantity: float,
        timestamp: pd.Timestamp,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        trailing_stop: bool = False,
        trailing_distance: Optional[float] = None,
        order_id: Optional[str] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy, sell)
            price: Order price
            quantity: Order quantity
            timestamp: Order timestamp
            sl_price: Stop loss price
            tp_price: Take profit price
            trailing_stop: Whether to use trailing stop
            trailing_distance: Distance for trailing stop
            order_id: Unique order ID
        """
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.trailing_stop = trailing_stop
        self.trailing_distance = trailing_distance
        self.order_id = order_id or f"{symbol}_{side}_{timestamp.strftime('%Y%m%d%H%M%S')}_{id(self)}"
        
        # Execution details
        self.executed_price = None
        self.executed_quantity = None
        self.executed_timestamp = None
        self.status = "CREATED"
        
        # Profit details
        self.exit_price = None
        self.exit_timestamp = None
        self.profit_amount = None
        self.profit_percentage = None
        self.exit_reason = None
    
    def execute(self, price: float, timestamp: pd.Timestamp, slippage: float = 0.002) -> float:
        """
        Execute the order with slippage.
        
        Args:
            price: Execution price
            timestamp: Execution timestamp
            slippage: Slippage percentage
            
        Returns:
            Executed price
        """
        # Apply slippage
        if self.side == "buy":
            execution_price = price * (1 + slippage)
        else:  # sell
            execution_price = price * (1 - slippage)
        
        self.executed_price = execution_price
        self.executed_quantity = self.quantity
        self.executed_timestamp = timestamp
        self.status = "FILLED"
        
        return execution_price
    
    def update_trailing_stop(self, current_price: float) -> float:
        """
        Update trailing stop price based on current price.
        
        Args:
            current_price: Current market price
            
        Returns:
            Updated stop loss price
        """
        if not self.trailing_stop or self.trailing_distance is None:
            return self.sl_price
        
        if self.side == "buy":  # Long position
            # Trail stop loss higher as price increases
            new_sl = current_price - self.trailing_distance
            if self.sl_price is None or new_sl > self.sl_price:
                self.sl_price = new_sl
        else:  # Short position
            # Trail stop loss lower as price decreases
            new_sl = current_price + self.trailing_distance
            if self.sl_price is None or new_sl < self.sl_price:
                self.sl_price = new_sl
        
        return self.sl_price
    
    def close(self, price: float, timestamp: pd.Timestamp, reason: str, slippage: float = 0.002) -> Tuple[float, float]:
        """
        Close the order with slippage.
        
        Args:
            price: Close price
            timestamp: Close timestamp
            reason: Reason for closing (TP, SL, signal, etc.)
            slippage: Slippage percentage
            
        Returns:
            Tuple of (profit_amount, profit_percentage)
        """
        # Apply slippage
        if self.side == "buy":  # For a long position, we sell to close
            exit_price = price * (1 - slippage)
        else:  # For a short position, we buy to close
            exit_price = price * (1 + slippage)
        
        self.exit_price = exit_price
        self.exit_timestamp = timestamp
        self.exit_reason = reason
        self.status = "CLOSED"
        
        # Calculate profit
        if self.side == "buy":
            self.profit_amount = (exit_price - self.executed_price) * self.executed_quantity
            self.profit_percentage = (exit_price / self.executed_price - 1) * 100
        else:
            self.profit_amount = (self.executed_price - exit_price) * self.executed_quantity
            self.profit_percentage = (self.executed_price / exit_price - 1) * 100
        
        return self.profit_amount, self.profit_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "order_type": self.order_type,
            "side": self.side,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "order_id": self.order_id,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "trailing_stop": self.trailing_stop,
            "trailing_distance": self.trailing_distance,
            "executed_price": self.executed_price,
            "executed_quantity": self.executed_quantity,
            "executed_timestamp": self.executed_timestamp.isoformat() if self.executed_timestamp else None,
            "status": self.status,
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "profit_amount": self.profit_amount,
            "profit_percentage": self.profit_percentage,
            "exit_reason": self.exit_reason
        }

class BaseStrategy:
    """Base class for all trading strategies."""
    
    def __init__(self, config_manager):
        """
        Initialize the strategy.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        
        # Basic parameters
        self.symbol = None
        self.timeframe = None
        self.initial_capital = self.config.get('backtest.initial_capital', 10000.0)
        self.capital = self.initial_capital
        self.position_size = self.config.get('strategy.risk_per_trade', 2.0) / 100.0  # Convert to fraction
        self.trading_fee = self.config.get('backtest.trading_fee', 0.0002)
        self.slippage = self.config.get('backtest.slippage', 0.002)
        
        # Risk management
        self.adaptive_sl_tp = self.config.get('strategy.adaptive_sl_tp', True)
        self.trailing_stop = self.config.get('strategy.trailing_stop', True)
        self.partial_profit_taking = self.config.get('strategy.partial_profit_taking', True)
        
        # Fixed SL/TP percentages
        self.fixed_sl_percentage = self.config.get('strategy.stop_loss.fixed_percentage', 3.0) / 100.0
        self.fixed_tp_percentage = self.config.get('strategy.take_profit.fixed_percentage', 5.0) / 100.0
        
        # Adaptive SL/TP parameters
        self.atr_multiplier = self.config.get('strategy.stop_loss.atr_multiplier', 2.0)
        self.min_risk_reward_ratio = self.config.get('strategy.take_profit.min_risk_reward_ratio', 1.5)
        
        # Trailing stop parameters
        self.trailing_activation_percentage = self.config.get('strategy.trailing_stop.activation_percentage', 1.0) / 100.0
        self.trailing_offset_percentage = self.config.get('strategy.trailing_stop.offset_percentage', 1.0) / 100.0
        
        # Partial profit taking parameters
        self.partial_profit_levels = self.config.get('strategy.partial_profit.levels', [25, 50, 75])
        self.partial_profit_percentages = self.config.get('strategy.partial_profit.percentages', [25, 25, 25])
        
        # Position and order tracking
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        
        # Performance tracking
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Market data
        self.data = None
        self.current_index = None
        self.current_timestamp = None
        self.current_price = None
    
    def set_data(self, data: pd.DataFrame, symbol: str = None, timeframe: str = None) -> None:
        """
        Set market data for the strategy.
        
        Args:
            data: Market data as pandas DataFrame
            symbol: Trading symbol (optional)
            timeframe: Trading timeframe (optional)
        """
        self.data = data.copy()
        
        if symbol:
            self.symbol = symbol
        if timeframe:
            self.timeframe = timeframe
        
        # Reset state
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Initialize market state
        if len(data) > 0:
            self.current_index = 0
            self.current_timestamp = data.index[0]
            self.current_price = data['close'].iloc[0]
    
    def update_state(self, index: int) -> None:
        """
        Update the current market state.
        
        Args:
            index: Current data index
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of bounds for data of length {len(self.data)}")
        
        self.current_index = index
        self.current_timestamp = self.data.index[index]
        self.current_price = self.data['close'].iloc[index]
        
        # Update trailing stop if active
        if self.current_order and self.current_order.status == "FILLED" and self.current_order.trailing_stop:
            self.current_order.update_trailing_stop(self.current_price)
        
        # Check for SL/TP hits
        self._check_exit_conditions()
        
        # Update equity curve
        self._update_equity()
    
    def _update_equity(self) -> None:
        """Update equity curve, drawdowns, and returns."""
        if self.position == Position.FLAT:
            current_equity = self.capital
        else:
            # Calculate unrealized P&L
            if self.current_order and self.current_order.status == "FILLED":
                if self.current_order.side == "buy":
                    pnl = (self.current_price - self.current_order.executed_price) * self.current_order.executed_quantity
                else:
                    pnl = (self.current_order.executed_price - self.current_price) * self.current_order.executed_quantity
                
                current_equity = self.capital + pnl
            else:
                current_equity = self.capital
        
        # Update equity curve
        self.equity_curve.append(current_equity)
        
        # Calculate return
        prev_equity = self.equity_curve[-2] if len(self.equity_curve) > 1 else self.initial_capital
        if prev_equity > 0:
            ret = (current_equity / prev_equity) - 1
        else:
            ret = 0
        self.returns.append(ret)
        
        # Calculate drawdown
        max_equity = max(self.equity_curve)
        drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
        self.drawdowns.append(drawdown)
    
    def _check_exit_conditions(self) -> None:
        """Check for stop loss and take profit conditions."""
        if not self.current_order or self.current_order.status != "FILLED":
            return
        
        # Check stop loss
        if self.current_order.sl_price is not None:
            if (self.current_order.side == "buy" and self.current_price <= self.current_order.sl_price) or \
               (self.current_order.side == "sell" and self.current_price >= self.current_order.sl_price):
                self._close_position("SL")
                return
        
        # Check take profit
        if self.current_order.tp_price is not None:
            if (self.current_order.side == "buy" and self.current_price >= self.current_order.tp_price) or \
               (self.current_order.side == "sell" and self.current_price <= self.current_order.tp_price):
                self._close_position("TP")
                return
    
    def _calculate_sl_tp_levels(self, side: str, entry_price: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            side: Order side (buy or sell)
            entry_price: Entry price
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if not self.adaptive_sl_tp:
            # Fixed percentage
            if side == "buy":
                sl_price = entry_price * (1 - self.fixed_sl_percentage)
                tp_price = entry_price * (1 + self.fixed_tp_percentage)
            else:
                sl_price = entry_price * (1 + self.fixed_sl_percentage)
                tp_price = entry_price * (1 - self.fixed_tp_percentage)
            
            return sl_price, tp_price
        
        # Adaptive SL/TP based on market conditions
        
        # Get ATR for volatility-based stops
        atr_value = None
        # Try to find ATR column
        for col in self.data.columns:
            if 'atr' in col.lower():
                atr_value = self.data[col].iloc[self.current_index]
                break
        
        if atr_value and atr_value > 0:
            # ATR-based stops
            if side == "buy":
                sl_price = entry_price - atr_value * self.atr_multiplier
                tp_price = entry_price + atr_value * self.atr_multiplier * self.min_risk_reward_ratio
            else:
                sl_price = entry_price + atr_value * self.atr_multiplier
                tp_price = entry_price - atr_value * self.atr_multiplier * self.min_risk_reward_ratio
        else:
            # Fall back to fixed percentage
            if side == "buy":
                sl_price = entry_price * (1 - self.fixed_sl_percentage)
                tp_price = entry_price * (1 + self.fixed_tp_percentage)
            else:
                sl_price = entry_price * (1 + self.fixed_sl_percentage)
                tp_price = entry_price * (1 - self.fixed_tp_percentage)
        
        # Ensure minimum risk-reward ratio
        if side == "buy":
            risk = entry_price - sl_price
            reward = tp_price - entry_price
            
            if risk > 0 and reward / risk < self.min_risk_reward_ratio:
                tp_price = entry_price + risk * self.min_risk_reward_ratio
        else:
            risk = sl_price - entry_price
            reward = entry_price - tp_price
            
            if risk > 0 and reward / risk < self.min_risk_reward_ratio:
                tp_price = entry_price - risk * self.min_risk_reward_ratio
        
        return sl_price, tp_price
    
    def _calculate_trailing_distance(self, side: str, entry_price: float, sl_price: float) -> float:
        """
        Calculate trailing stop distance.
        
        Args:
            side: Order side (buy or sell)
            entry_price: Entry price
            sl_price: Initial stop loss price
            
        Returns:
            Trailing stop distance
        """
        if side == "buy":
            return entry_price - sl_price
        else:
            return sl_price - entry_price
    
    def _calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate position size based on risk percentage.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in units
        """
        # Risk amount in currency units
        risk_amount = self.capital * self.position_size
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        # Position size in units
        position_size = risk_amount / risk_per_unit
        
        return position_size
    
    def _open_position(self, position: Position, reason: str) -> Optional[Order]:
        """
        Open a new position.
        
        Args:
            position: Position to open (LONG or SHORT)
            reason: Reason for opening the position
            
        Returns:
            Newly created order
        """
        # Ensure we're not already in a position
        if self.position != Position.FLAT:
            self.logger.warning(f"Cannot open position, already in {self.position} position")
            return None
        
        # Set position
        self.position = position
        
        # Calculate order details
        side = "buy" if position == Position.LONG else "sell"
        price = self.current_price
        
        # Calculate stop loss and take profit levels
        sl_price, tp_price = self._calculate_sl_tp_levels(side, price)
        
        # Calculate position size based on risk
        quantity = self._calculate_position_size(price, sl_price)
        
        # Calculate trailing stop distance if enabled
        trailing_distance = None
        if self.trailing_stop:
            trailing_distance = self._calculate_trailing_distance(side, price, sl_price)
        
        # Create and execute order
        order = Order(
            symbol=self.symbol,
            order_type="market",
            side=side,
            price=price,
            quantity=quantity,
            timestamp=self.current_timestamp,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_stop=self.trailing_stop,
            trailing_distance=trailing_distance
        )
        
        # Execute order with slippage
        executed_price = order.execute(price, self.current_timestamp, self.slippage)
        
        # Apply trading fee
        fee = executed_price * quantity * self.trading_fee
        self.capital -= fee
        
        # Store order
        self.current_order = order
        self.orders.append(order)
        
        self.logger.info(f"Opened {side} position at {executed_price} with quantity {quantity}")
        self.logger.info(f"SL: {sl_price}, TP: {tp_price}, Fee: {fee}")
        
        return order
    
    def _close_position(self, reason: str) -> Optional[Tuple[float, float]]:
        """
        Close the current position.
        
        Args:
            reason: Reason for closing the position
            
        Returns:
            Tuple of (profit_amount, profit_percentage)
        """
        # Ensure we're in a position
        if self.position == Position.FLAT or not self.current_order:
            self.logger.warning("Cannot close position, no position open")
            return None
        
        # Close the order
        price = self.current_price
        profit_amount, profit_percentage = self.current_order.close(
            price, self.current_timestamp, reason, self.slippage
        )
        
        # Apply trading fee
        fee = price * self.current_order.executed_quantity * self.trading_fee
        
        # Update capital
        self.capital += profit_amount - fee
        
        # Store trade details
        trade = {
            "order_id": self.current_order.order_id,
            "symbol": self.symbol,
            "side": self.current_order.side,
            "entry_price": self.current_order.executed_price,
            "exit_price": self.current_order.exit_price,
            "quantity": self.current_order.executed_quantity,
            "entry_time": self.current_order.executed_timestamp,
            "exit_time": self.current_order.exit_timestamp,
            "profit_amount": profit_amount,
            "profit_percentage": profit_percentage,
            "exit_reason": reason,
            "fees": fee
        }
        self.trades.append(trade)
        
        self.logger.info(f"Closed position at {price} with {profit_percentage:.2f}% profit")
        self.logger.info(f"Reason: {reason}, Fee: {fee}")
        
        # Reset position
        self.position = Position.FLAT
        self.current_order = None
        
        return profit_amount, profit_percentage
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Returns:
            DataFrame with added signal column
        """
        # Implement in derived classes
        raise NotImplementedError("Implement in derived class")
    
    def backtest(self) -> Dict[str, Any]:
        """
        Run backtest on the loaded data.
        
        Returns:
            Dictionary with backtest results
        """
        if self.data is None:
            raise ValueError("No data set. Call set_data() first.")
        
        # Reset state
        self.position = Position.FLAT
        self.current_order = None
        self.orders = []
        self.trades = []
        self.capital = self.initial_capital
        self.equity_curve = [self.initial_capital]
        self.drawdowns = [0.0]
        self.returns = [0.0]
        
        # Generate signals
        signals = self.generate_signals()
        
        # Run backtest
        for i in range(len(signals)):
            # Update state for current bar
            self.update_state(i)
            
            # Get signal
            signal = signals['signal'].iloc[i]
            
            # Process signal
            if signal == 1 and self.position == Position.FLAT:
                # Buy signal
                self._open_position(Position.LONG, "SIGNAL")
            elif signal == -1 and self.position == Position.FLAT:
                # Sell signal
                self._open_position(Position.SHORT, "SIGNAL")
            elif signal == 0 and self.position != Position.FLAT:
                # Close position
                self._close_position("SIGNAL")
        
        # Close any open position at the end
        if self.position != Position.FLAT:
            self._close_position("END_OF_DATA")
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        # Prepare results
        results = {
            "performance": performance,
            "equity_curve": self.equity_curve,
            "trades": self.trades,
            "signals": signals
        }
        
        return results
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        initial_capital = self.initial_capital
        final_capital = self.equity_curve[-1] if self.equity_curve else initial_capital
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Trade metrics
        num_trades = len(self.trades)
        if num_trades == 0:
            return {
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "total_return": total_return,
                "num_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0
            }
        
        # Win/loss metrics
        winning_trades = [t for t in self.trades if t['profit_amount'] > 0]
        losing_trades = [t for t in self.trades if t['profit_amount'] <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        avg_profit = np.mean([t['profit_percentage'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_percentage'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum(t['profit_amount'] for t in winning_trades)
        total_loss = abs(sum(t['profit_amount'] for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Drawdown
        max_drawdown = max(self.drawdowns) * 100 if self.drawdowns else 0
        
        # Risk-adjusted returns
        returns_series = pd.Series(self.returns)
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        
        # Sortino ratio (downside risk only)
        downside_returns = returns_series[returns_series < 0]
        sortino_ratio = returns_series.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        return {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio
        }
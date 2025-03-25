"""
Machine learning-based trading strategy implementation.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from utils.profiling import profile
from strategy.base_strategy import BaseStrategy, Position, Order
from strategy.risk_management import RiskManager
from utils.market_utils import calculate_atr
from utils.progress import ProgressTracker



logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine learning-based trading strategy that implements signal generation
    and trade management using trained ML models.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model,
        lookback_window: int,
        prediction_horizon: int,
        config: Dict[str, Any],
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize the ML trading strategy.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            model: Trained machine learning model
            lookback_window: Lookback window for features
            prediction_horizon: Prediction horizon
            config: Strategy configuration parameters
            risk_manager: Risk manager instance for position sizing
        """

        # Set these properties after calling parent constructor
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Call parent constructor
        super().__init__(config)

        # Model parameters
        self.model = model
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        
        # Signal generation parameters
        self.threshold = config.get('threshold', 0.005)
        self.confidence_multiplier = config.get('confidence_multiplier', 1.5)
        
        # Trade management parameters
        self.use_adaptive_sl_tp = config.get('use_adaptive_sl_tp', True)
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 2.0)
        self.min_risk_reward_ratio = config.get('min_risk_reward_ratio', 1.5)
        self.trail_after_pct = config.get('trail_after_pct', 0.5)  # Start trailing after 0.5% profit
        
        # Multi-timeframe parameters
        self.use_multi_timeframe = config.get('use_multi_timeframe', True)
        self.higher_tf_weight = config.get('higher_tf_weight', 0.5)
        
        # Risk management
        self.risk_manager = risk_manager or RiskManager(config.get('risk', {}))
        
        # Feature transforms and preprocessing
        self.feature_columns = []
        self.use_pca = config.get('use_pca', True)
        
        # Performance tracking
        self.performance_metrics = {}
        
        logger.info(f"Initialized ML strategy for {symbol} {timeframe}")
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model prediction with appropriate normalization.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Normalized feature array for model input
        """
        # Check if symbol and timeframe are set
        if not hasattr(self, 'symbol') or not self.symbol or not hasattr(self, 'timeframe') or not self.timeframe:
            raise ValueError("Symbol and timeframe must be provided either as parameters or set as instance attributes")
        
        # Extract feature columns (exclude OHLCV and other special columns)
        feature_cols = [col for col in data.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'signal', 'prediction', 'confidence']]
        
        # Check model input shape vs available features
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'input_shape'):
            expected_feature_count = self.model.model.input_shape[0][2]  # Get feature dimension from model
            if len(feature_cols) != expected_feature_count:
                # Select only the first expected_feature_count features to match model input
                if len(feature_cols) > expected_feature_count:
                    feature_cols = feature_cols[:expected_feature_count]
        
        self.feature_columns = feature_cols
        
        # Get lookback window data
        features = data[feature_cols].values[-self.lookback_window:]
        
        # Apply normalization (z-score)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std = np.where(std == 0, 1e-8, std)  # Prevent division by zero
        normalized_features = (features - mean) / std
        
        # Reshape for model input
        model_input = np.expand_dims(normalized_features, axis=0)
        
        return model_input
    
    
    @profile
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using vectorized operations for optimal performance.
        
        Args:
            data: Market data DataFrame
                
        Returns:
            DataFrame with signal column added
        """
        # Initialization with pre-allocation
        signals = data.copy()
        n_samples = len(signals)
        
        if n_samples <= self.lookback_window:
            self.logger.warning(f"Insufficient data for prediction: {n_samples} samples, {self.lookback_window} required")
            return signals.assign(signal=0, prediction=0.0, confidence=0.0)
        
        # Extract feature columns once
        feature_cols = [col for col in signals.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'signal', 'prediction', 'confidence']]
        
        # Pre-allocate arrays for optimization
        valid_indices = range(self.lookback_window, n_samples)
        n_predictions = len(valid_indices)
        predictions_array = np.zeros(n_predictions)
        confidence_array = np.zeros(n_predictions)
        signals_array = np.zeros(n_predictions)
        
        # Create input windows efficiently using strided operations
        X = np.zeros((n_predictions, self.lookback_window, len(feature_cols)))
        for i, idx in enumerate(valid_indices):
            X[i] = signals.iloc[idx-self.lookback_window:idx][feature_cols].values
        
        # Batch prediction (already optimized)
        try:
            all_predictions = self.model.predict(X, self.symbol, self.timeframe, verbose=0)
            
            # Current prices for prediction normalization
            current_prices = signals['close'].iloc[self.lookback_window:].values
            
            # Vectorized prediction processing
            predictions = all_predictions[:, -1]  # Last value in horizon
            predicted_returns = (predictions / current_prices) - 1
            
            # Vector threshold operations
            threshold = 0.001  # Lowered for more signals
            confidence_values = np.abs(predicted_returns) / (threshold + 1e-8)
            
            # Generate signals vectorized
            buy_mask = predicted_returns > threshold
            sell_mask = predicted_returns < -threshold
            
            # Compute signal strength vectorized
            signals_array = np.zeros(n_predictions)
            signals_array[buy_mask] = np.minimum(3, 1 + confidence_values[buy_mask]/2)
            signals_array[sell_mask] = np.maximum(-3, -1 - confidence_values[sell_mask]/2)
            
            # Apply results efficiently
            idx_array = signals.index[self.lookback_window:]
            signals.loc[idx_array, 'prediction'] = predicted_returns
            signals.loc[idx_array, 'confidence'] = confidence_values
            signals.loc[idx_array, 'signal'] = signals_array.astype(int)
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        
        return signals
    
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
    
    def calculate_adaptive_levels(self, side: str, price: float, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate adaptive stop-loss and take-profit levels based on market conditions.
        
        Args:
            side: 'long' or 'short'
            price: Entry price
            data: Recent market data
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Default values based on percentage
        sl_pct = 0.03  # 3%
        tp_pct = 0.05  # 5%
        
        # Calculate adaptive levels if enabled
        if self.use_adaptive_sl_tp:
            # Get ATR for volatility-based stops
            if 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
                
                # ATR-based stop loss
                sl_distance = atr * self.atr_sl_multiplier
                
                # Risk-reward based take profit
                tp_distance = sl_distance * self.min_risk_reward_ratio
                
                # Apply based on side
                if side == 'long':
                    stop_loss = price - sl_distance
                    take_profit = price + tp_distance
                else:  # short
                    stop_loss = price + sl_distance
                    take_profit = price - tp_distance
                
                return stop_loss, take_profit
            
            # Use swing points if available
            if 'swing_high' in data.columns and 'swing_low' in data.columns:
                # Find recent swing points
                swing_high_idx = data.index[data['swing_high'] > 0]
                swing_low_idx = data.index[data['swing_low'] > 0]
                
                if len(swing_high_idx) > 0 and len(swing_low_idx) > 0:
                    recent_highs = data.loc[swing_high_idx[-3:], 'high'] if len(swing_high_idx) >= 3 else data.loc[swing_high_idx, 'high']
                    recent_lows = data.loc[swing_low_idx[-3:], 'low'] if len(swing_low_idx) >= 3 else data.loc[swing_low_idx, 'low']
                    
                    if side == 'long':
                        # Use nearest swing low below entry for stop loss
                        valid_lows = recent_lows[recent_lows < price]
                        if not valid_lows.empty:
                            stop_loss = valid_lows.max()
                        else:
                            stop_loss = price * (1 - sl_pct)
                        
                        # Use nearest swing high above entry for take profit
                        valid_highs = recent_highs[recent_highs > price]
                        if not valid_highs.empty:
                            take_profit = valid_highs.min()
                        else:
                            take_profit = price * (1 + tp_pct)
                    else:  # short
                        # Use nearest swing high above entry for stop loss
                        valid_highs = recent_highs[recent_highs > price]
                        if not valid_highs.empty:
                            stop_loss = valid_highs.min()
                        else:
                            stop_loss = price * (1 + sl_pct)
                        
                        # Use nearest swing low below entry for take profit
                        valid_lows = recent_lows[recent_lows < price]
                        if not valid_lows.empty:
                            take_profit = valid_lows.max()
                        else:
                            take_profit = price * (1 - tp_pct)
                    
                    return stop_loss, take_profit
        
        # Fallback to percentage-based levels
        if side == 'long':
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:  # short
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)
        
        return stop_loss, take_profit
    


    def _identify_market_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify key market structures for better trade setups.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with added market structure columns
        """
        df = data.copy()
        
        # Calculate swing points
        df['swing_high'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(2)) & 
            (df['high'] > df['high'].shift(-1)) & 
            (df['high'] > df['high'].shift(-2))
        ).astype(int)
        
        df['swing_low'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(2)) & 
            (df['low'] < df['low'].shift(-1)) & 
            (df['low'] < df['low'].shift(-2))
        ).astype(int)
        
        # Identify higher timeframe trend
        df['uptrend'] = df['close'].rolling(20).mean() > df['close'].rolling(50).mean()
        df['downtrend'] = df['close'].rolling(20).mean() < df['close'].rolling(50).mean()
        
        # Identify support/resistance levels
        high_points = df.loc[df['swing_high'] == 1, 'high']
        low_points = df.loc[df['swing_low'] == 1, 'low']
        
        # Check proximity to support/resistance levels
        def proximity_to_level(price, levels, threshold=0.02):
            """Check if price is close to any level."""
            for level in levels:
                if abs(price - level) / price < threshold:
                    return True
            return False
        
        # Add support/resistance proximity indicators
        df['near_resistance'] = df.apply(
            lambda row: proximity_to_level(row['close'], high_points, 0.02), axis=1
        ).astype(int)
        
        df['near_support'] = df.apply(
            lambda row: proximity_to_level(row['close'], low_points, 0.02), axis=1
        ).astype(int)
        
        return df
    
    def update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> float:
        """
        Update trailing stop based on current price and position information.
        
        Args:
            position: Position dictionary
            current_price: Current market price
            
        Returns:
            Updated stop loss price
        """
        if not self.use_trailing_stop:
            return position['stop_loss']
        
        if position['status'] != 'open':
            return position['stop_loss']
        
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate profit percentage
        if side == 'long':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        # Only trail if we have reached minimum profit threshold
        if profit_pct < self.trail_after_pct:
            return position['stop_loss']
        
        # Calculate trailing stop distance (ATR-based if available)
        if 'atr' in position:
            trail_distance = position['atr'] * (self.atr_sl_multiplier * 0.75)  # Tighter trail than initial SL
        else:
            # Use a percentage of current price
            trail_distance = current_price * 0.015  # 1.5%
        
        # Update stop loss based on side
        if side == 'long':
            new_stop = current_price - trail_distance
            # Only move stop loss up for longs
            if new_stop > position['stop_loss']:
                return new_stop
        else:  # short
            new_stop = current_price + trail_distance
            # Only move stop loss down for shorts
            if new_stop < position['stop_loss']:
                return new_stop
        
        return position['stop_loss']
    
    def should_close_position(self, 
                             position: Dict[str, Any], 
                             current_price: float, 
                             current_time: datetime,
                             data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if a position should be closed based on stop-loss, take-profit, 
        or signal conditions.
        
        Args:
            position: Position dictionary
            current_price: Current market price
            current_time: Current timestamp
            data: Recent market data with signals
            
        Returns:
            Tuple of (should_close, reason)
        """
        if position['status'] != 'open':
            return False, ''
        
        side = position['side']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Check stop loss
        if (side == 'long' and current_price <= stop_loss) or \
           (side == 'short' and current_price >= stop_loss):
            return True, 'stop_loss'
        
        # Check take profit
        if (side == 'long' and current_price >= take_profit) or \
           (side == 'short' and current_price <= take_profit):
            return True, 'take_profit'
        
        # Check signal reversal (if we have recent signals)
        try:
            # Get last signal
            recent_signals = data.loc[data.index <= current_time, 'signal'].tail(3)
            if not recent_signals.empty:
                last_signal = recent_signals.iloc[-1]
                
                # Close long if strong sell signal appears
                if side == 'long' and last_signal <= -2:
                    return True, 'signal_reversal'
                
                # Close short if strong buy signal appears
                if side == 'short' and last_signal >= 2:
                    return True, 'signal_reversal'
        except Exception as e:
            logger.warning(f"Error checking signal reversals: {e}")
        
        # Check time-based exit (if position held too long)
        if 'entry_time' in position:
            position_duration = (current_time - position['entry_time']).total_seconds() / 3600  # hours
            max_hold_time = 168  # 7 days
            
            if position_duration > max_hold_time:
                return True, 'time_exit'
        
        return False, ''
    
    def open_position(self, side: str, price: float, time: datetime, data: pd.DataFrame, reason: str = 'signal') -> Dict[str, Any]:
        """
        Open a new trading position with enhanced confirmation and risk management.
        
        Args:
            side: 'long' or 'short'
            price: Entry price
            time: Entry timestamp
            data: Recent market data
            reason: Reason for opening the position
            
        Returns:
            New position information
        """
        # Enhanced confirmation check
        confidence_threshold = 1.5  # Higher threshold for stronger confirmation
        
        # Get current data point
        current_data = data.iloc[-1]
        
        # Check for confluence of signals
        signal_strength = abs(current_data.get('signal', 0))
        prediction_value = abs(current_data.get('prediction', 0))
        confidence_value = current_data.get('confidence', 0)
        
        # Market structure confirmation
        structure_confirmation = 0
        
        if side == 'long':
            # For long positions, check bullish structure
            if current_data.get('uptrend', False):
                structure_confirmation += 1
            if current_data.get('near_support', 0) > 0:
                structure_confirmation += 1
            if current_data.get('swing_low', 0) > 0:
                structure_confirmation += 1
        else:  # short
            # For short positions, check bearish structure
            if current_data.get('downtrend', False):
                structure_confirmation += 1
            if current_data.get('near_resistance', 0) > 0:
                structure_confirmation += 1
            if current_data.get('swing_high', 0) > 0:
                structure_confirmation += 1
        
        # Combined confirmation score
        confirmation_score = signal_strength * 0.3 + confidence_value * 0.4 + structure_confirmation * 0.3
        
        # Adjust position size based on confirmation
        position_scale = min(1.0, confirmation_score / confidence_threshold)
        
        # Calculate stop loss and take profit levels
        stop_loss, take_profit = self.calculate_adaptive_levels(side, price, data)
        
        # Calculate base position size
        base_position_size = self._calculate_position_size(price, stop_loss)
        
        # Apply dynamic sizing based on confirmation strength
        position_size = base_position_size * position_scale
        
        # Create position record
        position = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'side': side,
            'entry_price': price,
            'entry_time': time,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'status': 'open',
            'reason': reason,
            'confirmation_score': confirmation_score,
            'position_scale': position_scale
        }
        
        # Add ATR if available for reference
        if 'atr' in data.columns:
            position['atr'] = data['atr'].iloc[-1]
        
        # Update account state
        if side == 'long':
            self.position = 1
        else:  # short
            self.position = -1
        
        # Record position
        position_id = f"{self.symbol}_{side}_{time.strftime('%Y%m%d%H%M%S')}"
        self.positions[position_id] = position
        
        logger.info(f"Opened {side} position for {self.symbol} at {price:.8f}, "
                f"SL: {stop_loss:.8f}, TP: {take_profit:.8f}, Size: {position_size:.8f}, "
                f"Confirmation: {confirmation_score:.2f}")
        
        return position
    
    def close_position(self,
                      position_id: str,
                      price: float,
                      time: datetime,
                      reason: str) -> Dict[str, Any]:
        """
        Close an existing position and calculate profit/loss.
        
        Args:
            position_id: Position identifier
            price: Exit price
            time: Exit timestamp
            reason: Reason for closing the position
            
        Returns:
            Updated position information
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return {}
        
        position = self.positions[position_id]
        
        if position['status'] != 'open':
            logger.warning(f"Position {position_id} already closed")
            return position
        
        # Calculate profit/loss
        entry_price = position['entry_price']
        side = position['side']
        size = position['size']
        
        if side == 'long':
            profit_pct = (price - entry_price) / entry_price * 100
            profit_amount = size * profit_pct / 100
        else:  # short
            profit_pct = (entry_price - price) / entry_price * 100
            profit_amount = size * profit_pct / 100
        
        # Apply fees
        fee_amount = size * self.fee_rate
        net_profit = profit_amount - fee_amount
        
        # Update position record
        position.update({
            'exit_price': price,
            'exit_time': time,
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'fee_amount': fee_amount,
            'net_profit': net_profit,
            'status': 'closed',
            'exit_reason': reason
        })
        
        # Update account state
        self.capital += net_profit
        self.position = 0
        self.positions[position_id] = position
        
        # Add to closed trades history
        self.trades.append(position)
        
        logger.info(f"Closed {side} position for {self.symbol} at {price:.8f}, "
                   f"P/L: {profit_pct:.2f}% (${net_profit:.2f}), Reason: {reason}")
        
        return position

    @profile
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest using the strategy on historical data with optimized performance."""
        # Reset strategy state
        self.reset()
        
        # Add market structure analysis
        data = self._identify_market_structure(data)
        
        # Generate signals - already optimized
        self.logger.info(f"Generating signals for {self.symbol} {self.timeframe}")
        signals = self.generate_signals(data)
        
        # Store original capital
        initial_capital = self.capital
        
        # Pre-allocate arrays instead of list append
        n_bars = len(signals)
        equity_curve = np.zeros(n_bars + 1, dtype=np.float64)
        equity_curve[0] = initial_capital
        
        # Extract key data as numpy arrays for faster access
        timestamps = signals.index.values
        prices = signals['close'].values
        signal_values = signals['signal'].values if 'signal' in signals.columns else np.zeros(len(signals))
        
        # Create progress tracker
        progress = ProgressTracker(
            name=f"Backtest for {self.symbol} {self.timeframe}", 
            total_steps=n_bars - self.lookback_window,
            log_interval=5000,  # Reduced logging frequency
            logger=self.logger
        )
        
        # Process each bar with vectorized operations
        for i in range(self.lookback_window, n_bars):
            current_price = prices[i]
            current_time = timestamps[i]
            current_signal = signal_values[i]
            
            # Update trailing stops efficiently
            for position_id, position in list(self.positions.items()):
                if position['status'] == 'open':
                    # Update trailing stop
                    new_stop = self.update_trailing_stop(position, current_price)
                    if new_stop != position['stop_loss']:
                        self.positions[position_id]['stop_loss'] = new_stop
                    
                    # Check position exit conditions
                    should_close, reason = self.should_close_position(
                        position, current_price, current_time, signals.iloc[:i+1]
                    )
                    
                    if should_close:
                        # Close position
                        self.close_position(position_id, current_price, current_time, reason)
            
            # Process new position signals
            if self.position == 0 and abs(current_signal) >= 2:
                side = 'long' if current_signal > 0 else 'short'
                self.open_position(side, current_price, current_time, signals.iloc[:i+1])
            
            # Update equity - direct calculation
            current_equity = self.calculate_equity(current_price)
            equity_curve[i+1] = current_equity
            
            # Update progress less frequently
            if i % 100 == 0:
                progress.update(min(100, n_bars - self.lookback_window - i))
        
        # Close any remaining positions
        for position_id, position in list(self.positions.items()):
            if position['status'] == 'open':
                self.close_position(position_id, prices[-1], timestamps[-1], 'end_of_data')
        
        # Final equity value after closing positions
        equity_curve[-1] = self.capital
        
        # Optimize trade statistics calculation
        trade_array = np.array([t.get('profit_pct', 0) for t in self.trades])
        win_mask = trade_array > 0
        
        win_rate = np.sum(win_mask) / len(trade_array) * 100 if len(trade_array) > 0 else 0
        total_profit = np.sum([t.get('net_profit', 0) for t in self.trades])
        
        self.logger.info(f"Backtest summary: {len(self.trades)} trades, {np.sum(win_mask)} wins, "
                        f"total profit: ${total_profit:.2f}")
        self.logger.info(f"Initial capital: ${initial_capital:.2f}, Final capital: ${self.capital:.2f}")
        self.logger.info(f"Return: {(self.capital/initial_capital - 1)*100:.2f}%")
        
        # Calculate performance metrics efficiently
        self.performance_metrics = self.calculate_performance_metrics(
            equity_curve, initial_capital, signals
        )
        
        # Prepare results without extra copies
        results = {
            'signals': signals,
            'equity_curve': equity_curve.tolist(),  # Convert to list for JSON serialization
            'trades': self.trades,
            'performance': self.performance_metrics
        }
        
        progress.complete()
        return results
    
    def calculate_equity(self, current_price: float) -> float:
        """Calculate current equity value with minimal allocations."""
        # Start with base capital
        equity = self.capital
        
        # Efficient unrealized P&L calculation
        for position in self.positions.values():
            if position['status'] != 'open':
                continue
                
            side = position['side']
            entry_price = position['entry_price']
            size = position['size']
            
            # Calculate profit percentage directly
            if side == 'long':
                profit_pct = (current_price - entry_price) / entry_price
            else:  # short
                profit_pct = (entry_price - current_price) / entry_price
            
            # Apply to equity
            equity += size * profit_pct
        
        return equity
    
    def calculate_performance_metrics(self, 
                                     equity_curve: List[float],
                                     initial_capital: float,
                                     data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics from backtest results.
        
        Args:
            equity_curve: Equity curve values
            initial_capital: Initial capital
            data: Market data with timestamps
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic metrics
        final_capital = equity_curve[-1]
        total_return = (final_capital / initial_capital - 1) * 100
        num_trades = len(self.trades)
        
        if num_trades == 0:
            return {
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'num_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0
            }
        
        # Win/loss metrics
        wins = [t for t in self.trades if t['profit_pct'] > 0]
        losses = [t for t in self.trades if t['profit_pct'] <= 0]
        
        win_rate = len(wins) / num_trades * 100
        
        avg_win = np.mean([t['profit_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['profit_pct'] for t in losses]) if losses else 0
        
        profit_sum = sum(t['net_profit'] for t in wins)
        loss_sum = abs(sum(t['net_profit'] for t in losses))
        
        profit_factor = profit_sum / loss_sum if loss_sum > 0 else float('inf')
        
        # Calculate drawdown
        peaks = pd.Series(equity_curve).cummax()
        drawdowns = (pd.Series(equity_curve) / peaks - 1) * 100
        max_drawdown = abs(drawdowns.min())
        
        # Convert equity to returns
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.0
        sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() - risk_free_rate) / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Calculate average trade duration
        if 'entry_time' in self.trades[0] and 'exit_time' in self.trades[0]:
            durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 3600 for t in self.trades]  # hours
            avg_duration = np.mean(durations)
        else:
            avg_duration = 0
        
        return {
            'initial_capital': float(initial_capital),
            'final_capital': float(final_capital),
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'max_drawdown': float(max_drawdown),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'avg_trade_duration_hours': float(avg_duration)
        }
    
    def reset(self):
        """Reset strategy state for new backtests."""
        self.position = 0
        self.positions = {}
        self.trades = []
        self.capital = self.initial_capital
        self.performance_metrics = {}
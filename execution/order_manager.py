import logging
import time
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from uuid import uuid4

from execution.exchange_connector import ExchangeConnector
from strategy.risk_management import RiskManager

class OrderManager:
    """
    Manages order creation, execution, tracking, and lifecycle.
    """
    
    def __init__(self, config_manager, exchange_connector: ExchangeConnector, risk_manager: RiskManager):
        """
        Initialize the order manager.
        
        Args:
            config_manager: Configuration manager instance
            exchange_connector: Exchange connector instance
            risk_manager: Risk manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.order_config = self.config.get('orders', {})
        
        self.exchange = exchange_connector
        self.risk_manager = risk_manager
        
        # Order tracking
        self.orders = {}  # Internal order tracking
        self.positions = {}  # Current open positions
        self.order_history = []  # Historical orders
        
        # Order parameters
        self.default_order_type = self.order_config.get('default_type', 'LIMIT')
        self.default_time_in_force = self.order_config.get('default_time_in_force', 'GTC')
        self.price_offset_pct = self.order_config.get('price_offset_pct', 0.001)  # 0.1% for limit orders
        self.enable_sl_tp = self.order_config.get('enable_sl_tp', True)
        self.enable_trailing_stop = self.order_config.get('enable_trailing_stop', True)
        
        # Order execution settings
        self.retry_attempts = self.order_config.get('retry_attempts', 3)
        self.retry_delay = self.order_config.get('retry_delay', 1.0)
        self.max_order_age = self.order_config.get('max_order_age', 60)  # seconds
        
        # Load order state if available
        self._load_state()
        
        self.logger.info("Initialized order manager")
    
    def _load_state(self) -> None:
        """Load order state from disk if available."""
        state_file = self.order_config.get('state_file', 'data/orders/order_state.json')
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.orders = state.get('orders', {})
                self.positions = state.get('positions', {})
                self.order_history = state.get('order_history', [])
                
                self.logger.info(f"Loaded order state with {len(self.orders)} orders, {len(self.positions)} positions")
            except Exception as e:
                self.logger.error(f"Error loading order state: {e}")
    
    def _save_state(self) -> None:
        """Save order state to disk."""
        state_file = self.order_config.get('state_file', 'data/orders/order_state.json')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        
        try:
            state = {
                'orders': self.orders,
                'positions': self.positions,
                'order_history': self.order_history,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.debug("Saved order state")
        except Exception as e:
            self.logger.error(f"Error saving order state: {e}")
    
    def _generate_order_id(self) -> str:
        """
        Generate a unique internal order ID.
        
        Returns:
            Unique order ID
        """
        return str(uuid4())
    
    def _calculate_price_levels(self, symbol: str, side: str, 
                             current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate price levels for order (entry, SL, TP).
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            current_price: Current price (optional, will be fetched if not provided)
            
        Returns:
            Dictionary of price levels
        """
        # Fetch current price if not provided
        if current_price is None:
            ticker = self.exchange.get_ticker(symbol)
            current_price = float(ticker['lastPrice'])
        
        # Calculate limit order price with offset
        if side.upper() == 'BUY':
            # Buy slightly above current price for guaranteed execution
            entry_price = current_price * (1 + self.price_offset_pct)
        else:
            # Sell slightly below current price for guaranteed execution
            entry_price = current_price * (1 - self.price_offset_pct)
        
        # Get ATR for adaptive stop loss if available, otherwise use default
        atr = self._get_atr(symbol)
        
        # Calculate SL and TP levels
        if atr is not None and self.enable_sl_tp:
            # Use risk manager to calculate adaptive levels
            sl_price, tp_price = self.risk_manager.get_risk_adjusted_sl_tp(
                atr=atr,
                entry_price=current_price,
                side=side.lower()
            )
        else:
            # Use fixed percentages
            sl_pct = self.order_config.get('default_sl_pct', 0.03)  # 3%
            tp_pct = self.order_config.get('default_tp_pct', 0.06)  # 6%
            
            if side.upper() == 'BUY':
                sl_price = current_price * (1 - sl_pct)
                tp_price = current_price * (1 + tp_pct)
            else:
                sl_price = current_price * (1 + sl_pct)
                tp_price = current_price * (1 - tp_pct)
        
        return {
            'entry': entry_price,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'current': current_price
        }
    
    def _get_atr(self, symbol: str, timeframe: str = '1h', window: int = 14) -> Optional[float]:
        """
        Calculate ATR for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to use
            window: ATR window
            
        Returns:
            ATR value or None if calculation fails
        """
        try:
            # Get historical data
            data = self.exchange.get_historical_data(symbol, timeframe, limit=window+10)
            
            # Calculate ATR
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # True Range
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Simple ATR calculation
            atr = np.mean(tr[-window:])
            
            return atr
        except Exception as e:
            self.logger.warning(f"Error calculating ATR for {symbol}: {e}")
            return None
    
    def create_order(self, symbol: str, side: str, quantity: Optional[float] = None,
                   capital: Optional[float] = None, price: Optional[float] = None,
                   order_type: Optional[str] = None, time_in_force: Optional[str] = None,
                   stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new order with risk management.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity (optional if capital is provided)
            capital: Amount of capital to risk (optional if quantity is provided)
            price: Order price (optional, will use market price if not provided)
            order_type: Order type (optional, uses default if not provided)
            time_in_force: Time in force (optional, uses default if not provided)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            Order details dictionary
        """
        # Validate and normalize parameters
        side = side.upper()
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid order side: {side}")
        
        order_type = (order_type or self.default_order_type).upper()
        time_in_force = time_in_force or self.default_time_in_force
        
        # Calculate price levels
        price_levels = self._calculate_price_levels(symbol, side, price)
        entry_price = price or price_levels['entry']
        
        # Use provided SL/TP or calculated levels
        sl_price = stop_loss or price_levels['stop_loss']
        tp_price = take_profit or price_levels['take_profit']
        
        # Calculate quantity if not provided
        if quantity is None:
            if capital is None:
                raise ValueError("Either quantity or capital must be provided")
            
            # Calculate quantity based on risk management
            quantity = self.risk_manager.calculate_position_size(
                capital=capital,
                entry_price=entry_price,
                stop_loss=sl_price,
                symbol=symbol
            ) / entry_price
            
            # Round quantity to appropriate precision
            quantity = round(quantity, 6)  # Adjust precision as needed
        
        # Generate internal order ID
        internal_id = self._generate_order_id()
        
        # Create order object
        order = {
            'internal_id': internal_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'price': entry_price,
            'time_in_force': time_in_force,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'status': 'CREATED',
            'exchange_id': None,
            'created_time': datetime.now().isoformat(),
            'executed_time': None,
            'executed_price': None,
            'executed_quantity': None,
            'closed_time': None,
            'profit_loss': None,
            'profit_loss_pct': None
        }
        
        # Store order
        self.orders[internal_id] = order
        
        # Save state
        self._save_state()
        
        self.logger.info(f"Created {side} order for {symbol}: {quantity} @ {entry_price}")
        
        return order
    
    def execute_order(self, order_id: str) -> Dict[str, Any]:
        """
        Execute an order on the exchange.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            Updated order details
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_id]
        
        # Check if order already executed
        if order['status'] != 'CREATED':
            self.logger.warning(f"Order {order_id} already in status: {order['status']}")
            return order
        
        # Execute with retry
        for attempt in range(self.retry_attempts):
            try:
                # Place order on exchange
                response = self.exchange.place_order(
                    symbol=order['symbol'],
                    side=order['side'],
                    order_type=order['type'],
                    quantity=order['quantity'],
                    price=order['price'] if order['type'] == 'LIMIT' else None,
                    time_in_force=order['time_in_force'] if order['type'] == 'LIMIT' else None
                )
                
                # Update order with exchange details
                order['exchange_id'] = response.get('orderId') or response.get('order_id')
                order['status'] = 'OPEN'  # or map from exchange status
                order['executed_time'] = datetime.now().isoformat()
                
                # Update state
                self.orders[order_id] = order
                self._save_state()
                
                self.logger.info(f"Executed order {order_id} on exchange, ID: {order['exchange_id']}")
                
                # Place SL/TP orders if needed
                if self.enable_sl_tp:
                    self._place_sl_tp_orders(order)
                
                return order
            
            except Exception as e:
                self.logger.error(f"Error executing order {order_id}, attempt {attempt+1}/{self.retry_attempts}: {e}")
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        # Mark order as failed if all attempts fail
        order['status'] = 'FAILED'
        self.orders[order_id] = order
        self._save_state()
        
        self.logger.error(f"Failed to execute order {order_id} after {self.retry_attempts} attempts")
        
        return order
    
    def _place_sl_tp_orders(self, order: Dict[str, Any]) -> None:
        """
        Place stop loss and take profit orders for a main order.
        
        Args:
            order: Main order details
        """
        # Only place SL/TP for successfully executed orders
        if order['status'] != 'OPEN' or order['exchange_id'] is None:
            return
        
        try:
            symbol = order['symbol']
            side = order['side']
            quantity = order['quantity']
            
            # Calculate SL/TP sides (opposite of main order)
            sl_side = 'SELL' if side == 'BUY' else 'BUY'
            
            # Place stop loss order
            sl_response = self.exchange.place_order(
                symbol=symbol,
                side=sl_side,
                order_type='STOP_LOSS',
                quantity=quantity,
                stop_price=order['stop_loss']
            )
            
            # Place take profit order
            tp_response = self.exchange.place_order(
                symbol=symbol,
                side=sl_side,
                order_type='TAKE_PROFIT',
                quantity=quantity,
                stop_price=order['take_profit']
            )
            
            # Update order with SL/TP IDs
            order['sl_order_id'] = sl_response.get('orderId') or sl_response.get('order_id')
            order['tp_order_id'] = tp_response.get('orderId') or tp_response.get('order_id')
            
            self.logger.info(f"Placed SL/TP orders for {order['internal_id']}")
        
        except Exception as e:
            self.logger.error(f"Error placing SL/TP orders for {order['internal_id']}: {e}")
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing order.
        
        Args:
            order_id: Internal order ID
            updates: Dictionary of fields to update
            
        Returns:
            Updated order details
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_id]
        
        # Update order fields
        for key, value in updates.items():
            if key in order:
                order[key] = value
        
        # Save updated state
        self.orders[order_id] = order
        self._save_state()
        
        self.logger.info(f"Updated order {order_id}")
        
        return order
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: Internal order ID
            
        Returns:
            Updated order details
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        
        order = self.orders[order_id]
        
        # Only cancel open orders
        if order['status'] != 'OPEN':
            self.logger.warning(f"Cannot cancel order {order_id} with status {order['status']}")
            return order
        
        # Cancel on exchange if exchange ID is available
        if order['exchange_id']:
            try:
                response = self.exchange.cancel_order(order['symbol'], order['exchange_id'])
                
                # Update order status
                order['status'] = 'CANCELED'
                order['closed_time'] = datetime.now().isoformat()
                
                # Save updated state
                self.orders[order_id] = order
                self._save_state()
                
                self.logger.info(f"Canceled order {order_id}")
                
                # Also cancel SL/TP orders if they exist
                if hasattr(order, 'sl_order_id') and order['sl_order_id']:
                    try:
                        self.exchange.cancel_order(order['symbol'], order['sl_order_id'])
                    except Exception as e:
                        self.logger.warning(f"Error canceling SL order: {e}")
                
                if hasattr(order, 'tp_order_id') and order['tp_order_id']:
                    try:
                        self.exchange.cancel_order(order['symbol'], order['tp_order_id'])
                    except Exception as e:
                        self.logger.warning(f"Error canceling TP order: {e}")
                
                return order
            
            except Exception as e:
                self.logger.error(f"Error canceling order {order_id}: {e}")
                
                # If order not found on exchange, mark as canceled locally
                if "order does not exist" in str(e).lower():
                    order['status'] = 'CANCELED'
                    order['closed_time'] = datetime.now().isoformat()
                    self.orders[order_id] = order
                    self._save_state()
        
        return order
    
    def update_trailing_stops(self) -> None:
        """Update trailing stops for open positions based on current prices."""
        if not self.enable_trailing_stop:
            return
        
        for position_id, position in self.positions.items():
            if position['status'] != 'OPEN':
                continue
            
            try:
                symbol = position['symbol']
                side = position['side']
                entry_price = position['entry_price']
                current_sl = position['stop_loss']
                
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                current_price = float(ticker['lastPrice'])
                
                # Calculate new stop loss based on trailing logic
                if side == 'BUY':  # Long position
                    # Calculate trailing distance as percentage of entry price
                    trail_pct = self.order_config.get('trailing_pct', 0.02)  # 2%
                    trail_distance = entry_price * trail_pct
                    
                    # Calculate potential new stop loss
                    new_sl = current_price - trail_distance
                    
                    # Only move stop loss up for long positions
                    if new_sl > current_sl:
                        # Update stop loss on exchange
                        self._update_sl_order(position, new_sl)
                        
                        # Update position
                        position['stop_loss'] = new_sl
                        self.positions[position_id] = position
                else:  # Short position
                    # Calculate trailing distance
                    trail_pct = self.order_config.get('trailing_pct', 0.02)  # 2%
                    trail_distance = entry_price * trail_pct
                    
                    # Calculate potential new stop loss
                    new_sl = current_price + trail_distance
                    
                    # Only move stop loss down for short positions
                    if new_sl < current_sl:
                        # Update stop loss on exchange
                        self._update_sl_order(position, new_sl)
                        
                        # Update position
                        position['stop_loss'] = new_sl
                        self.positions[position_id] = position
            
            except Exception as e:
                self.logger.error(f"Error updating trailing stop for {position_id}: {e}")
        
        # Save updated state
        self._save_state()
    
    def _update_sl_order(self, position: Dict[str, Any], new_sl: float) -> None:
        """
        Update a stop loss order on the exchange.
        
        Args:
            position: Position details
            new_sl: New stop loss price
        """
        try:
            # Cancel existing SL order
            if position.get('sl_order_id'):
                self.exchange.cancel_order(position['symbol'], position['sl_order_id'])
            
            # Place new SL order
            sl_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            
            sl_response = self.exchange.place_order(
                symbol=position['symbol'],
                side=sl_side,
                order_type='STOP_LOSS',
                quantity=position['quantity'],
                stop_price=new_sl
            )
            
            # Update position with new SL order ID
            position['sl_order_id'] = sl_response.get('orderId') or sl_response.get('order_id')
            
            self.logger.info(f"Updated trailing stop for {position['symbol']} to {new_sl}")
        
        except Exception as e:
            self.logger.error(f"Error updating SL order: {e}")
    
    def sync_with_exchange(self) -> None:
        """Synchronize local order state with exchange."""
        try:
            # Get open orders from exchange
            exchange_orders = self.exchange.get_open_orders()
            
            # Create mapping of exchange order IDs to our internal orders
            exchange_id_map = {}
            for order_id, order in self.orders.items():
                if order['exchange_id']:
                    exchange_id_map[order['exchange_id']] = order_id
            
            # Check for orders that are closed on exchange but open locally
            for order_id, order in list(self.orders.items()):
                if order['status'] == 'OPEN' and order['exchange_id']:
                    # Check if order is still open on exchange
                    found = False
                    for exch_order in exchange_orders:
                        if str(exch_order['orderId']) == str(order['exchange_id']):
                            found = True
                            break
                    
                    if not found:
                        # Order is closed on exchange but open locally
                        # Get order details from exchange to determine final status
                        try:
                            # This would require a separate API call to get order details
                            # For simplicity, we'll mark it as FILLED
                            order['status'] = 'FILLED'
                            order['closed_time'] = datetime.now().isoformat()
                            
                            # Move to position tracking if this was a main entry order
                            if not order.get('is_sl_tp', False):
                                self._create_position_from_order(order)
                            
                            self.logger.info(f"Synchronized order {order_id} to status FILLED")
                        except Exception as e:
                            self.logger.error(f"Error getting order details for {order_id}: {e}")
            
            # Check for order age and cancel old orders
            current_time = datetime.now()
            for order_id, order in list(self.orders.items()):
                if order['status'] == 'OPEN':
                    created_time = datetime.fromisoformat(order['created_time'].replace('Z', '+00:00'))
                    age_seconds = (current_time - created_time).total_seconds()
                    
                    if age_seconds > self.max_order_age:
                        self.logger.info(f"Canceling old order {order_id} (age: {age_seconds:.1f}s)")
                        self.cancel_order(order_id)
            
            # Save updated state
            self._save_state()
        
        except Exception as e:
            self.logger.error(f"Error synchronizing with exchange: {e}")
    
    def _create_position_from_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a position from a filled order.
        
        Args:
            order: Filled order details
            
        Returns:
            Position details
        """
        # Generate position ID
        position_id = str(uuid4())
        
        # Create position object
        position = {
            'position_id': position_id,
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'entry_price': order['executed_price'] or order['price'],
            'entry_time': order['executed_time'] or order['created_time'],
            'stop_loss': order['stop_loss'],
            'take_profit': order['take_profit'],
            'status': 'OPEN',
            'closed_time': None,
            'exit_price': None,
            'profit_loss': None,
            'profit_loss_pct': None,
            'exit_reason': None,
            'sl_order_id': order.get('sl_order_id'),
            'tp_order_id': order.get('tp_order_id')
        }
        
        # Store position
        self.positions[position_id] = position
        
        self.logger.info(f"Created position {position_id} from order {order['internal_id']}")
        
        return position
    
    def close_position(self, position_id: str, price: Optional[float] = None,
                     reason: str = 'manual') -> Dict[str, Any]:
        """
        Close an open position.
        
        Args:
            position_id: Position ID
            price: Close price (optional, uses market price if not provided)
            reason: Reason for closing position
            
        Returns:
            Updated position details
        """
        if position_id not in self.positions:
            raise ValueError(f"Position not found: {position_id}")
        
        position = self.positions[position_id]
        
        # Only close open positions
        if position['status'] != 'OPEN':
            self.logger.warning(f"Position {position_id} already closed")
            return position
        
        # Get current price if not provided
        if price is None:
            ticker = self.exchange.get_ticker(position['symbol'])
            price = float(ticker['lastPrice'])
        
        # Create closing order
        close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
        
        try:
            close_order = self.exchange.place_order(
                symbol=position['symbol'],
                side=close_side,
                order_type='MARKET',
                quantity=position['quantity']
            )
            
            # Calculate profit/loss
            entry_price = position['entry_price']
            if position['side'] == 'BUY':
                profit_pct = (price / entry_price - 1) * 100
                profit_amount = position['quantity'] * (price - entry_price)
            else:
                profit_pct = (entry_price / price - 1) * 100
                profit_amount = position['quantity'] * (entry_price - price)
            
            # Update position
            position['status'] = 'CLOSED'
            position['closed_time'] = datetime.now().isoformat()
            position['exit_price'] = price
            position['profit_loss'] = profit_amount
            position['profit_loss_pct'] = profit_pct
            position['exit_reason'] = reason
            position['close_order_id'] = close_order.get('orderId') or close_order.get('order_id')
            
            # Cancel any remaining SL/TP orders
            if position.get('sl_order_id'):
                try:
                    self.exchange.cancel_order(position['symbol'], position['sl_order_id'])
                except Exception as e:
                    self.logger.warning(f"Error canceling SL order: {e}")
            
            if position.get('tp_order_id'):
                try:
                    self.exchange.cancel_order(position['symbol'], position['tp_order_id'])
                except Exception as e:
                    self.logger.warning(f"Error canceling TP order: {e}")
            
            # Update position and save state
            self.positions[position_id] = position
            self._save_state()
            
            # Record trade for risk manager
            self.risk_manager.record_trade({
                'symbol': position['symbol'],
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': position['exit_price'],
                'quantity': position['quantity'],
                'profit_amount': position['profit_loss'],
                'profit_pct': position['profit_loss_pct'],
                'exit_reason': position['exit_reason']
            })
            
            self.logger.info(f"Closed position {position_id} with P/L: {profit_pct:.2f}%")
            
            return position
        
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
            return position
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of current positions.
        
        Returns:
            Dictionary with position summary
        """
        open_positions = [p for p in self.positions.values() if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions.values() if p['status'] == 'CLOSED']
        
        # Calculate totals
        total_profit = sum(p.get('profit_loss', 0) or 0 for p in closed_positions)
        total_open_value = sum(
            p['quantity'] * float(self.exchange.get_ticker(p['symbol'])['lastPrice'])
            for p in open_positions
        )
        
        # Count by side
        long_count = sum(1 for p in open_positions if p['side'] == 'BUY')
        short_count = sum(1 for p in open_positions if p['side'] == 'SELL')
        
        return {
            'total_open_positions': len(open_positions),
            'total_closed_positions': len(closed_positions),
            'total_profit_loss': total_profit,
            'total_open_value': total_open_value,
            'long_positions': long_count,
            'short_positions': short_count,
            'open_positions': open_positions,
            'recent_closed': closed_positions[-10:] if closed_positions else []
        }
    
    def export_trade_history(self, filepath: str) -> None:
        """
        Export trade history to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        # Combine positions and order history
        trade_history = []
        
        # Add closed positions
        for position in self.positions.values():
            if position['status'] == 'CLOSED':
                trade = {
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': position['exit_price'],
                    'exit_time': position['closed_time'],
                    'profit_loss': position['profit_loss'],
                    'profit_loss_pct': position['profit_loss_pct'],
                    'exit_reason': position['exit_reason'],
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit']
                }
                trade_history.append(trade)
        
        # Convert to DataFrame and export
        if trade_history:
            df = pd.DataFrame(trade_history)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Exported {len(trade_history)} trades to {filepath}")
        else:
            self.logger.warning("No trades to export")
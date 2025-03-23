#!/usr/bin/env python
"""
Trading Bot - Live Trading

This script handles live trading execution using the trained models.
"""

import os
import sys
import logging
import argparse
import signal
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Set
import json

from config.config_manager import ConfigManager
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.feature_selection import FeatureSelector
from models.universal_model import UniversalModel
from models.rl_model import RLModel
from execution.exchange_connector import BinanceConnector
from execution.order_manager import OrderManager
from strategy.risk_management import RiskManager
from strategy.ml_strategy import MLStrategy
from utils.logger import setup_trading_logger, TradeLogger

# Global variables for clean shutdown
running = True
trading_instances = {}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Live Trading')
    
    parser.add_argument('--config', type=str, default='config/user_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--max-capital', type=float, help='Maximum capital to use')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Dry run (generate signals but do not execute trades)')
    
    return parser.parse_args()

def load_model(model_path: str, config_manager) -> Any:
    """
    Load pre-trained model from disk.
    
    Args:
        model_path: Path to model directory
        config_manager: Configuration manager instance
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    # Check model type from metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata.get('model_type', None)
    else:
        # Try to infer model type from directory structure
        if os.path.exists(os.path.join(model_path, 'model.h5')):
            model_type = 'rl'
        else:
            model_type = config_manager.get('model.architecture', 'lstm')
    
    # Load model based on type
    if model_type == 'rl':
        logger.info("Loading RL model")
        model = RLModel(config_manager)
        model.load(model_path)
    else:
        logger.info("Loading universal model")
        model = UniversalModel(config_manager)
        model.load(model_path)
    
    logger.info(f"Loaded model from {model_path}")
    
    return model

def prepare_data_live(exchange: BinanceConnector, feature_engineer: FeatureEngineer, 
                    feature_selector: FeatureSelector, symbol: str, timeframe: str, 
                    lookback_bars: int = 500) -> pd.DataFrame:
    """
    Prepare data for live trading.
    
    Args:
        exchange: Exchange connector instance
        feature_engineer: Feature engineer instance
        feature_selector: Feature selector instance
        symbol: Trading symbol
        timeframe: Trading timeframe
        lookback_bars: Number of historical bars to fetch
        
    Returns:
        DataFrame with processed and selected features
    """
    logger = logging.getLogger(__name__)
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {symbol} {timeframe}")
    df = exchange.get_historical_data(symbol, timeframe, limit=lookback_bars)
    logger.info(f"Fetched {len(df)} bars of historical data")
    
    # Generate features
    logger.info("Generating features")
    df = feature_engineer.generate_features(df)
    logger.info(f"Generated features, shape: {df.shape}")
    
    # Select features
    logger.info("Selecting features")
    df, selected_features = feature_selector.select_features(df)
    logger.info(f"Selected {len(selected_features)} features")
    
    # Apply dimensionality reduction if configured
    if feature_engineer.config.get('features.pca.use_pca', False):
        n_components = feature_engineer.config.get('features.pca.n_components', 30)
        logger.info(f"Applying PCA with {n_components} components")
        df, pca_model, scaler = feature_engineer.apply_pca(df, n_components)
        logger.info(f"Applied PCA, new shape: {df.shape}")
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def update_data_live(df: pd.DataFrame, new_bar: pd.Series, feature_engineer: FeatureEngineer,
                   feature_selector: FeatureSelector) -> pd.DataFrame:
    """
    Update data with new bar and recalculate features.
    
    Args:
        df: Existing DataFrame with features
        new_bar: New price bar to add
        feature_engineer: Feature engineer instance
        feature_selector: Feature selector instance
        
    Returns:
        Updated DataFrame with recalculated features
    """
    logger = logging.getLogger(__name__)
    
    # Copy existing DataFrame
    updated_df = df.copy()
    
    # Add new bar
    updated_df = pd.concat([updated_df, pd.DataFrame([new_bar], index=[new_bar.name])])
    
    # Regenerate features
    updated_df = feature_engineer.generate_features(updated_df)
    
    # Reselect features if needed
    if feature_selector is not None:
        updated_df, _ = feature_selector.select_features(updated_df)
    
    # Apply PCA if needed
    if feature_engineer.config.get('features.pca.use_pca', False):
        n_components = feature_engineer.config.get('features.pca.n_components', 30)
        updated_df, _, _ = feature_engineer.apply_pca(updated_df, n_components)
    
    # Drop rows with NaN values
    updated_df.dropna(inplace=True)
    
    return updated_df

def get_next_bar_time(timeframe: str) -> datetime:
    """
    Calculate the next bar closing time for a given timeframe.
    
    Args:
        timeframe: Trading timeframe
        
    Returns:
        Datetime of next bar close
    """
    now = datetime.now()
    
    if timeframe == '1m':
        return now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    elif timeframe == '3m':
        minutes = now.minute
        next_minute = ((minutes // 3) + 1) * 3
        return now.replace(minute=next_minute % 60, second=0, microsecond=0) + timedelta(hours=next_minute // 60)
    elif timeframe == '5m':
        minutes = now.minute
        next_minute = ((minutes // 5) + 1) * 5
        return now.replace(minute=next_minute % 60, second=0, microsecond=0) + timedelta(hours=next_minute // 60)
    elif timeframe == '15m':
        minutes = now.minute
        next_minute = ((minutes // 15) + 1) * 15
        return now.replace(minute=next_minute % 60, second=0, microsecond=0) + timedelta(hours=next_minute // 60)
    elif timeframe == '30m':
        minutes = now.minute
        next_minute = ((minutes // 30) + 1) * 30
        return now.replace(minute=next_minute % 60, second=0, microsecond=0) + timedelta(hours=next_minute // 60)
    elif timeframe == '1h':
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    elif timeframe == '2h':
        hours = now.hour
        next_hour = ((hours // 2) + 1) * 2
        return now.replace(hour=next_hour % 24, minute=0, second=0, microsecond=0) + timedelta(days=next_hour // 24)
    elif timeframe == '4h':
        hours = now.hour
        next_hour = ((hours // 4) + 1) * 4
        return now.replace(hour=next_hour % 24, minute=0, second=0, microsecond=0) + timedelta(days=next_hour // 24)
    elif timeframe == '1d':
        return (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

def setup_signal_handling():
    """Set up signal handling for clean shutdown."""
    def signal_handler(sig, frame):
        global running
        print("Received shutdown signal, closing gracefully...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def save_trading_state(config_manager, trading_instances: Dict[str, Any]) -> None:
    """
    Save trading state to disk.
    
    Args:
        config_manager: Configuration manager instance
        trading_instances: Dictionary of trading instances
    """
    logger = logging.getLogger(__name__)
    
    state_file = config_manager.get('trading.state_file', 'data/trading/trading_state.json')
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    
    # Prepare state to save
    state = {
        'timestamp': datetime.now().isoformat(),
        'instances': {}
    }
    
    # Extract state from each instance
    for key, instance in trading_instances.items():
        symbol, timeframe = key.split('_')
        strategy = instance.get('strategy')
        
        if strategy:
            state['instances'][key] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'position': strategy.position,
                'capital': strategy.capital,
                'trades': strategy.trades
            }
    
    # Save state
    try:
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved trading state to {state_file}")
    except Exception as e:
        logger.error(f"Error saving trading state: {e}")

def load_trading_state(config_manager) -> Dict[str, Any]:
    """
    Load trading state from disk.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Trading state dictionary
    """
    logger = logging.getLogger(__name__)
    
    state_file = config_manager.get('trading.state_file', 'data/trading/trading_state.json')
    
    if not os.path.exists(state_file):
        logger.info("No trading state file found")
        return {}
    
    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        logger.info(f"Loaded trading state from {state_file}")
        return state
    except Exception as e:
        logger.error(f"Error loading trading state: {e}")
        return {}

def run_trading(config_manager=None, args=None):
    """
    Run live trading.
    
    Args:
        config_manager: Configuration manager instance (optional)
        args: Command line arguments (optional)
    """
    global trading_instances, running
    
    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Load config if not provided
    if config_manager is None:
        config_manager = ConfigManager(args.config)
    
    # Set up logger
    logger = setup_trading_logger(config_manager)
    trade_logger = TradeLogger(logger)
    
    # Set up signal handling for clean shutdown
    setup_signal_handling()
    
    # Get symbols and timeframes to trade
    symbols = [args.symbol] if args.symbol else config_manager.get('data.symbols', ['BTCUSDT'])
    timeframes = [args.timeframe] if args.timeframe else config_manager.get('data.timeframes', ['1h'])
    
    # Enable paper trading if specified
    if args.paper:
        config_manager.set('exchange.testnet', True)
    
    # Set maximum capital if provided
    max_capital = args.max_capital or config_manager.get('trading.max_capital')
    if max_capital:
        config_manager.set('risk.max_capital', max_capital)
    
    # Initialize exchange connector
    exchange = BinanceConnector(config_manager)
    
    # Connect to exchange
    if not exchange.connect():
        logger.error("Failed to connect to exchange")
        return 1
    
    # Initialize other components
    feature_engineer = FeatureEngineer(config_manager)
    feature_selector = FeatureSelector(config_manager)
    risk_manager = RiskManager(config_manager.get('risk', {}))
    order_manager = OrderManager(config_manager, exchange, risk_manager)
    
    # Load model if path provided, otherwise attempt to find models for each symbol/timeframe
    models = {}
    if args.model_path:
        # Use same model for all symbol/timeframe pairs
        model = load_model(args.model_path, config_manager)
        for symbol in symbols:
            for timeframe in timeframes:
                models[f"{symbol}_{timeframe}"] = model
    else:
        # Look for model for each symbol/timeframe pair
        model_dir = config_manager.get('model.directory', 'models')
        for symbol in symbols:
            for timeframe in timeframes:
                # Find most recent model for this symbol/timeframe
                model_pattern = f"{symbol}_{timeframe}_"
                matching_dirs = [d for d in os.listdir(model_dir) 
                              if os.path.isdir(os.path.join(model_dir, d)) and d.startswith(model_pattern)]
                
                if matching_dirs:
                    # Use most recent model
                    latest_model_dir = sorted(matching_dirs)[-1]
                    model_path = os.path.join(model_dir, latest_model_dir)
                    models[f"{symbol}_{timeframe}"] = load_model(model_path, config_manager)
                    logger.info(f"Found model for {symbol} {timeframe}: {model_path}")
                else:
                    logger.warning(f"No model found for {symbol} {timeframe}")
    
    # Initialize account state
    account_balance = exchange.get_account_balance()
    logger.info("Account Balance:")
    for asset, balance in account_balance.items():
        if balance['total'] > 0:
            logger.info(f"  {asset}: {balance['free']} free, {balance['locked']} locked")
    
    # Initialize strategies
    trading_instances = {}
    for symbol in symbols:
        for timeframe in timeframes:
            key = f"{symbol}_{timeframe}"
            
            if key not in models:
                logger.warning(f"Skipping {key}, no model available")
                continue
            
            # Calculate lookback window required for features
            lookback_window = config_manager.get('model.lookback_window', 60)
            lookback_bars = max(500, lookback_window * 3)  # Fetch more than needed for feature calculation
            
            try:
                # Prepare initial data
                df = prepare_data_live(exchange, feature_engineer, feature_selector, 
                                    symbol, timeframe, lookback_bars)
                
                # Initialize strategy
                strategy = MLStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    model=models[key],
                    lookback_window=lookback_window,
                    prediction_horizon=config_manager.get('model.prediction_horizon', 5),
                    config=config_manager.get('strategy', {}),
                    risk_manager=risk_manager
                )
                
                # Set initial capital
                initial_capital = config_manager.get('trading.initial_capital', 10000.0)
                strategy.initial_capital = initial_capital
                strategy.capital = initial_capital
                
                # Initialize trading instance
                trading_instances[key] = {
                    'strategy': strategy,
                    'data': df,
                    'last_update': datetime.now(),
                    'next_bar_time': get_next_bar_time(timeframe)
                }
                
                logger.info(f"Initialized trading for {symbol} {timeframe}")
            
            except Exception as e:
                logger.error(f"Error initializing trading for {symbol} {timeframe}: {e}", exc_info=True)
    
    # Load previous trading state if available
    previous_state = load_trading_state(config_manager)
    if previous_state and 'instances' in previous_state:
        for key, instance_state in previous_state['instances'].items():
            if key in trading_instances:
                # Restore account state
                strategy = trading_instances[key]['strategy']
                strategy.position = instance_state.get('position', 0)
                strategy.capital = instance_state.get('capital', strategy.initial_capital)
                
                # Restore trades history if available
                if 'trades' in instance_state:
                    strategy.trades = instance_state['trades']
                
                logger.info(f"Restored trading state for {key}")
    
    # Print trading configuration
    logger.info("Trading Configuration:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Paper Trading: {config_manager.get('exchange.testnet', True)}")
    logger.info(f"  Dry Run: {args.dry_run}")
    logger.info(f"  Initial Capital: ${config_manager.get('trading.initial_capital', 10000)}")
    
    # Main trading loop
    logger.info("Starting trading loop")
    
    try:
        while running:
            current_time = datetime.now()
            
            # Check for any new bar closes
            for key, instance in trading_instances.items():
                symbol, timeframe = key.split('_')
                strategy = instance['strategy']
                next_bar_time = instance['next_bar_time']
                
                # Check if it's time for a new bar
                if current_time >= next_bar_time:
                    logger.info(f"Processing new bar for {symbol} {timeframe}")
                    
                    try:
                        # Fetch latest data to ensure we have the complete bar
                        new_data = exchange.get_historical_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=3  # Get a few bars to ensure we have the latest
                        )
                        
                        if len(new_data) < 2:
                            logger.warning(f"Insufficient data from exchange for {symbol} {timeframe}")
                            continue
                        
                        # Get latest complete bar
                        latest_bar = new_data.iloc[-2]
                        
                        # Update data with new bar
                        updated_df = update_data_live(
                            df=instance['data'],
                            new_bar=latest_bar,
                            feature_engineer=feature_engineer,
                            feature_selector=feature_selector
                        )
                        
                        # Generate signals on updated data
                        signals = strategy.generate_signals(updated_df)
                        
                        # Check for trade signals
                        last_signal = signals['signal'].iloc[-1]
                        current_price = latest_bar['close']
                        
                        # Log signal
                        trade_logger.log_signal(
                            symbol=symbol,
                            timeframe=timeframe,
                            signal=last_signal,
                            price=current_price,
                            confidence=signals['confidence'].iloc[-1] if 'confidence' in signals.columns else None,
                            prediction=signals['prediction'].iloc[-1] if 'prediction' in signals.columns else None
                        )
                        
                        # Execute trades if not in dry run mode
                        if not args.dry_run:
                            if strategy.position == 0 and abs(last_signal) >= 2:
                                # Open new position
                                side = 'long' if last_signal > 0 else 'short'
                                
                                # Create order
                                order = order_manager.create_order(
                                    symbol=symbol,
                                    side='BUY' if side == 'long' else 'SELL',
                                    capital=strategy.capital * 0.95,  # Use 95% of available capital
                                    price=current_price
                                )
                                
                                # Execute order
                                executed_order = order_manager.execute_order(order['internal_id'])
                                
                                if executed_order['status'] == 'OPEN':
                                    # Update strategy state
                                    position = strategy.open_position(
                                        side=side,
                                        price=current_price,
                                        time=datetime.now(),
                                        data=updated_df,
                                        reason='signal'
                                    )
                                    
                                    # Log trade entry
                                    trade_logger.log_trade_entry(
                                        symbol=symbol,
                                        side=side,
                                        price=current_price,
                                        quantity=position['quantity'],
                                        order_id=executed_order['exchange_id'],
                                        reason=f"Signal: {last_signal}"
                                    )
                            
                            elif strategy.position != 0 and (
                                (strategy.position == 1 and last_signal <= -2) or
                                (strategy.position == -1 and last_signal >= 2)
                            ):
                                # Close position on signal reversal
                                for position_id, position in list(strategy.positions.items()):
                                    if position['status'] == 'OPEN':
                                        closed_position = strategy.close_position(
                                            position_id=position_id,
                                            price=current_price,
                                            reason='signal_reversal'
                                        )
                                        
                                        # Log trade exit
                                        trade_logger.log_trade_exit(
                                            symbol=symbol,
                                            side='SELL' if position['side'] == 'long' else 'BUY',
                                            price=current_price,
                                            quantity=position['quantity'],
                                            profit_pct=closed_position['profit_loss_pct'],
                                            profit_amount=closed_position['profit_loss'],
                                            reason=f"Signal Reversal: {last_signal}"
                                        )
                        
                        # Update instance data
                        instance['data'] = updated_df
                        instance['last_update'] = current_time
                        instance['next_bar_time'] = get_next_bar_time(timeframe)
                        
                        # Save trading state after each update
                        save_trading_state(config_manager, trading_instances)
                    
                    except Exception as e:
                        logger.error(f"Error processing {symbol} {timeframe}: {e}", exc_info=True)
            
            # Update trailing stops for open positions
            if not args.dry_run:
                order_manager.update_trailing_stops()
            
            # Sync with exchange and update positions
            if not args.dry_run:
                try:
                    order_manager.sync_with_exchange()
                except Exception as e:
                    logger.error(f"Error synchronizing with exchange: {e}", exc_info=True)
            
            # Wait a bit to avoid excessive API calls
            time.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    finally:
        # Perform clean shutdown
        logger.info("Shutting down trading...")
        
        # Save final trading state
        save_trading_state(config_manager, trading_instances)
        
        # Print trading summary
        logger.info("Trading Summary:")
        for key, instance in trading_instances.items():
            symbol, timeframe = key.split('_')
            strategy = instance['strategy']
            
            logger.info(f"  {symbol} {timeframe}:")
            logger.info(f"    Final Capital: ${strategy.capital:.2f}")
            logger.info(f"    Return: {(strategy.capital / strategy.initial_capital - 1) * 100:.2f}%")
            logger.info(f"    Trades: {len(strategy.trades)}")
        
        logger.info("Trading session ended")

if __name__ == '__main__':
    run_trading()
#!/usr/bin/env python
"""
Trading Bot - Model Training

This script handles the training of machine learning models for the trading bot.
"""

#!/usr/bin/env python
"""
Trading Bot - Model Training

This script handles the training of machine learning models for the trading bot.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple  # Added Tuple here

from config.config_manager import ConfigManager
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.feature_selection import FeatureSelector
from models.universal_model import UniversalModel
from models.rl_model import RLModel
from backtest.backtest_engine import BacktestEngine
from reporting.performance_report import PerformanceReport
from utils.logger import setup_trading_logger
from utils.profiling import profile
from utils.progress import ProgressTracker
from datetime import timedelta
from utils.profiling import print_profiling_stats
import tensorflow as tf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Trading Models')
    
    parser.add_argument('--config', type=str, default='config/user_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    parser.add_argument('--start-date', type=str, help='Start date for training (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for training (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, help='Directory to save model and artifacts')
    parser.add_argument('--no-validation', action='store_true', 
                      help='Skip validation and use all data for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--continue-training', type=str, 
                      help='Path to existing model to continue training')
    
    return parser.parse_args()

def prepare_data(config_manager, symbol: str, timeframe: str, start_date: Optional[str] = None,
               end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare data for model training.
    
    Args:
        config_manager: Configuration manager instance
        symbol: Trading symbol
        timeframe: Trading timeframe
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
        
    Returns:
        DataFrame with processed and selected features
    """
    logger = logging.getLogger(__name__)
    
    # Initialize components
    data_loader = DataLoader(config_manager)
    feature_engineer = FeatureEngineer(config_manager)
    feature_selector = FeatureSelector(config_manager)
    
    # Load raw data
    logger.info(f"Loading data for {symbol} {timeframe}")
    df = data_loader.load_data(symbol, timeframe, start_date, end_date)
    logger.info(f"Loaded {len(df)} rows of data")
    
    # Generate features
    logger.info("Generating features")
    df = feature_engineer.generate_features(df)
    logger.info(f"Generated features, shape: {df.shape}")
    
    # Select features
    logger.info("Selecting features")
    df, selected_features = feature_selector.select_features(df)
    logger.info(f"Selected {len(selected_features)} features")
    
    # Apply dimensionality reduction if configured
    if config_manager.get('features.pca.use_pca', False):
        n_components = config_manager.get('features.pca.n_components', 30)
        logger.info(f"Applying PCA with {n_components} components")
        df, pca_model, scaler = feature_engineer.apply_pca(df, n_components)
        logger.info(f"Applied PCA, new shape: {df.shape}")
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

def split_train_validation(df: pd.DataFrame, validation_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets respecting time order.
    
    Args:
        df: DataFrame with processed features
        validation_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_df, val_df)
    """
    # Calculate split index
    split_idx = int(len(df) * (1 - validation_ratio))
    
    # Split data
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()
    
    return train_df, val_df

def train_model(config_manager, args, symbol: str, timeframe: str, data: pd.DataFrame) -> Any:
    """
    Train a trading model.
    
    Args:
        config_manager: Configuration manager instance
        args: Command line arguments
        symbol: Trading symbol
        timeframe: Trading timeframe
        data: DataFrame with processed features
        
    Returns:
        Trained model
    """
    logger = logging.getLogger(__name__)

    # Add this block at the beginning of the function to handle missing attributes
    # Handle missing attributes in args safely using getattr with defaults
    epochs = getattr(args, 'epochs', None) or config_manager.get('model.epochs', 100)
    batch_size = getattr(args, 'batch_size', None) or config_manager.get('model.batch_size', 32)
    no_validation = getattr(args, 'no_validation', False)
    
    # Get model configuration
    model_type = config_manager.get('model.architecture', 'lstm')
    
    # Split data into training and validation sets
    no_validation = getattr(args, 'no_validation', False)
    if no_validation:
        train_data = data
        val_data = None
    else:
        validation_ratio = config_manager.get('model.validation_ratio', 0.2)
        train_data, val_data = split_train_validation(data, validation_ratio)
    
    logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data) if val_data is not None else 0} samples")
    
    # Create model
    if model_type == 'rl':
        model = RLModel(config_manager)
        feature_count = train_data.shape[1] - 5  # Subtract OHLCV columns
        model.build_model(feature_count)
        
        # Extract feature columns (exclude OHLCV)
        feature_cols = [col for col in train_data.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Prepare features
        X = train_data[feature_cols].values
        y = train_data['close'].pct_change().shift(-1).values  # Next period's return
        
        # Remove NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Train model (for RL, we use BacktestEngine to train through experiences)
        backtest_engine = BacktestEngine(config_manager)
        results = backtest_engine.run_backtest(
            symbols=[symbol],
            timeframes=[timeframe],
            model=model
        )
        
        # Return trained model
        return model
    else:
        # Create universal model
        model = UniversalModel(config_manager)
        
        # Extract feature columns (exclude OHLCV)
        feature_cols = [col for col in train_data.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Get training parameters
        epochs = getattr(args, 'epochs', None) or config_manager.get('model.epochs', 100)
        batch_size = getattr(args, 'batch_size', None) or config_manager.get('model.batch_size', 32)
        
        # Train model
        history = model.train(
            df=train_data,
            symbol=symbol,
            timeframe=timeframe,
            feature_columns=feature_cols,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2 if val_data is None else 0.0
        )
        
        # Return trained model
        return model

def save_model(model: Any, config_manager, symbol: str, timeframe: str, args) -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        config_manager: Configuration manager instance
        symbol: Trading symbol
        timeframe: Trading timeframe
        args: Command line arguments
        
    Returns:
        Path to saved model
    """
    logger = logging.getLogger(__name__)
    
    # Get output directory
    output_dir = getattr(args, 'output_dir', None)
    if output_dir is None:
        output_dir = config_manager.get('model.output_directory', 'models')
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create model directory
    model_dir = os.path.join(output_dir, f"{symbol}_{timeframe}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
 
    # Create subdirectories for model artifacts
    os.makedirs(os.path.join(model_dir, 'charts'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'reports'), exist_ok=True)
    

    # Save model
    model.save(model_dir)
    
    logger.info(f"Saved model to {model_dir}")
    
    return model_dir



def continue_training(model_path: str, config_manager, symbol: str, timeframe: str, data: pd.DataFrame) -> Any:
    """
    Continue training an existing model with new data.
    
    Args:
        model_path: Path to existing model
        config_manager: Configuration manager
        symbol: Trading symbol
        timeframe: Trading timeframe
        data: New training data
        
    Returns:
        Updated model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Continuing training for model: {model_path}")
    
    # Load existing model
    model = UniversalModel(config_manager)
    model.load(model_path)
    
    # Set symbol and timeframe attributes
    model.symbol = symbol
    model.timeframe = timeframe
    
    # Extract feature columns
    feature_cols = [col for col in data.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'signal', 'prediction', 'confidence']]
    
    # Configure for incremental learning
    epochs = config_manager.get('model.incremental_training.epochs', 20)
    learning_rate = config_manager.get('model.incremental_training.learning_rate', 0.0005)
    
    # Set the lower learning rate for fine-tuning
    if hasattr(model.model, 'optimizer'):
        tf.keras.backend.set_value(model.model.optimizer.learning_rate, learning_rate)
        logger.info(f"Adjusted learning rate to {learning_rate} for continued training")
    
    logger.info(f"Continuing training with {epochs} epochs and learning rate {learning_rate}")
    logger.info(f"Using {len(feature_cols)} features")
    
    # Train on new data
    history = model.train(
        df=data,
        symbol=symbol,
        timeframe=timeframe,
        feature_columns=feature_cols,
        epochs=epochs,
        batch_size=config_manager.get('model.batch_size', 32),
        validation_split=0.2
    )
    
    # Log training results
    final_loss = history['loss'][-1] if 'loss' in history and len(history['loss']) > 0 else 'unknown'
    logger.info(f"Continued training complete. Final loss: {final_loss}")
    
    return model

@profile
def run_training(config_manager=None, args=None):
    """
    Run model training process.
    
    Args:
        config_manager: Configuration manager instance (optional)
        args: Command line arguments (optional)
    """
    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Load config if not provided
    if config_manager is None:
        config_manager = ConfigManager(args.config)
    
    # Set up logger
    logger = setup_trading_logger(config_manager)
    
    # Record the start time for overall process
    total_start_time = time.time()

    # Get symbols and timeframes to train
    symbols = [args.symbol] if args.symbol else config_manager.get('data.symbols', ['BTCUSDT'])
    timeframes = [args.timeframe] if args.timeframe else config_manager.get('data.timeframes', ['1h'])
    
    # Get date range
    start_date = args.start_date or config_manager.get('data.start_date')
    end_date = args.end_date or config_manager.get('data.end_date')
    
    # Print training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Date Range: {start_date or 'earliest'} to {end_date or 'latest'}")
    
    # Check if we're continuing training or starting fresh
    continuing_training = hasattr(args, 'continue_training') and args.continue_training and os.path.exists(args.continue_training)
    
    if continuing_training:
        logger.info(f"Continuing training from existing model: {args.continue_training}")
    
    # Train models for each symbol/timeframe pair
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Training model for {symbol} {timeframe}")
            
            try:
                # Prepare data
                df = prepare_data(config_manager, symbol, timeframe, start_date, end_date)
                
                # Train or continue training model
                if continuing_training:
                    model = continue_training(args.continue_training, config_manager, symbol, timeframe, df)
                else:
                    model = train_model(config_manager, args, symbol, timeframe, df)
                
                # Save model
                model_path = save_model(model, config_manager, symbol, timeframe, args)
                
                # Run validation backtest
                logger.info("Running validation backtest")
                backtest_engine = BacktestEngine(config_manager)
                results = backtest_engine.run_backtest(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    model=model,
                    start_date=None,  # Use all data
                    end_date=None
                )
                
                # Generate performance report
                logger.info("Generating performance report")
                report_generator = PerformanceReport(config_manager)
                
                # Different report names based on training mode
                if continuing_training:
                    report_name = f"{symbol}_{timeframe}_continued_validation"
                else:
                    report_name = f"{symbol}_{timeframe}_validation"
                
                report = report_generator.generate_report(results, report_name=report_name)
                
                logger.info(f"Validation report generated: {report}")
                
                # Print key performance metrics
                performance_metrics = results[symbol][timeframe]['performance_metrics']
                logger.info("Validation Performance:")
                logger.info(f"  Total Return: {performance_metrics.get('total_return', 0):.2f}%")
                logger.info(f"  Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2f}%")
                logger.info(f"  Win Rate: {performance_metrics.get('win_rate', 0):.2f}%")
                
                logger.info(f"Completed training for {symbol} {timeframe}")
            
            except Exception as e:
                logger.error(f"Error training model for {symbol} {timeframe}: {e}", exc_info=True)

    # After all training is done, log the total time and print profiling stats
    total_elapsed = time.time() - total_start_time
    logger.info(f"Total training time: {str(timedelta(seconds=int(total_elapsed)))}")
    logger.info(f"Processed {len(symbols)} symbols and {len(timeframes)} timeframes")
    
    # Print profiling statistics to see which functions took the most time
    print_profiling_stats()
    
if __name__ == '__main__':
    run_training()
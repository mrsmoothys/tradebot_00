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




def configure_tensorflow_memory():
    """Configure TensorFlow for optimal memory usage on M1 Mac"""
    import tensorflow as tf
    
    # Enable memory growth to prevent TF from allocating all available memory
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass
    
    # Optimize thread allocation for M1 architecture
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Enable Metal acceleration on M1 if available
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
    except:
        pass
    
    # Enable mixed precision for better performance/memory usage
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        return True
    except:
        return False
    
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

def prepare_data_optimized(config_manager, symbol: str, timeframe: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Memory-efficient data preparation pipeline that processes data in chunks.
    
    Args:
        config_manager: Configuration manager instance
        symbol: Trading symbol
        timeframe: Trading timeframe
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
        
    Returns:
        DataFrame with optimized feature set
    """
    logger = logging.getLogger(__name__)
    
    # Initialize components
    data_loader = DataLoader(config_manager)
    feature_engineer = FeatureEngineer(config_manager)
    feature_selector = FeatureSelector(config_manager)
    
    # Load raw data
    logger.info(f"Loading data for {symbol} {timeframe}")
    df = data_loader.load_data(symbol, timeframe, start_date, end_date)
    
    # Memory optimization: Downsample if dataset is very large
    row_limit = 20000  # Maximum rows to process for memory efficiency
    if len(df) > row_limit:
        logger.info(f"Memory optimization: Downsampling large dataset from {len(df)} rows")
        # Calculate sampling interval to achieve target size
        sample_interval = max(2, len(df) // row_limit)
        df = df.iloc[::sample_interval].copy()  # Take every nth row
        logger.info(f"Downsampled to {len(df)} rows (interval: {sample_interval})")
    
    # Memory optimization: Convert to efficient dtypes
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Memory optimization: Generate features in batches with explicit cleanup between batches
    logger.info("Generating features with memory-efficient batching")
    
    # Define feature batches to process sequentially
    feature_batches = [
        ['rsi', 'atr'],  # Most essential indicators first
        ['macd', 'bollinger_bands'],
        ['sma', 'ema'], 
        ['adx', 'ichimoku', 'vwap', 'slope_momentum']  # More complex indicators last
    ]
    
    # Track generated features
    generated_features = set(df.columns)
    
    # Process each batch with memory management
    import gc
    for i, batch in enumerate(feature_batches):
        # Filter indicator configs for current batch
        batch_indicators = [
            ind for ind in config_manager.get('features.indicators', [])
            if ind['name'] in batch
        ]
        
        if not batch_indicators:
            continue
            
        logger.info(f"Processing feature batch {i+1}/{len(feature_batches)}: {batch}")
        
        # Create temporary config with just this batch
        batch_config = config_manager.config.copy() if hasattr(config_manager, 'config') else {}
        batch_config['features'] = batch_config.get('features', {})
        batch_config['features']['indicators'] = batch_indicators
        
        temp_config = ConfigManager(None)
        temp_config.config = batch_config
        
        # Create temporary feature engineer with reduced config
        batch_engineer = FeatureEngineer(temp_config)
        
        # Generate features for this batch
        try:
            df = batch_engineer.generate_features(df)
            
            # Track newly generated features
            new_features = set(df.columns) - generated_features
            generated_features = set(df.columns)
            
            logger.info(f"Generated {len(new_features)} features in batch {i+1}")
            
            # Convert any new float64 columns to float32
            for col in df.select_dtypes(include=['float64']).columns:
                if col in new_features:
                    df[col] = df[col].astype('float32')
                    
        except Exception as e:
            logger.error(f"Error generating feature batch {i+1}: {e}")
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Memory optimization: Select fewer features if there are too many
    logger.info("Selecting optimal features for memory efficiency")
    
    # Limit feature count based on available memory
    max_features = 30  # Reduced from default for memory efficiency
    
    # Update feature selector config
    if 'features' in config_manager.config and 'selection' in config_manager.config['features']:
        current_n_features = config_manager.config['features']['selection'].get('params', {}).get('n_features', 50)
        if current_n_features > max_features:
            logger.info(f"Memory optimization: Reducing feature count from {current_n_features} to {max_features}")
            config_manager.set('features.selection.params.n_features', max_features)
    
    # Select features
    df, selected_features = feature_selector.select_features(df)
    logger.info(f"Selected {len(selected_features)} optimized features")
    
    # Memory optimization: Skip PCA for memory efficiency
    if config_manager.get('features.pca.use_pca', False):
        # Override PCA settings for memory efficiency
        logger.info("Memory optimization: Disabling PCA to save memory")
        config_manager.set('features.pca.use_pca', False)
    
    # Final cleanup of dataframe
    df.dropna(inplace=True)
    
    # Log memory usage after preparation
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage after data preparation: {memory_info.rss / (1024 * 1024):.2f} MB")
    except ImportError:
        pass
    
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


def train_model_optimized(config_manager, args, symbol: str, timeframe: str, data: pd.DataFrame) -> Any:
    """
    Memory-optimized model training with incremental batch processing.
    
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
    
    feature_selector = FeatureSelector(config_manager)

    # Memory optimization: Enforce memory-efficient training parameters
    epochs = min(getattr(args, 'epochs', None) or config_manager.get('model.epochs', 100), 20)
    batch_size = min(getattr(args, 'batch_size', None) or config_manager.get('model.batch_size', 32), 16)
    
    # Get model configuration
    model_type = config_manager.get('model.architecture', 'lstm')
    
    # Split data with memory efficiency considerations
    no_validation = getattr(args, 'no_validation', False)
    if no_validation:
        train_data = data
        val_data = None
    else:
        # Use smaller validation split for memory efficiency
        validation_ratio = min(config_manager.get('model.validation_ratio', 0.2), 0.1)
        split_idx = int(len(data) * (1 - validation_ratio))
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
    
    logger.info(f"Memory-optimized training on {len(train_data)} samples, validating on {len(val_data) if val_data is not None else 0} samples")
    
    # Create model with architecture optimized for memory constraints
    if model_type == 'rl':
        # Handle RL model
        model = RLModel(config_manager)
        feature_count = train_data.shape[1] - 5  # Subtract OHLCV columns
        model.build_model(feature_count)
        
        # Extract feature columns (exclude OHLCV)
        feature_cols = [col for col in train_data.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Memory optimization: Limit feature count if too many
        if len(feature_cols) > 30:
            logger.info(f"Memory optimization: Limiting feature count from {len(feature_cols)} to 30")
            feature_cols = feature_cols[:30]
        
        # Train model using BacktestEngine's experience generation
        backtest_engine = BacktestEngine(config_manager)
        results = backtest_engine.run_backtest(
            symbols=[symbol],
            timeframes=[timeframe],
            model=model,
            data={symbol: {timeframe: train_data}}  # Pass data directly
        )
        
    else:
        # Create universal model with memory optimizations
        import tensorflow as tf
        
        # Memory optimization: Create memory monitoring callback
        class MemoryCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger
                self.epoch_count = 0
                
            def on_epoch_end(self, epoch, logs=None):
                import gc
                self.epoch_count += 1
                
                # Force cleanup every 3 epochs
                if self.epoch_count % 3 == 0:
                    gc.collect()
                    self.logger.info(f"Memory cleanup after epoch {epoch+1}")
        
        model = UniversalModel(config_manager)
        
        # Extract feature columns efficiently
        feature_cols = [col for col in train_data.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Memory optimization: Limit feature count
        max_features = 30
        if len(feature_cols) > max_features:
            logger.info(f"Memory optimization: Limiting features from {len(feature_cols)} to {max_features}")
            # Choose highest correlation features if available
            if hasattr(feature_selector, 'feature_importances_') and feature_selector.feature_importances_ is not None:
                sorted_features = sorted(zip(feature_cols, feature_selector.feature_importances_), 
                                      key=lambda x: x[1], reverse=True)
                feature_cols = [f[0] for f in sorted_features[:max_features]]
            else:
                # Otherwise just take first N features
                feature_cols = feature_cols[:max_features]
        
        # Add memory monitoring callback
        memory_callback = MemoryCallback(logger)
        
        # Memory-efficient early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',  # Monitor training loss to save memory by skipping validation
            patience=3,
            restore_best_weights=True
        )
        
        # Progressive training approach for large datasets
        if len(train_data) > 5000:
            logger.info("Memory optimization: Using progressive training for large dataset")
            
            # Phase 1: Train on smaller subset first
            subset_size = min(3000, len(train_data) // 2)
            subset_data = train_data.iloc[:subset_size].copy()
            
            logger.info(f"Phase 1: Training on {subset_size} samples")
            model.train(
                df=subset_data,
                symbol=symbol,
                timeframe=timeframe,
                feature_columns=feature_cols,
                epochs=epochs // 2,  # Fewer epochs for initial phase
                batch_size=batch_size,
                validation_split=0.0,  # Skip validation to save memory
                callbacks=[early_stopping, memory_callback]
            )
            
            # Force cleanup between phases
            import gc
            gc.collect()
            
            # Phase 2: Continue with full dataset at lower learning rate
            logger.info(f"Phase 2: Training on full dataset ({len(train_data)} samples)")
            
            # Reduce learning rate for fine-tuning
            original_lr = config_manager.get('model.learning_rate', 0.001)
            config_manager.set('model.learning_rate', original_lr / 5)
            
            # Train on full dataset
            model.train(
                df=train_data,
                symbol=symbol,
                timeframe=timeframe,
                feature_columns=feature_cols,
                epochs=epochs // 2,
                batch_size=batch_size,
                validation_split=0.0,
                callbacks=[early_stopping, memory_callback]
            )
            
            # Restore original learning rate
            config_manager.set('model.learning_rate', original_lr)
            
        else:
            # Standard training for smaller datasets
            logger.info(f"Standard training on {len(train_data)} samples")
            model.train(
                df=train_data,
                symbol=symbol,
                timeframe=timeframe,
                feature_columns=feature_cols,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.0 if val_data is None else 0.1,
                callbacks=[early_stopping, memory_callback]
            )
    
    # Force garbage collection after training
    import gc
    gc.collect()
    
    return model

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
    Memory-efficient implementation for continuing training of an existing model.
    
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
    
    # Load existing model with memory optimizations
    try:
        # Force garbage collection before loading model
        import gc
        gc.collect()
        
        model = UniversalModel(config_manager)
        model.load(model_path)
        
        # Set symbol and timeframe attributes
        model.symbol = symbol
        model.timeframe = timeframe
        
        # Extract feature columns with memory constraints
        feature_cols = [col for col in data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'signal', 'prediction', 'confidence']]
        
        # Memory optimization: Limit features if too many
        max_features = 30
        if len(feature_cols) > max_features:
            logger.info(f"Memory optimization: Limiting features from {len(feature_cols)} to {max_features}")
            feature_cols = feature_cols[:max_features]
        
        # Memory-efficient configuration for incremental learning
        epochs = min(config_manager.get('model.incremental_training.epochs', 20), 10)
        batch_size = min(config_manager.get('model.batch_size', 32), 16)
        learning_rate = config_manager.get('model.incremental_training.learning_rate', 0.0001)
        
        # Use lower learning rate for fine-tuning
        if hasattr(model.model, 'optimizer'):
            import tensorflow as tf
            tf.keras.backend.set_value(model.model.optimizer.learning_rate, learning_rate)
            logger.info(f"Adjusted learning rate to {learning_rate} for continued training")
        
        logger.info(f"Memory-optimized continued training with {epochs} epochs, batch size {batch_size}, and {len(feature_cols)} features")
        
        # Memory monitoring callback
        import tensorflow as tf
        class MemoryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 2 == 0:  # Every 2 epochs
                    gc.collect()
        
        # Early stopping for efficiency
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=2,
            restore_best_weights=True
        )
        
        # Train on new data with memory optimization
        history = model.train(
            df=data,
            symbol=symbol,
            timeframe=timeframe,
            feature_columns=feature_cols,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,  # Skip validation to save memory
            callbacks=[MemoryCallback(), early_stopping]
        )
        
        # Log training results
        final_loss = history['loss'][-1] if 'loss' in history and len(history['loss']) > 0 else 'unknown'
        logger.info(f"Continued training complete. Final loss: {final_loss}")
        
        # Force cleanup
        gc.collect()
        
        return model
        
    except Exception as e:
        logger.error(f"Error during continued training: {e}", exc_info=True)
        raise


def continue_training_efficiently(model_path, config_manager, symbol, timeframe, data, logger):
    """Continue training with memory-optimized approach."""
    import gc
    
    # Force cleanup before loading model
    gc.collect()
    
    logger.info(f"Loading model from {model_path} for continued training")
    model = UniversalModel(config_manager)
    model.load(model_path)
    
    # Set symbol and timeframe attributes
    model.symbol = symbol
    model.timeframe = timeframe
    
    # Extract feature columns with limit
    feature_cols = [col for col in data.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'signal', 'prediction', 'confidence']]
    
    # Memory optimization: Limit feature count
    max_features = 30
    if len(feature_cols) > max_features:
        logger.info(f"Limiting features from {len(feature_cols)} to {max_features}")
        feature_cols = feature_cols[:max_features]
    
    # Configure for incremental training
    epochs = min(config_manager.get('model.incremental_training.epochs', 20), 10)
    batch_size = min(config_manager.get('model.batch_size', 32), 16)
    
    # Reduce learning rate for fine-tuning
    original_lr = model.learning_rate
    reduced_lr = original_lr * 0.2  # 20% of original learning rate
    
    if hasattr(model.model, 'optimizer'):
        import tensorflow as tf
        tf.keras.backend.set_value(model.model.optimizer.learning_rate, reduced_lr)
        logger.info(f"Reduced learning rate from {original_lr} to {reduced_lr} for fine-tuning")
    
    # Memory-efficient continued training
    train_with_memory_efficiency(
        model=model,
        df=data,
        symbol=symbol,
        timeframe=timeframe,
        feature_columns=feature_cols,
        epochs=epochs,
        batch_size=batch_size,
        logger=logger
    )
    
    # Force cleanup after training
    gc.collect()
    
    return model


def train_with_memory_efficiency(model, df, symbol, timeframe, feature_columns, epochs, batch_size, logger):
    """Implement progressive training to minimize memory footprint without requiring callbacks."""
    import gc
    import tensorflow as tf
    
    # Force garbage collection before training
    gc.collect()
    
    # Progressive training approach 
    if len(df) > 3000:
        logger.info("Using progressive training approach for large dataset")
        
        # Phase 1: Train on small subset (30% of data)
        subset_size = min(2000, len(df) // 3)
        subset_data = df.sample(subset_size, random_state=42).copy()
        
        logger.info(f"Phase 1: Training on {subset_size} samples ({subset_size/len(df)*100:.1f}%)")
        
        # First training phase - without callbacks parameter
        initial_history = model.train(
            df=subset_data,
            symbol=symbol,
            timeframe=timeframe,
            feature_columns=feature_columns,
            epochs=max(5, epochs // 3),  # Fewer epochs for initial training
            batch_size=batch_size,
            validation_split=0.0  # Skip validation to save memory
        )
        
        # Force cleanup between phases
        gc.collect()
        
        # Phase 2: Train on full dataset with lower learning rate
        logger.info(f"Phase 2: Training on full dataset ({len(df)} samples)")
        
        # Adjust learning rate through the model's attributes rather than via callbacks
        if hasattr(model.model, 'optimizer'):
            initial_lr = tf.keras.backend.get_value(model.model.optimizer.learning_rate)
            reduced_lr = initial_lr * 0.2  # 20% of original learning rate
            tf.keras.backend.set_value(model.model.optimizer.learning_rate, reduced_lr)
            logger.info(f"Reduced learning rate from {initial_lr} to {reduced_lr}")
        
        # Second training phase - without callbacks parameter
        final_history = model.train(
            df=df,
            symbol=symbol,
            timeframe=timeframe, 
            feature_columns=feature_columns,
            epochs=max(5, epochs // 2),  # Fewer epochs for fine-tuning
            batch_size=batch_size,
            validation_split=0.0  # Skip validation to save memory
        )
        
        # Manual garbage collection after training
        gc.collect()
        
        # Combine histories
        combined_history = {}
        for key in initial_history:
            if key in final_history:
                combined_history[key] = initial_history[key] + final_history[key]
            else:
                combined_history[key] = initial_history[key]
                
        return combined_history
    else:
        # Standard training for smaller datasets - without callbacks parameter
        return model.train(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            feature_columns=feature_columns,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1  # Small validation split
        )
    
def prepare_data_efficiently(config_manager, symbol, timeframe, start_date, end_date, logger):
    """Prepare data with memory efficiency as the priority."""
    import gc
    
    # Initialize components
    data_loader = DataLoader(config_manager)
    feature_engineer = FeatureEngineer(config_manager)
    feature_selector = FeatureSelector(config_manager)
    
    # Load data with memory optimization
    logger.info(f"Loading data for {symbol} {timeframe}")
    df = data_loader.load_data(symbol, timeframe, start_date, end_date)
    
    # Memory optimization: Sample if dataset is very large
    if len(df) > 15000:  # Threshold for M1 Mac
        sample_ratio = 15000 / len(df)
        logger.info(f"Dataset too large ({len(df)} rows), sampling {sample_ratio*100:.1f}% for memory efficiency")
        df = df.sample(frac=sample_ratio, random_state=42).copy()
    
    # Convert to float32 for memory efficiency
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    # Generate features (already optimized)
    logger.info("Generating features with memory optimization")
    df = feature_engineer.generate_features(df)
    
    # Force cleanup
    gc.collect()
    
    # Feature selection with memory constraint
    logger.info("Selecting features")
    # Limit number of features for M1 Mac
    max_features = 30
    if hasattr(config_manager, 'set'):
        original_n_features = config_manager.get('features.selection.params.n_features', 50)
        if original_n_features > max_features:
            logger.info(f"Limiting feature count from {original_n_features} to {max_features} for memory efficiency")
            config_manager.set('features.selection.params.n_features', max_features)
    
    df, selected_features = feature_selector.select_features(df)
    logger.info(f"Selected {len(selected_features)} features")
    
    # Disable PCA to save memory
    if config_manager.get('features.pca.use_pca', False):
        logger.info("Disabling PCA to save memory")
        config_manager.set('features.pca.use_pca', False)
    
    # Final cleanup of NaN values
    df.dropna(inplace=True)
    
    # Force garbage collection
    gc.collect()
    
    return df


def optimize_config_for_memory(config_manager, logger):
    """Adjust configuration parameters for memory-constrained environment."""
    # Reduce model complexity
    original_epochs = config_manager.get('model.epochs', 100)
    if original_epochs > 20:
        logger.info(f"Reducing epochs from {original_epochs} to 20 for memory efficiency")
        config_manager.set('model.epochs', 20)
    
    original_batch = config_manager.get('model.batch_size', 32)
    if original_batch > 16:
        logger.info(f"Reducing batch size from {original_batch} to 16 for memory efficiency")
        config_manager.set('model.batch_size', 16)
    
    # Reduce feature count
    original_indicators = config_manager.get('features.indicators', [])
    if len(original_indicators) > 5:
        essential_indicators = [ind for ind in original_indicators 
                             if ind['name'] in ['rsi', 'macd', 'atr', 'sma', 'ema']]
        if len(essential_indicators) >= 3:
            logger.info(f"Limiting indicators from {len(original_indicators)} to {len(essential_indicators)}")
            config_manager.set('features.indicators', essential_indicators)
    
    # Disable memory-intensive options
    if config_manager.get('features.pca.use_pca', False):
        logger.info("Disabling PCA for memory efficiency")
        config_manager.set('features.pca.use_pca', False)
    
    # Use simpler model architecture if possible
    model_type = config_manager.get('model.architecture', 'lstm')
    if model_type == 'transformer':  # Most memory-intensive
        logger.info("Switching from transformer to GRU for memory efficiency")
        config_manager.set('model.architecture', 'gru')  # GRU is more memory efficient



@profile
def run_training(config_manager=None, args=None):
    """Run model training with memory optimization for M1 Mac."""
    # Configure TensorFlow for memory efficiency
    configure_tensorflow_memory()
    
    # Force garbage collection at start
    import gc
    gc.collect()
    
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

    # Memory optimization: Reduce parameter values
    optimize_config_for_memory(config_manager, logger)
    
    # Get symbols and timeframes to train
    symbols = [args.symbol] if args.symbol else config_manager.get('data.symbols', ['BTCUSDT'])
    timeframes = [args.timeframe] if args.timeframe else config_manager.get('data.timeframes', ['1h'])
    
    # Get date range
    start_date = args.start_date or config_manager.get('data.start_date')
    end_date = args.end_date or config_manager.get('data.end_date')
    
    # Print training configuration
    logger.info("Training Configuration (Memory-Optimized for M1 Mac):")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Date Range: {start_date or 'earliest'} to {end_date or 'latest'}")
    logger.info(f"  Batch Size: {config_manager.get('model.batch_size', 16)}")
    logger.info(f"  Epochs: {config_manager.get('model.epochs', 20)}")
    
    # Check if we're continuing training or starting fresh
    continuing_training = hasattr(args, 'continue_training') and args.continue_training and os.path.exists(args.continue_training)
    
    if continuing_training:
        logger.info(f"Continuing training from existing model: {args.continue_training}")
    
    # Train models for each symbol/timeframe pair
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"Training model for {symbol} {timeframe}")
            
            try:
                # Prepare data efficiently
                df = prepare_data_efficiently(config_manager, symbol, timeframe, start_date, end_date, logger)
                
                # Train or continue training model
                if continuing_training:
                    model = continue_training_efficiently(args.continue_training, config_manager, symbol, timeframe, df, logger)
                else:
                    # Create model with memory optimization
                    model_type = config_manager.get('model.architecture', 'lstm')
                    
                    if model_type == 'rl':
                        model = RLModel(config_manager)
                        feature_count = min(df.shape[1] - 5, 30)  # Limit features for memory
                        model.build_model(feature_count)
                    else:
                        model = UniversalModel(config_manager)
                    
                    # Extract feature columns with limit
                    feature_cols = [col for col in df.columns 
                                  if col not in ['open', 'high', 'low', 'close', 'volume']]
                    
                    # Memory optimization: Limit feature count
                    max_features = 30
                    if len(feature_cols) > max_features:
                        logger.info(f"Limiting features from {len(feature_cols)} to {max_features}")
                        feature_cols = feature_cols[:max_features]
                    
                    # Train with memory efficiency
                    epochs = min(args.epochs if hasattr(args, 'epochs') and args.epochs else 
                               config_manager.get('model.epochs', 100), 20)
                    batch_size = min(args.batch_size if hasattr(args, 'batch_size') and args.batch_size else 
                                   config_manager.get('model.batch_size', 32), 16)
                    
                    # Progressive training
                    train_with_memory_efficiency(
                        model=model,
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        feature_columns=feature_cols,
                        epochs=epochs,
                        batch_size=batch_size,
                        logger=logger
                    )
                
                # Force cleanup
                gc.collect()
                
                # Save model
                model_path = save_model(model, config_manager, symbol, timeframe, args)
                
                # Run reduced validation to save memory
                validation_size = min(len(df), 3000)
                validation_df = df.iloc[-validation_size:].copy() if len(df) > validation_size else df
                
                logger.info(f"Running validation on {len(validation_df)} samples")
                backtest_engine = BacktestEngine(config_manager)
                results = backtest_engine.run_backtest(
                    symbols=[symbol],
                    timeframes=[timeframe],
                    model=model,
                    data={symbol: {timeframe: validation_df}}
                )
                
                # Generate performance report
                logger.info("Generating performance report")
                report_generator = PerformanceReport(config_manager)
                report_name = f"{symbol}_{timeframe}_{'continued' if continuing_training else 'initial'}_validation"
                report = report_generator.generate_report(results, report_name=report_name)
                
                logger.info(f"Model training and validation complete for {symbol} {timeframe}")
                
                # Final cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error training model for {symbol} {timeframe}: {e}", exc_info=True)
                continue

    # Final cleanup
    gc.collect()

    # Log total time
    total_elapsed = time.time() - total_start_time
    logger.info(f"Total training time: {str(timedelta(seconds=int(total_elapsed)))}")
    
    # Print profiling statistics
    print_profiling_stats()
    
if __name__ == '__main__':
    run_training()
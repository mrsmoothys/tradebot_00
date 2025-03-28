import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization, Input,
    Bidirectional, Conv1D, MaxPooling1D, Flatten, Concatenate,
    Embedding, MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from strategy.ml_strategy import DataTypeValidator
from utils.profiling import profile
from utils.progress import ProgressTracker
import time
from utils.data_quality import DataQualityMonitor

# Limit memory growth to prevent TensorFlow from allocating all GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Set reasonable memory limits for CPU operations
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configure for Metal acceleration on M1
tf.config.experimental.set_visible_devices([], 'GPU')  # Disable GPU for now to use Metal

# Define the callback class outside of UniversalModel
class EpochProgressCallback(tf.keras.callbacks.Callback):
    """Callback to track training progress per epoch."""
    
    def __init__(self, logger, total_epochs):
        super().__init__()
        self.logger = logger
        self.total_epochs = total_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        
        # Create metrics string from logs
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        
        # Log progress
        self.logger.info(
            f"Epoch {epoch+1}/{self.total_epochs} completed in {elapsed:.2f}s - {metrics_str}"
        )



class UniversalModel:
    """
    Universal model that can handle multiple symbols and timeframes.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the universal model.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        
        # Extract model configuration
        self.lookback_window = self.config.get('model.lookback_window', 60)
        self.prediction_horizon = self.config.get('model.prediction_horizon', 5)
        self.hidden_layers = self.config.get('model.hidden_layers', [128, 64, 32])
        self.dropout_rate = self.config.get('model.dropout_rate', 0.2)
        self.learning_rate = self.config.get('model.learning_rate', 0.001)
        self.batch_size = self.config.get('model.batch_size', 32)
        self.epochs = self.config.get('model.epochs', 100)
        self.model_type = self.config.get('model.architecture', 'lstm')
        
        # Determine feature count (default to 30 for PCA components)
        feature_count = self.config.get('features.selection.params.n_features', 30)
        self.input_shape = (None, self.lookback_window, feature_count)  # Add this line to initialize input_shape

        # Map to store symbol and timeframe information
        self.symbol_map = {}  # Maps symbols to numeric IDs
        self.next_symbol_id = 0
        
        # Timeframe map (minutes)
        self.timeframe_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        # Initialize model
        self.model = None
        self._build_model()
        
        # Training history
        self.training_history = {}


    def _directional_loss(self, y_true, y_pred):
        """
        Custom loss function that emphasizes directional accuracy of predictions.
        
        Args:
            y_true: True values (target prices)
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        import tensorflow as tf
        
        # Calculate percentage change for true values
        true_change = (y_true[:, -1] - y_true[:, 0]) / y_true[:, 0]
        
        # Calculate percentage change for predictions
        pred_change = (y_pred[:, -1] - y_true[:, 0]) / y_true[:, 0]
        
        # Direction agreement component (binary)
        same_direction = tf.cast(tf.sign(true_change) == tf.sign(pred_change), tf.float32)
        direction_penalty = 1.0 - same_direction
        
        # Magnitude error component (MSE)
        magnitude_error = tf.square(pred_change - true_change)
        
        # Combined loss: weighted sum of direction penalty and magnitude error
        # Higher weight on direction to prioritize getting the direction right
        direction_weight = 2.0
        magnitude_weight = 1.0
        
        combined_loss = direction_weight * direction_penalty + magnitude_weight * magnitude_error
        
        return tf.reduce_mean(combined_loss)
    
    # Modify _build_model in universal_model.py

    def _build_model(self) -> None:
        """Build a dynamic, memory-efficient model architecture compatible with varying feature counts."""
        try:
            # Extract dimensions with explicit error handling
            if len(self.input_shape) == 3:
                _, timesteps, feature_dim = self.input_shape
            elif len(self.input_shape) == 2:
                # Handle legacy 2D input shape format
                timesteps, feature_dim = self.input_shape
                self.logger.warning(f"Legacy 2D input shape detected ({timesteps}, {feature_dim}), converting to 3D format")
                self.input_shape = (None, timesteps, feature_dim)
            else:
                raise ValueError(f"Invalid input shape: {self.input_shape}, expected 2D or 3D tensor specification")
            
            self.logger.info(f"Building model with input shape: (batch, {timesteps}, {feature_dim})")
            
            # Create input tensors with correct dimensions
            price_input = Input(shape=(timesteps, feature_dim), name='price_input')
            symbol_input = Input(shape=(1,), dtype='int32', name='symbol_input')
            timeframe_input = Input(shape=(1,), name='timeframe_input')
            
            # Memory-efficient model architecture
            # Reduced complexity symbol embedding
            symbol_embedding = Embedding(input_dim=100, output_dim=4)(symbol_input)
            symbol_embedding = Flatten()(symbol_embedding)
            
            # Specialized architecture based on model type
            if self.model_type == 'lstm':
                # Single LSTM layer to reduce memory consumption
                x = LSTM(64, return_sequences=False)(price_input)
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
            elif self.model_type == 'gru':
                # GRU is typically more memory-efficient than LSTM
                x = GRU(64, return_sequences=False)(price_input)
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
            else:
                # Fallback to simple dense network for maximum memory efficiency
                x = Flatten()(price_input)
                x = Dense(64, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(self.dropout_rate)(x)
            
            # Combine features with metadata
            combined = Concatenate()([x, symbol_embedding, timeframe_input])
            x = Dense(32, activation='relu')(combined)
            outputs = Dense(self.prediction_horizon, activation='linear')(x)
            
            # Create model with memory-efficient optimizer
            model = Model(inputs=[price_input, symbol_input, timeframe_input], outputs=outputs)
            
            # Configure optimizer with gradient clipping for stability and memory efficiency
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0  # Prevent explosive gradients
            )
            
            # Compile with custom loss
            model.compile(
                optimizer=optimizer,
                loss=self._directional_loss,
                metrics=['mae', 'mse']
            )
            
            self.model = model
            self.logger.info(f"Successfully built {self.model_type} model with {model.count_params()} parameters")
            
        except Exception as e:
            self.logger.error(f"Model building failed: {e}")
            # Create a minimal fallback model to prevent catastrophic failure
            self._build_fallback_model()

    def _build_fallback_model(self):
        """Build minimal fallback model when standard construction fails."""
        self.logger.warning("Building minimal fallback model")
        
        # Use absolute minimum feature count
        min_feature_count = 5
        min_lookback = 10
        
        # Override input shape
        self.input_shape = (None, min_lookback, min_feature_count)
        
        # Simple inputs
        price_input = Input(shape=(min_lookback, min_feature_count), name='price_input')
        symbol_input = Input(shape=(1,), dtype='int32', name='symbol_input')
        timeframe_input = Input(shape=(1,), name='timeframe_input')
        
        # Extremely simplified architecture
        x = Flatten()(price_input)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Simple symbol embedding
        symbol_embedding = Embedding(input_dim=10, output_dim=2)(symbol_input)
        symbol_embedding = Flatten()(symbol_embedding)
        
        # Combine features
        combined = Concatenate()([x, symbol_embedding, timeframe_input])
        outputs = Dense(self.prediction_horizon, activation='linear')(combined)
        
        # Create model
        model = Model(inputs=[price_input, symbol_input, timeframe_input], outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        self.model = model
        self.logger.warning(f"Fallback model built with {model.count_params()} parameters")

    
    def _build_lstm_layers(self, inputs):
        """Build LSTM layers for the model."""
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            return_sequences = i < len(self.hidden_layers) - 1
            x = LSTM(units, return_sequences=return_sequences)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        return x
    
    def _build_gru_layers(self, inputs):
        """Build GRU layers for the model."""
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            return_sequences = i < len(self.hidden_layers) - 1
            x = GRU(units, return_sequences=return_sequences)(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        return x
    
    def _build_cnn_layers(self, inputs):
        """Build CNN layers for the model."""
        x = inputs
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Flatten()(x)
        
        for units in self.hidden_layers:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        return x
    
    def _build_transformer_layers(self, inputs):
        """Build Transformer layers for the model."""
        x = inputs
        
        # Add positional encoding if needed
        
        # Transformer blocks
        for _ in range(3):  # 3 transformer blocks
            # Multi-head attention
            attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x, x)
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn = tf.keras.Sequential([
                Dense(self.hidden_layers[0], activation='relu'),
                Dense(inputs.shape[-1])
            ])
            
            # Skip connection and layer normalization
            x = LayerNormalization(epsilon=1e-6)(ffn(x) + x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        return x
    
    def _get_symbol_id(self, symbol: str) -> int:
        """
        Get numeric ID for a symbol, creating a new one if needed.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Numeric ID for the symbol
        """
        if symbol not in self.symbol_map:
            self.symbol_map[symbol] = self.next_symbol_id
            self.next_symbol_id += 1
        
        return self.symbol_map[symbol]
    
    def _get_timeframe_minutes(self, timeframe: str) -> float:
        """
        Convert timeframe to minutes for numeric representation.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '15m')
            
        Returns:
            Timeframe in minutes
        """
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return self.timeframe_map[timeframe]
    
    def prepare_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                    feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training or prediction.
        
        Args:
            df: DataFrame with OHLCV and features
            symbol: Trading symbol
            timeframe: Trading timeframe
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X_price, X_symbol, X_timeframe, y) arrays
        """

        expected_feature_count = self.input_shape[-1]
        actual_feature_count = len(feature_columns)
        
        # If mismatch detected, handle it
        if actual_feature_count < expected_feature_count:
            self.logger.warning(f"Feature count mismatch: padding from {actual_feature_count} to {expected_feature_count}")
            
            # Pad with zeros to reach expected dimension
            padding_count = expected_feature_count - actual_feature_count
            
            # Create synthetic feature columns with zeros
            for i in range(padding_count):
                padding_col = f"padding_{i}"
                df[padding_col] = 0.0
                feature_columns.append(padding_col)

        monitor = DataQualityMonitor(self.logger)
    
        # Verify input quality
        monitor.check_dataframe(df, f"{symbol}_{timeframe}_model_input")

        # Create sequences
        X, y = self._create_sequences(df, feature_columns)
        
        # Create symbol and timeframe inputs
        symbol_id = self._get_symbol_id(symbol)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        X_symbol = np.full((len(X), 1), symbol_id, dtype=np.int32)
        X_timeframe = np.full((len(X), 1), timeframe_minutes, dtype=np.float32)
        
        return X, X_symbol, X_timeframe, y
    
    def _create_sequences(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from DataFrame.
        
        Args:
            df: DataFrame with features
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Get target (close price)
        close_price = df['close'].values
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(df) - self.lookback_window - self.prediction_horizon + 1):
            # Extract sequence for features
            features_sequence = df[feature_columns].iloc[i:i+self.lookback_window].values
            
            # Extract target sequence (future close prices)
            target_sequence = close_price[i+self.lookback_window:i+self.lookback_window+self.prediction_horizon]
            
            X.append(features_sequence)
            y.append(target_sequence)
        
        return np.array(X), np.array(y)
    
    # Optimize train method in universal_model.py

    @profile
    def train(self, df: pd.DataFrame, symbol: str, timeframe: str,
            feature_columns: List[str], epochs: int = None,
            batch_size: int = None, validation_split: float = 0.2,
            callbacks: List[Any] = None) -> Dict[str, List[float]]:
        """
        Memory-optimized training with dynamic feature handling.
        
        Args:
            df: DataFrame with features
            symbol: Trading symbol
            timeframe: Trading timeframe
            feature_columns: List of feature column names
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            callbacks: List of Keras callbacks (optional)
            
        Returns:
            Dictionary of training history
        """
        # Use default values if not provided
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        # Check feature count compatibility
        actual_feature_count = len(feature_columns)
        expected_feature_count = self.input_shape[2] if len(self.input_shape) == 3 else self.input_shape[1]
        
        # Rebuild model if feature count doesn't match
        if actual_feature_count != expected_feature_count:
            self.logger.info(f"Rebuilding model for {actual_feature_count} features (was expecting {expected_feature_count})")
            self.input_shape = (None, self.lookback_window, actual_feature_count)
            self.model = None  # Clear existing model
            self._build_model()  # Rebuild with correct shape
        
        # Prepare data for training
        X, X_symbol, X_timeframe, y = self.prepare_data(df, symbol, timeframe, feature_columns)
        
        # Set up callbacks for memory management
        if callbacks is None:
            callbacks = []
            
            # Add epoch progress tracking with memory management
            class MemoryManagementCallback(tf.keras.callbacks.Callback):
                def __init__(self, logger):
                    super().__init__()
                    self.logger = logger
                    
                def on_epoch_end(self, epoch, logs=None):
                    import gc
                    gc.collect()
                    if (epoch + 1) % 5 == 0:
                        self.logger.info(f"Memory cleanup after epoch {epoch + 1}")
            
            callbacks.append(MemoryManagementCallback(self.logger))
        
        # Train model with memory management
        try:
            history = self.model.fit(
                [X, X_symbol, X_timeframe], y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store training history
            pair_key = f"{symbol}_{timeframe}"
            self.training_history[pair_key] = history.history
            
            return history.history
            
        except tf.errors.ResourceExhaustedError as e:
            self.logger.error(f"Memory exhausted during training: {e}")
            self.logger.info("Attempting recovery with reduced batch size")
            
            # Retry with reduced batch size
            reduced_batch = max(batch_size // 2, 4)
            return self.train(df, symbol, timeframe, feature_columns, epochs, reduced_batch, validation_split, callbacks)

    def _prepare_data_efficient(self, df: pd.DataFrame, symbol: str, timeframe: str,
                            feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Memory-efficient data preparation"""
        # Extract only needed columns to reduce memory
        df_minimal = df[['close'] + feature_columns].copy()
        
        # Create sequences with numpy for better memory efficiency
        X, y = self._create_sequences_numpy(df_minimal, feature_columns)
        
        # Create symbol and timeframe inputs efficiently
        symbol_id = self._get_symbol_id(symbol)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        X_symbol = np.full((len(X), 1), symbol_id, dtype=np.int32)
        X_timeframe = np.full((len(X), 1), timeframe_minutes, dtype=np.float32)
        
        # Convert to float32 to reduce memory usage
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        return X, X_symbol, X_timeframe, y

    def _create_sequences_numpy(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences using numpy for memory efficiency"""
        # Convert to numpy arrays first to reduce overhead
        features_array = df[feature_columns].values
        close_array = df['close'].values
        
        # Calculate total sequence count
        n_sequences = len(df) - self.lookback_window - self.prediction_horizon + 1
        if n_sequences <= 0:
            raise ValueError("Not enough data for sequences")
        
        # Pre-allocate arrays
        feature_dim = len(feature_columns)
        X = np.zeros((n_sequences, self.lookback_window, feature_dim), dtype=np.float32)
        y = np.zeros((n_sequences, self.prediction_horizon), dtype=np.float32)
        
        # Fill arrays using slicing (much more efficient than appending)
        for i in range(n_sequences):
            X[i] = features_array[i:i+self.lookback_window]
            y[i] = close_array[i+self.lookback_window:i+self.lookback_window+self.prediction_horizon]
        
        return X, y
    
    @profile
    def predict(self, X: np.ndarray, symbol: str = None, timeframe: str = None, **kwargs) -> np.ndarray:
        """Generate predictions with comprehensive type validation and error recovery."""
        # Use instance attributes if parameters not provided
        symbol = symbol or getattr(self, 'symbol', 'default')
        timeframe = timeframe or getattr(self, 'timeframe', '1h')
        self.symbol = symbol  # Store for future use
        self.timeframe = timeframe
        
        # Create symbol and timeframe inputs
        symbol_id = self._get_symbol_id(symbol)
        timeframe_minutes = self._get_timeframe_minutes(timeframe)
        
        # Ensure X has proper type and shape
        try:
            # Apply strict validation and conversion
            X = DataTypeValidator.ensure_numeric_array(X)
            
            # Ensure 3D shape (batch, sequence, features)
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=0)
            
            # Create properly typed inputs
            X_symbol = np.full((X.shape[0], 1), symbol_id, dtype=np.int32)
            X_timeframe = np.full((X.shape[0], 1), timeframe_minutes, dtype=np.float32)
            
            # Generate predictions with error isolation
            predictions = self.model.predict(
                [X, X_symbol, X_timeframe], 
                verbose=kwargs.get('verbose', 0)
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            # Return zero array as fallback with appropriate shape
            output_shape = (X.shape[0], self.prediction_horizon)
            return np.zeros(output_shape, dtype=np.float32)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'lookback_window': self.lookback_window,
            'prediction_horizon': self.prediction_horizon,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model_type': self.model_type,
            'symbol_map': self.symbol_map,
            'next_symbol_id': self.next_symbol_id,
            'training_history': {k: {metric: values[-1] for metric, values in v.items()}
                               for k, v in self.training_history.items()},
            'datetime_saved': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        # Load model
        model_path = os.path.join(path, "model.h5")
        self.model = load_model(model_path)
        
        # Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update attributes
        self.lookback_window = metadata['lookback_window']
        self.prediction_horizon = metadata['prediction_horizon']
        self.hidden_layers = metadata['hidden_layers']
        self.dropout_rate = metadata['dropout_rate']
        self.learning_rate = metadata['learning_rate']
        self.batch_size = metadata['batch_size']
        self.epochs = metadata['epochs']
        self.model_type = metadata['model_type']
        self.symbol_map = metadata['symbol_map']
        self.next_symbol_id = metadata['next_symbol_id']
        
        self.logger.info(f"Loaded model from {path}")


# Add to universal_model.py

class MemoryTrackingCallback(tf.keras.callbacks.Callback):
    """Monitor memory usage during training"""
    def __init__(self, logger=None, check_interval=5):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.epoch_count = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.check_interval == 0:
            self._log_memory_usage("Epoch Begin")
    
    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_count % self.check_interval == 0:
            self._log_memory_usage("Epoch End")
    
    def _log_memory_usage(self, point):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory Usage ({point}): "
                f"{memory_info.rss / 1024 / 1024:.1f} MB RSS, "
                f"{memory_info.vms / 1024 / 1024:.1f} MB Virtual"
            )
            
            # Force garbage collection if memory usage is high
            if memory_info.rss > 4 * 1024 * 1024 * 1024:  # 4GB
                import gc
                gc.collect()
                self.logger.warning("High memory usage detected, forcing garbage collection")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.warning(f"Error monitoring memory: {e}")
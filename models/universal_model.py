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
        self.input_shape = (self.lookback_window, feature_count)  # Add this line to initialize input_shape

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
    
    def _build_model(self) -> None:
        """Build the universal model architecture."""
        # Determine feature count (default to 30 for PCA components)
        feature_count = self.input_shape[-1]
        
        # Price sequence input
        price_input = Input(shape=self.input_shape, name='price_input')
        
        # Symbol embedding input
        symbol_input = Input(shape=(1,), dtype='int32', name='symbol_input')
        symbol_embedding = Embedding(input_dim=100, output_dim=8)(symbol_input)
        symbol_embedding = Flatten()(symbol_embedding)
        
        # Timeframe input (numeric)
        timeframe_input = Input(shape=(1,), name='timeframe_input')
        
        # Process price sequence based on model type
        if self.model_type == 'lstm':
            x = self._build_lstm_layers(price_input)
        elif self.model_type == 'gru':
            x = self._build_gru_layers(price_input)
        elif self.model_type == 'cnn':
            x = self._build_cnn_layers(price_input)
        elif self.model_type == 'transformer':
            x = self._build_transformer_layers(price_input)
        else:
            self.logger.warning(f"Unknown model type: {self.model_type}, using LSTM")
            x = self._build_lstm_layers(price_input)
        
        # Concatenate with symbol and timeframe information
        combined = Concatenate()([x, symbol_embedding, timeframe_input])
        
        # Additional layers for combined processing
        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer for prediction
        outputs = Dense(self.prediction_horizon, activation='linear')(x)
        
        # Create model
        model = Model(inputs=[price_input, symbol_input, timeframe_input], outputs=outputs)
        
        # Compile model with custom loss
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        loss=self._directional_loss,  # Use custom loss
        metrics=['mae', 'mse']
     )
    
        self.model = model
        self.logger.info(f"Built universal model with {self.model_type} architecture and directional loss")

    
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
    
    @profile
    def train(self, df: pd.DataFrame, symbol: str, timeframe: str,
            feature_columns: List[str], epochs: int = None,
            batch_size: int = None, validation_split: float = 0.2,
            callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict[str, List[float]]:
        """
        Train the model on new data.
        
        Args:
            df: DataFrame with OHLCV and features
            symbol: Trading symbol
            timeframe: Trading timeframe
            feature_columns: List of feature column names
            epochs: Number of epochs to train (optional)
            batch_size: Batch size for training (optional)
            validation_split: Proportion of data to use for validation
            callbacks: List of Keras callbacks (optional)
            
        Returns:
            Training history
        """
        # Use configured values if not provided
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        # Create epoch progress callback
        epoch_progress = EpochProgressCallback(self.logger, epochs)
        
        # Check if we need to rebuild the model due to feature count mismatch
        actual_feature_count = len(feature_columns)
        expected_feature_count = self.input_shape[-1]
        
        if actual_feature_count != expected_feature_count:
            self.logger.info(f"Rebuilding model for {actual_feature_count} features (was expecting {expected_feature_count})")
            self.input_shape = (self.lookback_window, actual_feature_count)
            self.model = None  # Clear existing model
            self._build_model()  # Rebuild with correct shape
        
        # Prepare data
        X, X_symbol, X_timeframe, y = self.prepare_data(df, symbol, timeframe, feature_columns)
        
        # Set up callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(monitor='val_loss' if validation_split > 0 else 'loss', 
                            patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss' if validation_split > 0 else 'loss', 
                                factor=0.5, patience=5, min_lr=1e-6),
                epoch_progress
            ]
        else:
            # Add epoch progress to existing callbacks
            callbacks.append(epoch_progress)
        
            # Split into train and validation sets - MODIFY THIS SECTION
        if validation_split > 0:
            split_idx = int(len(X) * (1 - validation_split))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            X_symbol_train, X_symbol_val = X_symbol[:split_idx], X_symbol[split_idx:]
            X_timeframe_train, X_timeframe_val = X_timeframe[:split_idx], X_timeframe[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Use validation data
            validation_data = ([X_val, X_symbol_val, X_timeframe_val], y_val)
        else:
            # No validation split
            X_train, X_symbol_train, X_timeframe_train, y_train = X, X_symbol, X_timeframe, y
            validation_data = None
        
        # Then modify the model.fit call to conditionally include validation_data
        history = self.model.fit(
            [X_train, X_symbol_train, X_timeframe_train], y_train,
            validation_data=validation_data,  # This will be None if validation_split is 0
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        
        # Store training history
        pair_key = f"{symbol}_{timeframe}"
        self.training_history[pair_key] = history.history
        
        self.logger.info(f"Trained on {len(X_train)} samples for {symbol} {timeframe}")
        
        return history.history
    
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
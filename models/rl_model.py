import os
import logging
import json
import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam

class RLModel:
    """
    Reinforcement learning model for trading.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the RL model.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        
        # Extract RL configuration
        self.algorithm = self.config.get('model.rl.algorithm', 'dqn')
        self.gamma = self.config.get('model.rl.gamma', 0.99)
        self.epsilon = self.config.get('model.rl.epsilon', 1.0)
        self.epsilon_decay = self.config.get('model.rl.epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('model.rl.epsilon_min', 0.1)
        self.memory_size = self.config.get('model.rl.memory_size', 10000)
        self.target_update_freq = self.config.get('model.rl.target_update_freq', 100)
        self.batch_size = self.config.get('model.batch_size', 32)
        
        # State and action dimensions
        self.state_dim = None
        self.action_dim = 3  # Buy, hold, sell
        
        # Memory for experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Models
        self.model = None
        self.target_model = None
        
        # Training counters
        self.train_step_counter = 0
    
    def build_model(self, state_dim: int) -> None:
        """
        Build the RL model.
        
        Args:
            state_dim: Dimension of the state space
        """
        self.state_dim = state_dim
        
        if self.algorithm == 'dqn':
            self._build_dqn_model()
        elif self.algorithm == 'dueling_dqn':
            self._build_dueling_dqn_model()
        else:
            self.logger.warning(f"Unknown RL algorithm: {self.algorithm}, using DQN")
            self._build_dqn_model()
    
    def _build_dqn_model(self) -> None:
        """Build a Deep Q-Network model."""
        # Create model
        inputs = Input(shape=(self.state_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.model = model
        
        # Create target model
        target_model = Model(inputs=inputs, outputs=outputs)
        target_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        target_model.set_weights(model.get_weights())
        
        self.target_model = target_model
        
        self.logger.info("Built DQN model")
    
    def _build_dueling_dqn_model(self) -> None:
        """Build a Dueling Deep Q-Network model."""
        # Create model
        inputs = Input(shape=(self.state_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Value stream
        value = Dense(32, activation='relu')(x)
        value = Dense(1)(value)
        
        # Advantage stream
        advantage = Dense(32, activation='relu')(x)
        advantage = Dense(self.action_dim)(advantage)
        
        # Combine streams
        outputs = Lambda(
            lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
            output_shape=(self.action_dim,)
        )([value, advantage])
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.model = model
        
        # Create target model
        target_model = Model(inputs=inputs, outputs=outputs)
        target_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        target_model.set_weights(model.get_weights())
        
        self.target_model = target_model
        
        self.logger.info("Built Dueling DQN model")
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether the model is in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_dim)
        
        # Exploitation: best action from Q-network
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self) -> float:
        """
        Train the model using experience replay.
        
        Returns:
            Training loss
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare arrays
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Get current Q values
        targets = self.model.predict(states, verbose=0)
        
        # Get next Q values from target model
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update targets for actions taken
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target model
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory
        os.makedirs(path, exist_ok=True)
        
        # Save models
        self.model.save(os.path.join(path, "model.h5"))
        self.target_model.save(os.path.join(path, "target_model.h5"))
        
        # Save metadata
        metadata = {
            'algorithm': self.algorithm,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'memory_size': self.memory_size,
            'target_update_freq': self.target_update_freq,
            'batch_size': self.batch_size,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'train_step_counter': self.train_step_counter,
            'datetime_saved': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved RL model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        # Load models
        self.model = load_model(os.path.join(path, "model.h5"))
        self.target_model = load_model(os.path.join(path, "target_model.h5"))
        
        # Load metadata
        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update attributes
        self.algorithm = metadata['algorithm']
        self.gamma = metadata['gamma']
        self.epsilon = metadata['epsilon']
        self.epsilon_decay = metadata['epsilon_decay']
        self.epsilon_min = metadata['epsilon_min']
        self.memory_size = metadata['memory_size']
        self.target_update_freq = metadata['target_update_freq']
        self.batch_size = metadata['batch_size']
        self.state_dim = metadata['state_dim']
        self.action_dim = metadata['action_dim']
        self.train_step_counter = metadata['train_step_counter']
        
        self.logger.info(f"Loaded RL model from {path}")
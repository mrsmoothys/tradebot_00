import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import itertools
import json
import os
import time
from functools import partial

from backtest.backtest_engine import BacktestEngine
from models.universal_model import UniversalModel
from models.rl_model import RLModel

class HyperparameterTuner:
    """
    Tunes hyperparameters for trading models and strategies.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.optimization_config = self.config.get('optimization', {})
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(config_manager)
        
        # Optimization parameters
        self.method = self.optimization_config.get('method', 'grid_search')
        self.metric_to_optimize = self.optimization_config.get('metric', 'sharpe_ratio')
        self.maximize = self.optimization_config.get('maximize', True)
        self.max_trials = self.optimization_config.get('max_trials', 50)
        
        # Results tracking
        self.results = []
        self.best_params = None
        self.best_score = float('-inf') if self.maximize else float('inf')
    
    def _evaluate_params(self, params: Dict[str, Any], symbol: str, timeframe: str, 
                      data: pd.DataFrame) -> float:
        """
        Evaluate a set of hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters to test
            symbol: Symbol to test on
            timeframe: Timeframe to test on
            data: Market data
            
        Returns:
            Performance metric value
        """
        # Apply parameters to configuration
        for param_path, value in params.items():
            self.config.set(param_path, value)
        
        # Initialize model based on current config
        model_type = self.config.get('model.architecture', 'lstm')
        if model_type == 'rl':
            model = RLModel(self.config)
            feature_count = data.shape[1] - 5  # Subtract OHLCV columns
            model.build_model(feature_count)
        else:
            model = UniversalModel(self.config)
        
        # Run backtest with current parameters
        results = self.backtest_engine.run_backtest(
            symbols=[symbol],
            timeframes=[timeframe],
            model=model
        )
        
        # Extract performance metric
        result = results[symbol][timeframe]
        metric_value = result['performance_metrics'].get(self.metric_to_optimize, 0)
        
        # For metrics that should be minimized
        if not self.maximize:
            metric_value = -metric_value
        
        return metric_value
    
    def grid_search(self, param_grid: Dict[str, List], symbol: str, timeframe: str,
                   data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            param_grid: Dictionary of parameter names and possible values
            symbol: Symbol to test on
            timeframe: Timeframe to test on
            data: Market data
            
        Returns:
            Dictionary of best parameters
        """
        # Generate all possible combinations
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        results = []
        
        # Test each combination
        for i, combination in enumerate(param_combinations):
            current_params = dict(zip(param_keys, combination))
            
            # Log progress
            self.logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {current_params}")
            
            # Evaluate parameters
            score = self._evaluate_params(current_params, symbol, timeframe, data)
            
            # Store result
            result = {
                'params': current_params,
                'score': score,
                'trial': i
            }
            results.append(result)
            
            # Update best if improved
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = current_params
                self.logger.info(f"New best: {best_score} with {best_params}")
        
        # Store results
        self.results = results
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def random_search(self, param_space: Dict[str, Tuple], symbol: str, timeframe: str,
                     data: pd.DataFrame, n_trials: int = None) -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            param_space: Dictionary of parameter names and ranges
            symbol: Symbol to test on
            timeframe: Timeframe to test on
            data: Market data
            n_trials: Number of random trials (default: use max_trials from config)
            
        Returns:
            Dictionary of best parameters
        """
        n_trials = n_trials or self.max_trials
        
        self.logger.info(f"Starting random search with {n_trials} trials")
        
        best_score = float('-inf') if self.maximize else float('inf')
        best_params = None
        results = []
        
        # Run random trials
        for i in range(n_trials):
            # Generate random parameters
            current_params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range[0], int):
                    # Integer parameter
                    current_params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                elif isinstance(param_range[0], float):
                    # Float parameter
                    current_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Categorical parameter
                    current_params[param_name] = np.random.choice(param_range)
            
            # Log progress
            self.logger.info(f"Trial {i+1}/{n_trials}: {current_params}")
            
            # Evaluate parameters
            score = self._evaluate_params(current_params, symbol, timeframe, data)
            
            # Store result
            result = {
                'params': current_params,
                'score': score,
                'trial': i
            }
            results.append(result)
            
            # Update best if improved
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_score = score
                best_params = current_params
                self.logger.info(f"New best: {best_score} with {best_params}")
        
        # Store results
        self.results = results
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def bayesian_optimization(self, param_space: Dict[str, Tuple], symbol: str, timeframe: str,
                           data: pd.DataFrame, n_trials: int = None) -> Dict[str, Any]:
        """
        Perform Bayesian optimization for hyperparameters.
        
        Args:
            param_space: Dictionary of parameter names and ranges
            symbol: Symbol to test on
            timeframe: Timeframe to test on
            data: Market data
            n_trials: Number of trials (default: use max_trials from config)
            
        Returns:
            Dictionary of best parameters
        """
        # Try to import optional dependency
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            self.logger.error("scikit-optimize not installed. Please install with pip install scikit-optimize")
            return self.random_search(param_space, symbol, timeframe, data, n_trials)
        
        n_trials = n_trials or self.max_trials
        
        self.logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        # Convert param_space to skopt space
        space = []
        param_names = []
        
        for param_name, param_range in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_range[0], int):
                space.append(Integer(param_range[0], param_range[1], name=param_name))
            elif isinstance(param_range[0], float):
                space.append(Real(param_range[0], param_range[1], name=param_name))
            elif isinstance(param_range, list):
                space.append(Categorical(param_range, name=param_name))
        
        # Define objective function
        @use_named_args(space)
        def objective(**params):
            # Convert to dictionary for _evaluate_params
            param_dict = {}
            for param_name, value in params.items():
                param_dict[param_name] = value
            
            # Evaluate parameters
            score = self._evaluate_params(param_dict, symbol, timeframe, data)
            
            # Bayesian optimization minimizes, so negate if maximizing
            return -score if self.maximize else score
        
        # Run optimization
        results = []
        
        def callback(res):
            # Extract parameters
            params = {}
            for i, param_name in enumerate(param_names):
                params[param_name] = res.x_iters[-1][i]
            
            # Calculate true score (not negated)
            score = -res.func_vals[-1] if self.maximize else res.func_vals[-1]
            
            # Store result
            result = {
                'params': params,
                'score': score,
                'trial': len(res.x_iters) - 1
            }
            results.append(result)
            
            # Log progress
            self.logger.info(f"Trial {len(res.x_iters)}/{n_trials}: {params}, Score: {score}")
        
        res = gp_minimize(
            objective,
            space,
            n_calls=n_trials,
            random_state=42,
            callback=callback
        )
        
        # Extract best parameters
        best_params = {}
        for i, param_name in enumerate(param_names):
            best_params[param_name] = res.x[i]
        
        # Calculate true best score (not negated)
        best_score = -res.fun if self.maximize else res.fun
        
        # Store results
        self.results = results
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params
    
    def optimize(self, param_space: Dict[str, Any], symbol: str, timeframe: str,
               data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the selected method.
        
        Args:
            param_space: Parameter space to search
            symbol: Symbol to test on
            timeframe: Timeframe to test on
            data: Market data (optional, will be loaded if not provided)
            
        Returns:
            Dictionary of best parameters
        """
        # Load data if not provided
        if data is None:
            data = self.backtest_engine.load_data([symbol], [timeframe])[symbol][timeframe]
        
        # Choose optimization method
        if self.method == 'grid_search':
            best_params = self.grid_search(param_space, symbol, timeframe, data)
        elif self.method == 'random_search':
            best_params = self.random_search(param_space, symbol, timeframe, data)
        elif self.method == 'bayesian':
            best_params = self.bayesian_optimization(param_space, symbol, timeframe, data)
        else:
            self.logger.warning(f"Unknown optimization method: {self.method}, using grid search")
            best_params = self.grid_search(param_space, symbol, timeframe, data)
        
        # Apply best parameters to config
        for param_path, value in best_params.items():
            self.config.set(param_path, value)
        
        return best_params
    
    def save_results(self, path: str) -> None:
        """
        Save optimization results to disk.
        
        Args:
            path: Path to save results
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create results dictionary
        results_dict = {
            'method': self.method,
            'metric': self.metric_to_optimize,
            'maximize': self.maximize,
            'trials': len(self.results),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Saved optimization results to {path}")
    
    def load_results(self, path: str) -> Dict[str, Any]:
        """
        Load optimization results from disk.
        
        Args:
            path: Path to load results from
            
        Returns:
            Dictionary of optimization results
        """
        with open(path, 'r') as f:
            results_dict = json.load(f)
        
        # Update instance variables
        self.method = results_dict['method']
        self.metric_to_optimize = results_dict['metric']
        self.maximize = results_dict['maximize']
        self.results = results_dict['all_results']
        self.best_score = results_dict['best_score']
        self.best_params = results_dict['best_params']
        
        self.logger.info(f"Loaded optimization results from {path}")
        
        return results_dict
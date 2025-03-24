import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt
from utils.profiling import profile

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.feature_selection import FeatureSelector
from models.universal_model import UniversalModel
from models.rl_model import RLModel
from strategy.ml_strategy import MLStrategy
from strategy.risk_management import RiskManager
from backtest.performance_metrics import PerformanceCalculator
from utils.progress import ProgressTracker

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the backtest engine.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.backtest_config = self.config.get('backtest', {})
        
        # Data components
        self.data_loader = DataLoader(config_manager)
        self.feature_engineer = FeatureEngineer(config_manager)
        self.feature_selector = FeatureSelector(config_manager)
        
        # Risk manager
        self.risk_manager = RiskManager(self.config.get('risk', {}))
        
        # Performance calculator
        self.performance_calculator = PerformanceCalculator()
        
        # Backtest parameters
        self.initial_capital = self.backtest_config.get('initial_capital', 10000.0)
        self.fee_rate = self.backtest_config.get('fee_rate', 0.0002)  # 0.02%
        self.slippage = self.backtest_config.get('slippage', 0.002)   # 0.2%
        
        # Tracking variables
        self.strategies = {}
        self.results = {}
        
        self.logger.info("Initialized backtest engine")
    
    def load_data(self, symbols: List[str], timeframes: List[str], 
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load and prepare data for backtesting.
        
        Args:
            symbols: List of symbols to test
            timeframes: List of timeframes to test
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Nested dictionary of DataFrames by symbol and timeframe
        """
        data = {}
        
        for symbol in symbols:
            data[symbol] = {}
            
            for timeframe in timeframes:
                # Load raw data
                try:
                    df = self.data_loader.load_data(symbol, timeframe, start_date, end_date)
                    
                    # Generate features
                    df = self.feature_engineer.generate_features(df)
                    
                    # Select relevant features
                    df, selected_features = self.feature_selector.select_features(df)
                    
                    # Apply dimensionality reduction if needed
                    if self.config.get('features.pca.use_pca', False):
                        n_components = self.config.get('features.pca.n_components', 30)
                        df, _, _ = self.feature_engineer.apply_pca(df, n_components)
                    
                    data[symbol][timeframe] = df
                    self.logger.info(f"Prepared data for {symbol} {timeframe} with shape {df.shape}")
                except Exception as e:
                    self.logger.error(f"Error preparing data for {symbol} {timeframe}: {e}")
        
        return data
    
    # In backtest/backtest_engine.py, initialize_strategy method
    def initialize_strategy(self, strategy_type: str, symbol: str, timeframe: str, 
                        model: Any, data: pd.DataFrame) -> Any:
        """
        Initialize a trading strategy.
        """
        if strategy_type == 'ml':
            # Initialize ML strategy
            lookback_window = self.config.get('model.lookback_window', 60)
            prediction_horizon = self.config.get('model.prediction_horizon', 5)
            strategy_config = self.config.get('strategy', {})
            
            strategy = MLStrategy(
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                lookback_window=lookback_window,
                prediction_horizon=prediction_horizon,
                config=strategy_config,
                risk_manager=self.risk_manager
            )
            
            # Explicitly ensure symbol and timeframe attributes are set
            strategy.symbol = symbol
            strategy.timeframe = timeframe
            
            return strategy
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
    
    @profile
    def run_backtest(self, symbols: List[str], timeframes: List[str], 
                start_date: Optional[str] = None, end_date: Optional[str] = None,
                model: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run backtest for multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols to test
            timeframes: List of timeframes to test
            start_date: Start date for backtest
            end_date: End date for backtest
            model: Trained model (optional)
            
        Returns:
            Dictionary of backtest results by symbol and timeframe
        """
        # Calculate total number of symbol/timeframe combinations
        total_combinations = len(symbols) * len(timeframes)

        # Create progress tracker
        progress = ProgressTracker(
            name="Overall Backtest", 
            total_steps=total_combinations,
            log_interval=1,
            logger=self.logger
        )

        # Load and prepare data
        data = self.load_data(symbols, timeframes, start_date, end_date)
        
        # Get strategy type
        strategy_type = self.config.get('strategy.type', 'ml')
        
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            
            for timeframe in timeframes:
                if symbol not in data or timeframe not in data[symbol]:
                    self.logger.warning(f"Skipping backtest for {symbol} {timeframe}: No data")
                    continue
                
                df = data[symbol][timeframe]
                
                # Skip if insufficient data
                if len(df) < 100:
                    self.logger.warning(f"Skipping backtest for {symbol} {timeframe}: Insufficient data")
                    continue
                
                # Initialize or load model
                if model is None:
                    # Create model based on configuration
                    model_type = self.config.get('model.architecture', 'lstm')
                    if model_type == 'rl':
                        model = RLModel(self.config)
                        model.build_model(df.shape[1] - 5)  # Subtract OHLCV columns
                    else:
                        model = UniversalModel(self.config)
                
                # If model is UniversalModel, set symbol and timeframe
                if hasattr(model, 'symbol') and hasattr(model, 'timeframe'):
                    model.symbol = symbol
                    model.timeframe = timeframe

                # Initialize strategy
                strategy = self.initialize_strategy(
                    strategy_type=strategy_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    model=model,
                    data=df
                )
                
                # Run backtest
                self.logger.info(f"Running backtest for {symbol} {timeframe}")
                
                # Set initial capital and fee
                strategy.initial_capital = self.initial_capital
                strategy.capital = self.initial_capital
                strategy.fee_rate = self.fee_rate
                
                # Ensure symbol and timeframe are set
                strategy.symbol = symbol
                strategy.timeframe = timeframe
                
                # Run backtest
                backtest_results = strategy.backtest(df)
                
                # Verify backtest results integrity
                self._verify_backtest_results(symbol, timeframe, backtest_results)
                
                # Debug trades
                trades = backtest_results.get('trades', [])
                if trades:
                    self.logger.info(f"Received {len(trades)} trades from strategy")
                    # Log sample of first and last trade
                    if len(trades) > 0:
                        self.logger.info(f"First trade: {trades[0]}")
                        self.logger.info(f"Last trade: {trades[-1]}")
                    
                    # Verify trade data integrity
                    self._verify_trades_for_performance(trades)
                else:
                    self.logger.warning(f"No trades generated for {symbol} {timeframe}")

                # Debug equity curve
                equity_curve = backtest_results.get('equity_curve', [])
                if equity_curve:
                    self.logger.info(f"Equity curve starts at {equity_curve[0]:.2f} and ends at {equity_curve[-1]:.2f}")
                    self.logger.info(f"Return from equity curve: {(equity_curve[-1]/equity_curve[0] - 1)*100:.2f}%")
                else:
                    self.logger.warning(f"No equity curve generated for {symbol} {timeframe}")


                # Save strategy instance
                strategy_key = f"{symbol}_{timeframe}"
                self.strategies[strategy_key] = strategy
                
                # Calculate detailed performance metrics
                equity_curve = backtest_results.get('equity_curve', [])
                trades = backtest_results.get('trades', [])
                
                if not equity_curve:
                    self.logger.error(f"Missing equity curve for {symbol} {timeframe}")
                    equity_curve = [self.initial_capital] * (len(df) + 1)  # Default flat equity curve
                
                # Log some details about trades for debugging
                if trades:
                    profitable_trades = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
                    total_profit = sum(t.get('net_profit', 0) for t in trades)
                    self.logger.info(f"Backtest generated {len(trades)} trades with {profitable_trades} profitable trades")
                    self.logger.info(f"Total profit from trades: ${total_profit:.2f}")
                else:
                    self.logger.warning(f"No trades generated for {symbol} {timeframe}")
                
                # Calculate performance metrics
                performance = self.performance_calculator.calculate_metrics(
                    equity_curve=equity_curve,
                    trades=trades,
                    benchmark_returns=None
                )
                
                # Update results with performance metrics
                backtest_results['performance_metrics'] = performance
                
                # Save results
                results[symbol][timeframe] = backtest_results
                self.results[strategy_key] = backtest_results
                
                self.logger.info(f"Completed backtest for {symbol} {timeframe}")
                self.logger.info(f"Total return: {performance['total_return']:.2f}%")
                self.logger.info(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")
                self.logger.info(f"Max drawdown: {performance['max_drawdown']:.2f}%")
                
                # Update progress
                progress.update(1, f"{symbol} {timeframe}")
        
        # Mark overall backtest as complete
        progress.complete()
        
        return results
    
    # In backtest_engine.py, add this before calculating performance metrics
    def _verify_trades_for_performance(self, trades):
        """Verify trades are in the correct format for performance calculation."""
        if not trades:
            self.logger.warning("No trades to verify for performance calculation")
            return

        # Check required fields
        required_fields = ['profit_pct', 'profit_amount', 'net_profit', 'entry_time', 'exit_time']
        for field in required_fields:
            if field not in trades[0]:
                self.logger.error(f"Trade data missing required field: {field}")
                
        # Log trade summary for debugging
        total_profit = sum(t.get('net_profit', 0) for t in trades)
        win_count = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
        self.logger.info(f"Trade summary: {len(trades)} trades, {win_count} wins, total profit: ${total_profit:.2f}")
        
        # Convert datetime objects to strings if needed
        for trade in trades:
            for time_field in ['entry_time', 'exit_time']:
                if time_field in trade and not isinstance(trade[time_field], str):
                    trade[time_field] = str(trade[time_field])

    def _verify_backtest_results(self, symbol: str, timeframe: str, results: Dict[str, Any]) -> None:
        """
        Verify the integrity of backtest results and log relevant information.
        
        Args:
            symbol: Trading symbol
            timeframe: Trading timeframe
            results: Backtest results dictionary
        """
        # Check for required components
        if 'equity_curve' not in results:
            self.logger.error(f"Missing equity curve in backtest results for {symbol} {timeframe}")
        elif not results['equity_curve']:
            self.logger.error(f"Empty equity curve in backtest results for {symbol} {timeframe}")
        else:
            equity_curve = results['equity_curve']
            self.logger.info(f"Equity curve starts at {equity_curve[0]:.2f} and ends at {equity_curve[-1]:.2f}")
            self.logger.info(f"Equity curve contains {len(equity_curve)} data points")
            
            # Check for flat equity curve
            if len(equity_curve) > 1 and all(v == equity_curve[0] for v in equity_curve):
                self.logger.warning(f"Equity curve for {symbol} {timeframe} is flat (no P&L changes)")
        
        # Check trades
        if 'trades' not in results:
            self.logger.error(f"Missing trades in backtest results for {symbol} {timeframe}")
        elif not results['trades']:
            self.logger.warning(f"No trades generated during backtest for {symbol} {timeframe}")
        else:
            trades = results['trades']
            self.logger.info(f"Generated {len(trades)} trades for {symbol} {timeframe}")
            
            # Verify trade data
            if len(trades) > 0:
                # Sample first and last trade
                first_trade = trades[0]
                last_trade = trades[-1]
                
                # Check for required trade fields
                required_fields = ['entry_price', 'exit_price', 'profit_pct', 'net_profit']
                missing_fields = [field for field in required_fields 
                                if field not in first_trade or first_trade[field] is None]
                
                if missing_fields:
                    self.logger.error(f"Trades missing required fields: {missing_fields}")
                
                # Log trade samples
                self.logger.info(f"First trade: Entry={first_trade.get('entry_price')}, "
                            f"Exit={first_trade.get('exit_price')}, "
                            f"P&L={first_trade.get('profit_pct', 0):.2f}%")
                
                self.logger.info(f"Last trade: Entry={last_trade.get('entry_price')}, "
                            f"Exit={last_trade.get('exit_price')}, "
                            f"P&L={last_trade.get('profit_pct', 0):.2f}%")
                
                # Verify profit calculation
                total_profit = sum(t.get('net_profit', 0) for t in trades)
                self.logger.info(f"Total profit from all trades: ${total_profit:.2f}")
    
    def compare_strategies(self, strategies: List[str]) -> pd.DataFrame:
        """
        Compare performance metrics of different strategies.
        
        Args:
            strategies: List of strategy keys to compare
            
        Returns:
            DataFrame with performance metrics
        """
        comparison = []
        
        for strategy_key in strategies:
            if strategy_key not in self.results:
                self.logger.warning(f"Strategy {strategy_key} not found in results")
                continue
            
            result = self.results[strategy_key]
            performance = result['performance_metrics']
            
            row = {
                'strategy': strategy_key,
                'total_return': performance['total_return'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'sortino_ratio': performance['sortino_ratio'],
                'max_drawdown': performance['max_drawdown'],
                'win_rate': performance['win_rate'],
                'profit_factor': performance['profit_factor'],
                'num_trades': performance['num_trades']
            }
            
            comparison.append(row)
        
        if not comparison:
            return pd.DataFrame()
        
        return pd.DataFrame(comparison)
    
    def plot_equity_curves(self, strategies: List[str], figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot equity curves for multiple strategies.
        
        Args:
            strategies: List of strategy keys to plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        for strategy_key in strategies:
            if strategy_key not in self.results:
                self.logger.warning(f"Strategy {strategy_key} not found in results")
                continue
            
            result = self.results[strategy_key]
            equity_curve = result['equity_curve']
            
            plt.plot(equity_curve, label=strategy_key)
        
        plt.title('Equity Curves')
        plt.xlabel('Bar')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    # Add this in your BacktestEngine class
    def _log_backtest_results(self, symbol: str, timeframe: str, results: dict):
        """Log detailed backtest results for debugging"""
        self.logger.info(f"Backtest results for {symbol} {timeframe}:")
        
        if 'equity_curve' in results:
            eq = results['equity_curve']
            self.logger.info(f"  Equity curve: start={eq[0]}, end={eq[-1]}, len={len(eq)}")
            if len(eq) > 1:
                self.logger.info(f"  Return: {(eq[-1]/eq[0] - 1) * 100:.2f}%")
        else:
            self.logger.warning("  Missing equity curve")
        
        if 'trades' in results:
            trades = results['trades']
            self.logger.info(f"  Trades: {len(trades)}")
            if trades:
                profit_sum = sum(t.get('net_profit', 0) for t in trades)
                win_count = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
                self.logger.info(f"  Total profit: ${profit_sum:.2f}")
                self.logger.info(f"  Win rate: {win_count/len(trades)*100:.2f}%")
        else:
            self.logger.warning("  Missing trades")

    def save_results(self, path: str) -> None:
        """
        Save backtest results to disk.
        
        Args:
            path: Path to save results
        """
        os.makedirs(path, exist_ok=True)
        
        for strategy_key, result in self.results.items():
            # Save performance metrics
            performance = result['performance_metrics']
            performance_path = os.path.join(path, f"{strategy_key}_performance.csv")
            pd.DataFrame([performance]).to_csv(performance_path, index=False)
            
            # Save trades
            trades = result['trades']
            trades_path = os.path.join(path, f"{strategy_key}_trades.csv")
            pd.DataFrame(trades).to_csv(trades_path, index=False)
            
            # Save equity curve
            equity_curve = result['equity_curve']
            equity_path = os.path.join(path, f"{strategy_key}_equity.csv")
            pd.DataFrame({'equity': equity_curve}).to_csv(equity_path, index=False)
        
        self.logger.info(f"Saved backtest results to {path}")
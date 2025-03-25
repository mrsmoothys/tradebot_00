import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

class PerformanceCalculator:
    """
    Calculates trading strategy performance metrics.
    """
    
    def __init__(self):
        """Initialize the performance calculator."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_metrics(self, equity_curve: List[float], trades: List[Dict[str, Any]], 
                   benchmark_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: List of equity values over time
            trades: List of trade dictionaries
            benchmark_returns: List of benchmark returns (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Basic metrics
        if not equity_curve or len(equity_curve) < 2:
            self.logger.warning("Empty or insufficient equity curve, returning minimal metrics")
            return {
                'initial_capital': equity_curve[0] if equity_curve else 0,
                'final_capital': equity_curve[-1] if equity_curve else 0,
                'total_return': 0.0,
                'total_profit': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': len(trades) if trades else 0
            }
        
        # Extract initial and final values
        metrics['initial_capital'] = float(equity_curve[0])
        metrics['final_capital'] = float(equity_curve[-1])
        metrics['total_return'] = float((metrics['final_capital'] / metrics['initial_capital'] - 1) * 100)
        metrics['total_profit'] = float(metrics['final_capital'] - metrics['initial_capital'])
        
        # Calculate returns
        returns = self._calculate_returns(equity_curve)
        metrics['annualized_return'] = self._annualize_return(returns)
        metrics['volatility'] = self._calculate_volatility(returns)
        metrics['annualized_volatility'] = self._annualize_volatility(returns)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        
        # Drawdown metrics
        max_dd_pct = self._calculate_max_drawdown(equity_curve)
        max_dd_duration = self._calculate_max_drawdown_duration(equity_curve)
        metrics['max_drawdown'] = float(max_dd_pct)
        metrics['max_drawdown_duration'] = int(max_dd_duration)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns, max_dd_pct)
        
        # Trade metrics if available
        if trades and len(trades) > 0:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
            
            # Verify trade metrics are calculated correctly
            self.logger.info(f"Calculated trade metrics: {len(trades)} trades, "
                        f"Win rate: {trade_metrics.get('win_rate', 0):.2f}%, "
                        f"Profit factor: {trade_metrics.get('profit_factor', 0):.2f}")
        else:
            self.logger.warning("No trades provided for metrics calculation")
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
        
        # Benchmark comparison if provided
        if benchmark_returns and len(benchmark_returns) > 0:
            benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_returns)
            metrics.update(benchmark_metrics)
        
        # Ensure all values are proper numeric types
        for key in metrics:
            if isinstance(metrics[key], (np.float32, np.float64, np.int32, np.int64)):
                metrics[key] = float(metrics[key])
        
        return metrics
    
    def _calculate_returns(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate period-to-period returns from equity curve.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            List of returns
        """
        returns = []
        for i in range(1, len(equity_curve)):
            returns.append(equity_curve[i] / equity_curve[i-1] - 1)
        
        return returns
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Maximum drawdown as a percentage
        """
        if not equity_curve:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max * 100
        
        return np.max(drawdown)
    
    def _calculate_max_drawdown_duration(self, equity_curve: List[float]) -> int:
        """
        Calculate maximum drawdown duration in periods.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Maximum drawdown duration in periods
        """
        if not equity_curve:
            return 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        
        # Find periods where drawdown is occurring
        is_drawdown = drawdown > 0
        
        if not any(is_drawdown):
            return 0
        
        # Calculate drawdown durations
        durations = []
        current_duration = 0
        
        for i in range(len(equity_curve)):
            if is_drawdown[i]:
                current_duration += 1
                if i == len(equity_curve) - 1 or (i < len(equity_curve) - 1 and not is_drawdown[i+1]):
                    durations.append(current_duration)
                    current_duration = 0
        
        return max(durations) if durations else 0
    
    def _annualize_return(self, returns: List[float], periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.
        
        Args:
            returns: List of period returns
            periods_per_year: Number of periods in a year
            
        Returns:
            Annualized return
        """
        if not returns:
            return 0.0
        
        return (np.mean(returns) + 1) ** periods_per_year - 1
    
    def _calculate_volatility(self, returns: List[float]) -> float:
        """
        Calculate return volatility.
        
        Args:
            returns: List of period returns
            
        Returns:
            Volatility (standard deviation of returns)
        """
        if not returns:
            return 0.0
        
        return np.std(returns)
    
    def _annualize_volatility(self, returns: List[float], periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: List of period returns
            periods_per_year: Number of periods in a year
            
        Returns:
            Annualized volatility
        """
        if not returns:
            return 0.0
        
        return np.std(returns) * np.sqrt(periods_per_year)
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0, 
                             periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods in a year
            
        Returns:
            Sharpe ratio
        """
        if not returns or np.std(returns) == 0:
            return 0.0
        
        excess_return = np.mean(returns) - risk_free_rate / periods_per_year
        return excess_return / np.std(returns) * np.sqrt(periods_per_year)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0,
                               periods_per_year: int = 252, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (annualized)
            periods_per_year: Number of periods in a year
            target_return: Target return (default: 0)
            
        Returns:
            Sortino ratio
        """
        if not returns:
            return 0.0
        
        # Calculate downside returns (below target)
        downside_returns = [r for r in returns if r < target_return]
        
        if not downside_returns or np.std(downside_returns) == 0:
            return float('inf') if np.mean(returns) > target_return else 0.0
        
        excess_return = np.mean(returns) - risk_free_rate / periods_per_year
        downside_risk = np.std(downside_returns)
        
        return excess_return / downside_risk * np.sqrt(periods_per_year)
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float, 
                             periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns: List of period returns
            max_drawdown: Maximum drawdown as a percentage
            periods_per_year: Number of periods in a year
            
        Returns:
            Calmar ratio
        """
        if not returns or max_drawdown == 0:
            return 0.0
        
        annualized_return = self._annualize_return(returns, periods_per_year)
        return annualized_return / (max_drawdown / 100)
    
    # In performance_metrics.py, update the calculate_metrics function
    def calculate_metrics(self, equity_curve: List[float], trades: List[Dict[str, Any]], 
                    benchmark_returns: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            equity_curve: List of equity values over time
            trades: List of trade dictionaries
            benchmark_returns: List of benchmark returns (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Ensure we have valid equity curve data
        if not equity_curve or len(equity_curve) < 2:
            self.logger.warning("Empty or insufficient equity curve, returning minimal metrics")
            return {
                'initial_capital': equity_curve[0] if equity_curve else 0,
                'final_capital': equity_curve[-1] if equity_curve else 0,
                'total_return': 0.0,
                'total_profit': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': len(trades) if trades else 0
            }
        
        # Extract initial and final values
        metrics['initial_capital'] = float(equity_curve[0])
        metrics['final_capital'] = float(equity_curve[-1])
        metrics['total_return'] = float((metrics['final_capital'] / metrics['initial_capital'] - 1) * 100)
        metrics['total_profit'] = float(metrics['final_capital'] - metrics['initial_capital'])
        
        # Calculate returns
        returns = self._calculate_returns(equity_curve)
        
        # Basic returns metrics (check for empty returns)
        if returns:
            metrics['annualized_return'] = self._annualize_return(returns)
            metrics['volatility'] = self._calculate_volatility(returns)
            metrics['annualized_volatility'] = self._annualize_volatility(returns)
            metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
        else:
            metrics['annualized_return'] = 0.0
            metrics['volatility'] = 0.0
            metrics['annualized_volatility'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
        
        # Drawdown metrics
        max_dd_pct = self._calculate_max_drawdown(equity_curve)
        max_dd_duration = self._calculate_max_drawdown_duration(equity_curve)
        metrics['max_drawdown'] = float(max_dd_pct)
        metrics['max_drawdown_duration'] = int(max_dd_duration)
        metrics['calmar_ratio'] = self._calculate_calmar_ratio(returns, max_dd_pct)
        
        # Log trade stats
        if trades:
            self.logger.info(f"Processing {len(trades)} trades for metrics")
        else:
            self.logger.warning("No trades to process for metrics")
            
        # Trade metrics if available
        if trades and len(trades) > 0:
            # Check for any None values in profit_pct
            if any(t.get('profit_pct') is None for t in trades):
                self.logger.warning("Some trades have None profit_pct values")
                # Fix None values
                for trade in trades:
                    if trade.get('profit_pct') is None:
                        trade['profit_pct'] = 0.0
            
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.update(trade_metrics)
            
            # Verify trade metrics are calculated correctly
            self.logger.info(f"Calculated trade metrics: {len(trades)} trades, "
                        f"Win rate: {trade_metrics.get('win_rate', 0):.2f}%, "
                        f"Profit factor: {trade_metrics.get('profit_factor', 0):.2f}")
        else:
            self.logger.warning("No trades provided for metrics calculation")
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['profit_factor'] = 0.0
        
        # Ensure all values are proper numeric types
        for key in metrics:
            if isinstance(metrics[key], (np.float32, np.float64, np.int32, np.int64)):
                metrics[key] = float(metrics[key])
        
        return metrics
    
    def calculate_performance_metrics(self, 
                                equity_curve: np.ndarray, 
                                initial_capital: float,
                                data: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate performance metrics with optimized memory usage."""
        # Convert to numpy array if not already
        if isinstance(equity_curve, list):
            equity_array = np.array(equity_curve, dtype=np.float64)
        else:
            equity_array = equity_curve
        
        # Basic metrics
        final_capital = float(equity_array[-1])
        total_return = (final_capital / initial_capital - 1) * 100
        
        # Convert equity to returns efficiently
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate key metrics
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino - only negative returns
            neg_returns = returns[returns < 0]
            sortino_ratio = np.mean(returns) / np.std(neg_returns) * np.sqrt(252) if len(neg_returns) > 0 and np.std(neg_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calculate drawdown efficiently
        peaks = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - peaks) / peaks * 100
        max_drawdown = float(abs(np.min(drawdowns)))
        
        # Trade-related metrics
        num_trades = len(self.trades)
        
        if num_trades > 0:
            # Vectorized calculations
            profits = np.array([t.get('profit_pct', 0) for t in self.trades])
            win_mask = profits > 0
            
            win_rate = np.sum(win_mask) / num_trades * 100
            
            # Win/loss stats
            win_profits = profits[win_mask] if any(win_mask) else np.array([0])
            loss_profits = profits[~win_mask] if any(~win_mask) else np.array([0])
            
            avg_win = float(np.mean(win_profits)) if any(win_mask) else 0
            avg_loss = float(np.mean(loss_profits)) if any(~win_mask) else 0
            
            # Calculate profit factor efficiently
            gross_profit = float(np.sum(win_profits)) if any(win_mask) else 0
            gross_loss = float(abs(np.sum(loss_profits))) if any(~win_mask) else 0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
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
            'sortino_ratio': float(sortino_ratio)
        }
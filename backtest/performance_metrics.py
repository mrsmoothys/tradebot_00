import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from scipy import stats
import matplotlib.pyplot as plt

class PerformanceCalculator:
    """
    Calculates trading strategy performance metrics.
    """
    
    def __init__(self):
        """Initialize the performance calculator."""
        pass
    
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
        drawdowns, max_dd_pct, max_dd_duration = self._calculate_drawdowns(equity_curve)
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
    
    def _calculate_drawdowns(self, equity_curve: List[float]) -> Tuple[np.ndarray, float, int]:
        """
        Calculate drawdowns series, maximum drawdown, and maximum drawdown duration.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Tuple of (drawdowns_series, max_drawdown_percentage, max_drawdown_duration)
        """
        if not equity_curve:
            return np.array([]), 0.0, 0
        
        # Convert to numpy array
        equity = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown in percentage
        drawdown = (running_max - equity) / running_max * 100
        
        # Maximum drawdown percentage
        max_dd_pct = np.max(drawdown)
        
        # Calculate drawdown duration
        is_drawdown = drawdown > 0
        durations = []
        current_duration = 0
        
        for i in range(len(equity)):
            if is_drawdown[i]:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Add final drawdown period if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        # Maximum drawdown duration
        max_dd_duration = max(durations) if durations else 0
        
        return drawdown, max_dd_pct, max_dd_duration
    
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
                if i == len(equity_curve) - 1 or not is_drawdown[i+1]:
                    durations.append(current_duration)
                    current_duration = 0
        
        return max(durations) if durations else 0
    
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
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics related to individual trades.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of trade metrics
        """
        metrics = {}
        
        # Basic trade metrics
        metrics['num_trades'] = len(trades)
        
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0,
                'largest_winner': 0.0,
                'largest_loser': 0.0,
                'avg_trade_duration': 0.0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.get('profit_amount', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_amount', 0) <= 0]
        
        # Win rate
        metrics['win_rate'] = len(winning_trades) / len(trades) * 100
        
        # Profit metrics
        gross_profit = sum(t.get('profit_amount', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit_amount', 0) for t in losing_trades))
        
        metrics['gross_profit'] = gross_profit
        metrics['gross_loss'] = gross_loss
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average metrics
        metrics['avg_profit_per_trade'] = sum(t.get('profit_amount', 0) for t in trades) / len(trades)
        metrics['avg_winning_trade'] = gross_profit / len(winning_trades) if winning_trades else 0.0
        metrics['avg_losing_trade'] = -gross_loss / len(losing_trades) if losing_trades else 0.0
        
        # Extreme values
        metrics['largest_winner'] = max((t.get('profit_amount', 0) for t in trades), default=0.0)
        metrics['largest_loser'] = min((t.get('profit_amount', 0) for t in trades), default=0.0)
        
        # Trade duration metrics if available
        if 'entry_time' in trades[0] and 'exit_time' in trades[0]:
            durations = []
            for trade in trades:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                if isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                
                duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
            
            metrics['avg_trade_duration'] = np.mean(durations)
            metrics['max_trade_duration'] = np.max(durations)
        
        return metrics
    
    def _calculate_benchmark_metrics(self, returns: List[float], 
                                  benchmark_returns: List[float]) -> Dict[str, float]:
        """
        Calculate metrics comparing strategy to benchmark.
        
        Args:
            returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            
        Returns:
            Dictionary of benchmark comparison metrics
        """
        metrics = {}
        
        # Truncate to match lengths
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        # Calculate alpha and beta if possible
        if min_length > 0:
            # Calculate beta (covariance / variance)
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            else:
                beta = 0.0
            
            # Calculate alpha (Jensen's alpha)
            alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
            
            metrics['alpha'] = alpha
            metrics['beta'] = beta
            
            # Information ratio
            tracking_error = np.std(np.array(returns) - np.array(benchmark_returns))
            excess_return = np.mean(returns) - np.mean(benchmark_returns)
            
            if tracking_error > 0:
                metrics['information_ratio'] = excess_return / tracking_error
            else:
                metrics['information_ratio'] = 0.0
            
            # Correlation
            metrics['correlation'] = np.corrcoef(returns, benchmark_returns)[0, 1]
            
            # Outperformance
            metrics['excess_return'] = excess_return
        
        return metrics
    
    def plot_drawdowns(self, equity_curve: List[float], top_n: int = 5, figsize: tuple = (12, 8)) -> None:
        """
        Plot the top N drawdowns.
        
        Args:
            equity_curve: List of equity values
            top_n: Number of top drawdowns to plot
            figsize: Figure size
        """
        if not equity_curve:
            return
        
        # Convert to numpy array
        equity = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        
        # Find drawdown periods
        is_drawdown = drawdown > 0
        
        # Find drawdown start and end indices
        drawdown_periods = []
        i = 0
        while i < len(drawdown):
            if is_drawdown[i]:
                start_idx = i
                while i < len(drawdown) and is_drawdown[i]:
                    i += 1
                end_idx = i - 1
                
                # Calculate max drawdown during this period
                max_dd = np.max(drawdown[start_idx:end_idx+1])
                drawdown_periods.append((start_idx, end_idx, max_dd))
            else:
                i += 1
        
        # Sort by drawdown size (largest first)
        drawdown_periods.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only top N
        drawdown_periods = drawdown_periods[:top_n]
        
        # Plot
        plt.figure(figsize=figsize)
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(equity, label='Equity Curve')
        
        # Highlight drawdown periods
        colors = plt.cm.tab10.colors
        for i, (start_idx, end_idx, max_dd) in enumerate(drawdown_periods):
            color = colors[i % len(colors)]
            plt.fill_between(range(start_idx, end_idx+1), equity[start_idx:end_idx+1], 
                           running_max[start_idx:end_idx+1], color=color, alpha=0.3,
                           label=f'DD #{i+1}: {max_dd:.2f}%')
        
        plt.title('Equity Curve with Top Drawdowns')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown values
        plt.subplot(2, 1, 2)
        plt.plot(drawdown, color='red', label='Drawdown')
        
        # Highlight drawdown periods
        for i, (start_idx, end_idx, max_dd) in enumerate(drawdown_periods):
            color = colors[i % len(colors)]
            plt.fill_between(range(start_idx, end_idx+1), 0, drawdown[start_idx:end_idx+1], 
                           color=color, alpha=0.3,
                           label=f'DD #{i+1}: {max_dd:.2f}%')
        
        plt.title('Drawdowns')
        plt.xlabel('Bar')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_distribution(self, trades: List[Dict[str, Any]], figsize: tuple = (14, 10)) -> None:
        """
        Plot trade profit distribution and other trade metrics.
        
        Args:
            trades: List of trade dictionaries
            figsize: Figure size
        """
        if not trades:
            return
        
        profit_percentages = [t.get('profit_percentage', 0) for t in trades]
        profit_amounts = [t.get('profit_amount', 0) for t in trades]
        durations = []
        
        if 'entry_time' in trades[0] and 'exit_time' in trades[0]:
            for trade in trades:
                entry_time = trade['entry_time']
                exit_time = trade['exit_time']
                
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                if isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                
                duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                durations.append(duration)
        
        plt.figure(figsize=figsize)
        
        # Profit distribution
        plt.subplot(2, 2, 1)
        plt.hist(profit_percentages, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Profit/Loss Distribution (%)')
        plt.xlabel('Profit/Loss (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Profit by trade number
        plt.subplot(2, 2, 2)
        plt.scatter(range(len(trades)), profit_percentages, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Profit/Loss by Trade Number')
        plt.xlabel('Trade Number')
        plt.ylabel('Profit/Loss (%)')
        plt.grid(True)
        
        # Cumulative profit
        plt.subplot(2, 2, 3)
        cumulative_profit = np.cumsum(profit_amounts)
        plt.plot(cumulative_profit)
        plt.title('Cumulative Profit/Loss')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Profit/Loss')
        plt.grid(True)
        
        # Trade duration vs profit (if available)
        if durations:
            plt.subplot(2, 2, 4)
            plt.scatter(durations, profit_percentages, alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.title('Profit/Loss vs Trade Duration')
            plt.xlabel('Duration (hours)')
            plt.ylabel('Profit/Loss (%)')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
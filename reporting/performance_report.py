import logging
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

class PerformanceReport:
    """
    Generates performance reports for trading strategies.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the performance report generator.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.report_config = self.config.get('reporting', {})
        
        # Report parameters
        self.output_dir = self.report_config.get('output_directory', 'reports')
        self.include_trades = self.report_config.get('include_trades', True)
        self.include_equity_curve = self.report_config.get('include_equity_curve', True)
        self.include_drawdowns = self.report_config.get('include_drawdowns', True)
        self.include_monthly_returns = self.report_config.get('include_monthly_returns', True)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, backtest_results: Dict[str, Any], 
                      report_name: Optional[str] = None) -> str:
        """
        Generate performance report from backtest results.
        
        Args:
            backtest_results: Dictionary of backtest results
            report_name: Name for the report (optional)
            
        Returns:
            Path to generated report
        """

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = report_name or f"performance_report_{timestamp}"
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Initialize report data
        report_data = {
            'timestamp': timestamp,
            'name': report_name,
            'overview': {},
            'performance_metrics': {},
            'symbol_metrics': {},
            'monthly_returns': {},
            'trade_statistics': {},
            'file_paths': {}
        }
        
        # Extract results
        performance_metrics = {}
        equity_curves = {}
        trades_list = {}
        
        # Process results for each symbol/timeframe
        for symbol, timeframe_results in backtest_results.items():
            for timeframe, results in timeframe_results.items():
                key = f"{symbol}_{timeframe}"
                
                # Extract metrics
                performance_metrics[key] = results['performance_metrics']
                
                # Extract equity curve
                if 'equity_curve' in results:
                    equity_curves[key] = results['equity_curve']
                
                # Extract trades
                if 'trades' in results:
                    trades_list[key] = results['trades']
        
        # Generate overview
        report_data['overview'] = self._generate_overview(performance_metrics)
        
        # Add performance metrics
        report_data['performance_metrics'] = performance_metrics
        
        # Generate and add symbol metrics
        report_data['symbol_metrics'] = self._generate_symbol_metrics(performance_metrics)
        
        # Generate trade statistics
        if self.include_trades:
            report_data['trade_statistics'] = self._generate_trade_statistics(trades_list)
            
            # Save trade details
            for key, trades in trades_list.items():
                if trades:
                    trades_df = pd.DataFrame(trades)
                    trades_file = os.path.join(report_dir, f"{key}_trades.csv")
                    trades_df.to_csv(trades_file, index=False)
                    report_data['file_paths'][f"{key}_trades"] = trades_file
        
        # Generate monthly returns
        if self.include_monthly_returns and equity_curves:
            report_data['monthly_returns'] = self._generate_monthly_returns(equity_curves)
        
        # Generate equity curve charts
        if self.include_equity_curve and equity_curves:
            for key, equity_curve in equity_curves.items():
                chart_file = os.path.join(report_dir, f"{key}_equity.png")
                self._plot_equity_curve(equity_curve, key, chart_file)
                report_data['file_paths'][f"{key}_equity"] = chart_file
        
        # Generate drawdown charts
        if self.include_drawdowns and equity_curves:
            for key, equity_curve in equity_curves.items():
                chart_file = os.path.join(report_dir, f"{key}_drawdown.png")
                self._plot_drawdowns(equity_curve, key, chart_file)
                report_data['file_paths'][f"{key}_drawdown"] = chart_file
        
        # Save report data
        report_json = os.path.join(report_dir, "report_data.json")
        with open(report_json, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_file = os.path.join(report_dir, "report.html")
        self._generate_html_report(report_data, html_file)
        
        self.logger.info(f"Generated performance report: {html_file}")
        
        return html_file
    
    def _generate_overview(self, performance_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overview metrics from performance data.
        
        Args:
            performance_metrics: Dictionary of performance metrics by symbol/timeframe
            
        Returns:
            Dictionary of overview metrics
        """
        if not performance_metrics:
            return {}
        
        # Aggregate metrics
        total_return = 0
        best_return = float('-inf')
        worst_return = float('inf')
        best_key = None
        worst_key = None
        
        for key, metrics in performance_metrics.items():
            return_val = metrics.get('total_return', 0)
            total_return += return_val
            
            if return_val > best_return:
                best_return = return_val
                best_key = key
            
            if return_val < worst_return:
                worst_return = return_val
                worst_key = key
        
        # Calculate average metrics
        avg_metrics = {}
        metric_keys = ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        for metric in metric_keys:
            values = [m.get(metric, 0) for m in performance_metrics.values()]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values) if values else 0
        
        # Create overview
        overview = {
            'total_strategies': len(performance_metrics),
            'cumulative_return': total_return,
            'average_return': total_return / len(performance_metrics) if performance_metrics else 0,
            'best_strategy': {
                'name': best_key,
                'return': best_return
            },
            'worst_strategy': {
                'name': worst_key,
                'return': worst_return
            },
            **avg_metrics
        }
        
        return overview
    
    def _generate_symbol_metrics(self, performance_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate metrics grouped by symbol.
        
        Args:
            performance_metrics: Dictionary of performance metrics by symbol/timeframe
            
        Returns:
            Dictionary of metrics by symbol
        """
        if not performance_metrics:
            return {}
        
        symbol_metrics = {}
        
        # Group by symbol
        for key, metrics in performance_metrics.items():
            symbol = key.split('_')[0]
            
            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = []
            
            symbol_metrics[symbol].append(metrics)
        
        # Calculate average metrics for each symbol
        result = {}
        
        for symbol, metrics_list in symbol_metrics.items():
            avg_metrics = {}
            metric_keys = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']
            
            for metric in metric_keys:
                values = [m.get(metric, 0) for m in metrics_list]
                avg_metrics[f"avg_{metric}"] = sum(values) / len(values) if values else 0
            
            result[symbol] = avg_metrics
        
        return result
    
    def _generate_trade_statistics(self, trades_list: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Generate detailed trade statistics.
        
        Args:
            trades_list: Dictionary of trades by symbol/timeframe
            
        Returns:
            Dictionary of trade statistics
        """
        if not trades_list:
            return {}
        
        trade_stats = {}
        
        for key, trades in trades_list.items():
            if not trades:
                continue
            
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.get('profit_amount', 0) > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            profit_amounts = [t.get('profit_amount', 0) for t in trades]
            total_profit = sum(max(0, p) for p in profit_amounts)
            total_loss = sum(abs(min(0, p)) for p in profit_amounts)
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Average values
            avg_profit = total_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
            
            # Largest trades
            largest_win = max(profit_amounts) if profit_amounts else 0
            largest_loss = min(profit_amounts) if profit_amounts else 0
            
            # Trade durations if available
            avg_duration = 0
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
                
                avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Create stats
            stats = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': total_profit - total_loss,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_duration_hours': avg_duration
            }
            
            # Add exit reason breakdown if available
            if 'exit_reason' in trades[0]:
                reason_counts = {}
                for trade in trades:
                    reason = trade.get('exit_reason', 'unknown')
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                stats['exit_reasons'] = reason_counts
            
            trade_stats[key] = stats
        
        return trade_stats
    
    def _generate_monthly_returns(self, equity_curves: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generate monthly returns table.
        
        Args:
            equity_curves: Dictionary of equity curves by symbol/timeframe
            
        Returns:
            Dictionary of monthly returns by symbol/timeframe
        """
        monthly_returns = {}
        
        for key, equity_curve in equity_curves.items():
            # Skip if no timestamps available
            if isinstance(equity_curve, list):
                continue
            
            # Extract returns
            daily_returns = pd.Series(equity_curve).pct_change().dropna()
            
            # Group by month
            monthly = daily_returns.groupby(pd.Grouper(freq='M')).sum()
            
            # Convert to dictionary
            monthly_data = {
                str(date.strftime('%Y-%m')): float(ret) 
                for date, ret in monthly.items()
            }
            
            monthly_returns[key] = monthly_data
        
        return monthly_returns
    
    def _plot_equity_curve(self, equity_curve: List[float], title: str, filepath: str) -> None:
        """
        Plot equity curve and save to file.
        
        Args:
            equity_curve: List of equity values
            title: Chart title
            filepath: Path to save chart
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve
        plt.plot(equity_curve)
        
        # Format chart
        plt.title(f"Equity Curve - {title}")
        plt.xlabel("Bar")
        plt.ylabel("Equity")
        plt.grid(True)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f"${x:,.2f}")
        )
        
        # Save chart
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _plot_drawdowns(self, equity_curve: List[float], title: str, filepath: str) -> None:
        """
        Plot drawdowns and save to file.
        
        Args:
            equity_curve: List of equity values
            title: Chart title
            filepath: Path to save chart
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdowns
        equity = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks * 100
        
        # Plot drawdowns
        plt.plot(drawdowns)
        
        # Format chart
        plt.title(f"Drawdown - {title}")
        plt.xlabel("Bar")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        
        # Add horizontal lines at 0%, -5%, -10%, etc.
        for level in range(0, -55, -5):
            plt.axhline(y=level, color='r' if level <= -20 else 'y' if level <= -10 else 'g',
                      linestyle='--', alpha=0.3)
        
        # Save chart
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _generate_html_report(self, report_data: Dict[str, Any], filepath: str) -> None:
        """
        Generate HTML performance report.
        
        Args:
            report_data: Report data dictionary
            filepath: Path to save HTML report
        """
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trading Strategy Performance Report - {report_data['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart {{ margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Performance Report</h1>
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Report name: {report_data['name']}</p>
            </div>
        """
        
        # Overview section
        overview = report_data['overview']
        html += f"""
            <div class="section">
                <h2>Performance Overview</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Strategies</td>
                        <td>{overview.get('total_strategies', 0)}</td>
                    </tr>
                    <tr>
                        <td>Cumulative Return</td>
                        <td class="{'positive' if overview.get('cumulative_return', 0) >= 0 else 'negative'}">
                            {overview.get('cumulative_return', 0):.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>Average Return</td>
                        <td class="{'positive' if overview.get('average_return', 0) >= 0 else 'negative'}">
                            {overview.get('average_return', 0):.2f}%
                        </td>
                    </tr>
                    <tr>
                        <td>Average Sharpe Ratio</td>
                        <td>{overview.get('avg_sharpe_ratio', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Sortino Ratio</td>
                        <td>{overview.get('avg_sortino_ratio', 0):.2f}</td>
                    </tr>
                    <tr>
                        <td>Average Max Drawdown</td>
                        <td class="negative">{overview.get('avg_max_drawdown', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Average Win Rate</td>
                        <td>{overview.get('avg_win_rate', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Best Strategy</td>
                        <td>{overview.get('best_strategy', {}).get('name', '')} 
                            <span class="positive">({overview.get('best_strategy', {}).get('return', 0):.2f}%)</span>
                        </td>
                    </tr>
                    <tr>
                        <td>Worst Strategy</td>
                        <td>{overview.get('worst_strategy', {}).get('name', '')} 
                            <span class="negative">({overview.get('worst_strategy', {}).get('return', 0):.2f}%)</span>
                        </td>
                    </tr>
                </table>
            </div>
        """
        
        # Strategy details section
        html += """
            <div class="section">
                <h2>Strategy Details</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Return</th>
                        <th>Sharpe</th>
                        <th>Sortino</th>
                        <th>Max DD</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                    </tr>
        """
        
        # Add each strategy's metrics
        for key, metrics in report_data['performance_metrics'].items():
            total_return = metrics.get('total_return', 0)
            return_class = 'positive' if total_return >= 0 else 'negative'
            
            html += f"""
                <tr>
                    <td>{key}</td>
                    <td class="{return_class}">{total_return:.2f}%</td>
                    <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                    <td>{metrics.get('sortino_ratio', 0):.2f}</td>
                    <td class="negative">{metrics.get('max_drawdown', 0):.2f}%</td>
                    <td>{metrics.get('win_rate', 0):.2f}%</td>
                    <td>{metrics.get('profit_factor', 0):.2f}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Trade statistics section
        if report_data['trade_statistics']:
            html += """
                <div class="section">
                    <h2>Trade Statistics</h2>
                    <table>
                        <tr>
                            <th>Strategy</th>
                            <th>Total Trades</th>
                            <th>Win Rate</th>
                            <th>Profit Factor</th>
                            <th>Net Profit</th>
                            <th>Avg Profit</th>
                            <th>Avg Loss</th>
                            <th>Avg Duration</th>
                        </tr>
            """
            
            # Add each strategy's trade stats
            for key, stats in report_data['trade_statistics'].items():
                net_profit = stats.get('net_profit', 0)
                profit_class = 'positive' if net_profit >= 0 else 'negative'
                
                html += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{stats.get('total_trades', 0)}</td>
                        <td>{stats.get('win_rate', 0):.2f}%</td>
                        <td>{stats.get('profit_factor', 0):.2f}</td>
                        <td class="{profit_class}">${net_profit:.2f}</td>
                        <td class="positive">${stats.get('avg_profit', 0):.2f}</td>
                        <td class="negative">${stats.get('avg_loss', 0):.2f}</td>
                        <td>{stats.get('avg_duration_hours', 0):.2f} hours</td>
                    </tr>
                """
            
            html += """
                    </table>
                </div>
            """
        
        # Charts section
        if report_data['file_paths']:
            html += """
                <div class="section">
                    <h2>Charts</h2>
            """
            
            # Add equity curve charts
            for key, path in report_data['file_paths'].items():
                if key.endswith('_equity'):
                    strategy = key.replace('_equity', '')
                    html += f"""
                        <div class="chart">
                            <h3>Equity Curve - {strategy}</h3>
                            <img src="{os.path.basename(path)}" alt="Equity Curve">
                        </div>
                    """
            
            # Add drawdown charts
            for key, path in report_data['file_paths'].items():
                if key.endswith('_drawdown'):
                    strategy = key.replace('_drawdown', '')
                    html += f"""
                        <div class="chart">
                            <h3>Drawdown - {strategy}</h3>
                            <img src="{os.path.basename(path)}" alt="Drawdown">
                        </div>
                    """
            
            html += """
                </div>
            """
        
        # Monthly returns section
        if report_data['monthly_returns']:
            html += """
                <div class="section">
                    <h2>Monthly Returns</h2>
            """
            
            for key, monthly_data in report_data['monthly_returns'].items():
                if not monthly_data:
                    continue
                
                html += f"""
                    <h3>{key}</h3>
                    <table>
                        <tr>
                            <th>Month</th>
                            <th>Return</th>
                        </tr>
                """
                
                # Sort months chronologically
                sorted_months = sorted(monthly_data.keys())
                
                for month in sorted_months:
                    return_val = monthly_data[month] * 100  # Convert to percentage
                    return_class = 'positive' if return_val >= 0 else 'negative'
                    
                    html += f"""
                        <tr>
                            <td>{month}</td>
                            <td class="{return_class}">{return_val:.2f}%</td>
                        </tr>
                    """
                
                html += """
                    </table>
                """
            
            html += """
                </div>
            """
        
        # Footer
        html += f"""
            <div class="footer">
                <p>Generated by the Trading Strategy Performance Reporter - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filepath, 'w') as f:
            f.write(html)
    
    def generate_comparison_report(self, results_list: List[Dict[str, Any]], 
                               names: List[str], report_name: Optional[str] = None) -> str:
        """
        Generate comparison report for multiple strategies.
        
        Args:
            results_list: List of backtest results to compare
            names: List of names for each result set
            report_name: Name for the report (optional)
            
        Returns:
            Path to generated report
        """
        if len(results_list) != len(names):
            raise ValueError("Number of results and names must match")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = report_name or f"comparison_report_{timestamp}"
        
        # Create report directory
        report_dir = os.path.join(self.output_dir, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Extract metrics for comparison
        metrics_list = []
        equity_curves = {}
        
        for i, (results, name) in enumerate(zip(results_list, names)):
            metrics = {}
            
            # Process each symbol/timeframe in the results
            for symbol, timeframe_results in results.items():
                for timeframe, result in timeframe_results.items():
                    key = f"{symbol}_{timeframe}"
                    
                    # Extract metrics
                    metrics[key] = result['performance_metrics']
                    
                    # Extract equity curve
                    if 'equity_curve' in result:
                        equity_curves[f"{name}_{key}"] = result['equity_curve']
            
            metrics_list.append({
                'name': name,
                'metrics': metrics
            })
        
        # Generate comparison chart
        chart_file = os.path.join(report_dir, "comparison_chart.png")
        self._plot_comparison(metrics_list, chart_file)
        
        # Generate equity curve comparison
        equity_chart_file = os.path.join(report_dir, "equity_comparison.png")
        self._plot_equity_comparison(equity_curves, equity_chart_file)
        
        # Generate HTML report
        html_file = os.path.join(report_dir, "comparison_report.html")
        self._generate_comparison_html(metrics_list, equity_curves, html_file,
                                    chart_file=chart_file,
                                    equity_chart_file=equity_chart_file)
        
        self.logger.info(f"Generated comparison report: {html_file}")
        
        return html_file
    
    def _plot_comparison(self, metrics_list: List[Dict[str, Any]], filepath: str) -> None:
        """
        Plot comparison chart and save to file.
        
        Args:
            metrics_list: List of metrics dictionaries
            filepath: Path to save chart
        """
        # Calculate average metrics for each strategy
        comparison_data = []
        
        for strategy in metrics_list:
            name = strategy['name']
            metrics = strategy['metrics']
            
            # Calculate averages
            avg_return = np.mean([m.get('total_return', 0) for m in metrics.values()])
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in metrics.values()])
            avg_sortino = np.mean([m.get('sortino_ratio', 0) for m in metrics.values()])
            avg_drawdown = np.mean([m.get('max_drawdown', 0) for m in metrics.values()])
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in metrics.values()])
            
            comparison_data.append({
                'name': name,
                'return': avg_return,
                'sharpe': avg_sharpe,
                'sortino': avg_sortino,
                'drawdown': avg_drawdown,
                'win_rate': avg_win_rate
            })
        
        # Create grouped bar chart
        plt.figure(figsize=(14, 8))
        
        # Define metrics to plot
        metrics = ['return', 'sharpe', 'sortino', 'drawdown', 'win_rate']
        metric_labels = ['Return (%)', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        
        # Create subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 8))
        
        # Plot each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [d[metric] for d in comparison_data]
            names = [d['name'] for d in comparison_data]
            
            # Plot bars
            bars = axes[i].bar(names, values)
            
            # Color bars based on metric (green is good, red is bad)
            for j, bar in enumerate(bars):
                if metric == 'drawdown':
                    # Lower drawdown is better
                    color = 'green' if values[j] < np.mean(values) else 'red'
                else:
                    # Higher value is better
                    color = 'green' if values[j] > np.mean(values) else 'red'
                
                bar.set_color(color)
            
            # Customize plot
            axes[i].set_title(label)
            axes[i].set_xlabel('Strategy')
            axes[i].set_ylabel(label)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom')
        
        # Finalize layout
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _plot_equity_comparison(self, equity_curves: Dict[str, List[float]], filepath: str) -> None:
        """
        Plot equity curve comparison and save to file.
        
        Args:
            equity_curves: Dictionary of equity curves by name
            filepath: Path to save chart
        """
        if not equity_curves:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Normalize equity curves to start at 100
        for name, curve in equity_curves.items():
            if not curve:
                continue
            
            normalized_curve = [100 * value / curve[0] for value in curve]
            plt.plot(normalized_curve, label=name)
        
        # Format chart
        plt.title("Equity Curve Comparison (Normalized)")
        plt.xlabel("Bar")
        plt.ylabel("Equity (Normalized to 100)")
        plt.grid(True)
        plt.legend()
        
        # Save chart
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def _generate_comparison_html(self, metrics_list: List[Dict[str, Any]], 
                                equity_curves: Dict[str, List[float]],
                                filepath: str, chart_file: str = None,
                                equity_chart_file: str = None) -> None:
        """
        Generate HTML comparison report.
        
        Args:
            metrics_list: List of metrics dictionaries
            equity_curves: Dictionary of equity curves by name
            filepath: Path to save HTML report
            chart_file: Path to comparison chart image
            equity_chart_file: Path to equity curve comparison image
        """
        # Start HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Strategy Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .chart {{ margin: 20px 0; }}
                .chart img {{ max-width: 100%; height: auto; }}
                .winner {{ background-color: rgba(0, 255, 0, 0.1); }}
                .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Strategy Comparison Report</h1>
                <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        # Comparison charts section
        if chart_file or equity_chart_file:
            html += """
                <div class="section">
                    <h2>Comparison Charts</h2>
            """
            
            if chart_file:
                html += f"""
                    <div class="chart">
                        <h3>Performance Metrics Comparison</h3>
                        <img src="{os.path.basename(chart_file)}" alt="Performance Comparison">
                    </div>
                """
            
            if equity_chart_file:
                html += f"""
                    <div class="chart">
                        <h3>Equity Curve Comparison</h3>
                        <img src="{os.path.basename(equity_chart_file)}" alt="Equity Comparison">
                    </div>
                """
            
            html += """
                </div>
            """
        
        # Performance comparison table
        html += """
            <div class="section">
                <h2>Performance Summary</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Avg Return</th>
                        <th>Avg Sharpe</th>
                        <th>Avg Sortino</th>
                        <th>Avg Max DD</th>
                        <th>Avg Win Rate</th>
                    </tr>
        """
        
        # Calculate average metrics for each strategy
        comparison_data = []
        
        for strategy in metrics_list:
            name = strategy['name']
            metrics = strategy['metrics']
            
            # Calculate averages
            avg_return = np.mean([m.get('total_return', 0) for m in metrics.values()])
            avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in metrics.values()])
            avg_sortino = np.mean([m.get('sortino_ratio', 0) for m in metrics.values()])
            avg_drawdown = np.mean([m.get('max_drawdown', 0) for m in metrics.values()])
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in metrics.values()])
            
            comparison_data.append({
                'name': name,
                'return': avg_return,
                'sharpe': avg_sharpe,
                'sortino': avg_sortino,
                'drawdown': avg_drawdown,
                'win_rate': avg_win_rate
            })
        
        # Find best value for each metric
        best_return = max([d['return'] for d in comparison_data])
        best_sharpe = max([d['sharpe'] for d in comparison_data])
        best_sortino = max([d['sortino'] for d in comparison_data])
        best_drawdown = min([d['drawdown'] for d in comparison_data])  # Lower is better
        best_win_rate = max([d['win_rate'] for d in comparison_data])
        
        # Add each strategy's summary
        for data in comparison_data:
            return_class = 'positive' if data['return'] >= 0 else 'negative'
            
            html += f"""
                <tr>
                    <td>{data['name']}</td>
                    <td class="{return_class} {'winner' if data['return'] == best_return else ''}">
                        {data['return']:.2f}%
                    </td>
                    <td class="{'winner' if data['sharpe'] == best_sharpe else ''}">
                        {data['sharpe']:.2f}
                    </td>
                    <td class="{'winner' if data['sortino'] == best_sortino else ''}">
                        {data['sortino']:.2f}
                    </td>
                    <td class="negative {'winner' if data['drawdown'] == best_drawdown else ''}">
                        {data['drawdown']:.2f}%
                    </td>
                    <td class="{'winner' if data['win_rate'] == best_win_rate else ''}">
                        {data['win_rate']:.2f}%
                    </td>
                </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Detailed metrics by symbol/timeframe
        html += """
            <div class="section">
                <h2>Detailed Metrics by Symbol/Timeframe</h2>
        """
        
        # Get unique keys across all strategies
        all_keys = set()
        for strategy in metrics_list:
            all_keys.update(strategy['metrics'].keys())
        
        # Sort keys
        sorted_keys = sorted(all_keys)
        
        # Create a table for each symbol/timeframe
        for key in sorted_keys:
            html += f"""
                <h3>{key}</h3>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Return</th>
                        <th>Sharpe</th>
                        <th>Sortino</th>
                        <th>Max DD</th>
                        <th>Win Rate</th>
                    </tr>
            """
            
            # Collect metrics for this key across all strategies
            key_metrics = []
            for strategy in metrics_list:
                if key in strategy['metrics']:
                    metrics = strategy['metrics'][key]
                    key_metrics.append({
                        'name': strategy['name'],
                        'metrics': metrics
                    })
            
            # Find best values for this key
            best_return = max([m['metrics'].get('total_return', float('-inf')) for m in key_metrics])
            best_sharpe = max([m['metrics'].get('sharpe_ratio', float('-inf')) for m in key_metrics])
            best_sortino = max([m['metrics'].get('sortino_ratio', float('-inf')) for m in key_metrics])
            best_drawdown = min([m['metrics'].get('max_drawdown', float('inf')) for m in key_metrics])
            best_win_rate = max([m['metrics'].get('win_rate', float('-inf')) for m in key_metrics])
            
            # Add each strategy's metrics for this key
            for item in key_metrics:
                m = item['metrics']
                total_return = m.get('total_return', 0)
                return_class = 'positive' if total_return >= 0 else 'negative'
                
                html += f"""
                    <tr>
                        <td>{item['name']}</td>
                        <td class="{return_class} {'winner' if total_return == best_return else ''}">
                            {total_return:.2f}%
                        </td>
                        <td class="{'winner' if m.get('sharpe_ratio', 0) == best_sharpe else ''}">
                            {m.get('sharpe_ratio', 0):.2f}
                        </td>
                        <td class="{'winner' if m.get('sortino_ratio', 0) == best_sortino else ''}">
                            {m.get('sortino_ratio', 0):.2f}
                        </td>
                        <td class="negative {'winner' if m.get('max_drawdown', 0) == best_drawdown else ''}">
                            {m.get('max_drawdown', 0):.2f}%
                        </td>
                        <td class="{'winner' if m.get('win_rate', 0) == best_win_rate else ''}">
                            {m.get('win_rate', 0):.2f}%
                        </td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        """
        
        # Footer
        html += f"""
            <div class="footer">
                <p>Generated by the Trading Strategy Comparison Reporter - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filepath, 'w') as f:
            f.write(html)
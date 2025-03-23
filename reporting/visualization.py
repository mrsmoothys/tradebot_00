import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.ticker import FuncFormatter

class Visualizer:
    """
    Creates visualizations of trading data and performance.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the visualizer.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        self.visualization_config = self.config.get('visualization', {})
        
        # Visualization parameters
        self.output_dir = self.visualization_config.get('output_directory', 'visualizations')
        self.default_figsize = self.visualization_config.get('default_figsize', (12, 8))
        self.chart_style = self.visualization_config.get('chart_style', 'darkgrid')
        self.color_palette = self.visualization_config.get('color_palette', 'viridis')
        
        # Set default style
        sns.set_style(self.chart_style)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_ohlc(self, df: pd.DataFrame, symbol: str, signals: Optional[pd.DataFrame] = None,
               trades: Optional[List[Dict[str, Any]]] = None, save_path: Optional[str] = None,
               interactive: bool = False) -> Any:
        """
        Plot OHLC chart with optional signals and trades.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            signals: DataFrame with signal data (optional)
            trades: List of trade dictionaries (optional)
            save_path: Path to save chart (optional)
            interactive: Whether to use interactive Plotly chart
            
        Returns:
            Figure object
        """
        if interactive:
            return self._plot_ohlc_plotly(df, symbol, signals, trades, save_path)
        else:
            return self._plot_ohlc_matplotlib(df, symbol, signals, trades, save_path)
    
    def _plot_ohlc_matplotlib(self, df: pd.DataFrame, symbol: str, signals: Optional[pd.DataFrame] = None,
                          trades: Optional[List[Dict[str, Any]]] = None, 
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot OHLC chart using Matplotlib.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            signals: DataFrame with signal data (optional)
            trades: List of trade dictionaries (optional)
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure and primary axis for price
        fig, ax1 = plt.subplots(figsize=self.default_figsize)
        
        # Plot candlestick chart
        width = 0.6
        width2 = width / 2
        
        # Get OHLC data
        dates = df.index
        opens = df['open']
        highs = df['high']
        lows = df['low']
        closes = df['close']
        
        # Plot up and down candles
        up = closes >= opens
        down = opens > closes
        
        # Plot candles
        ax1.bar(dates[up], height=closes[up]-opens[up], bottom=opens[up], width=width, color='green', alpha=0.7)
        ax1.bar(dates[up], height=highs[up]-closes[up], bottom=closes[up], width=width2, color='green', alpha=0.7)
        ax1.bar(dates[up], height=opens[up]-lows[up], bottom=lows[up], width=width2, color='green', alpha=0.7)
        
        ax1.bar(dates[down], height=opens[down]-closes[down], bottom=closes[down], width=width, color='red', alpha=0.7)
        ax1.bar(dates[down], height=highs[down]-opens[down], bottom=opens[down], width=width2, color='red', alpha=0.7)
        ax1.bar(dates[down], height=closes[down]-lows[down], bottom=lows[down], width=width2, color='red', alpha=0.7)
        
        # Set labels and title
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} Price Chart')
        ax1.grid(True)
        
        # Format x-axis with dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Create secondary axis for volume
        ax2 = ax1.twinx()
        ax2.bar(dates, df['volume'], width=width, color='gray', alpha=0.3)
        ax2.set_ylabel('Volume', color='gray')
        
        # Add signals if provided
        if signals is not None and 'signal' in signals.columns:
            # Plot buy signals
            buy_signals = signals[signals['signal'] > 0]
            if not buy_signals.empty:
                ax1.scatter(buy_signals.index, lows[buy_signals.index] * 0.99, 
                         marker='^', color='green', s=100, label='Buy Signal')
            
            # Plot sell signals
            sell_signals = signals[signals['signal'] < 0]
            if not sell_signals.empty:
                ax1.scatter(sell_signals.index, highs[sell_signals.index] * 1.01, 
                         marker='v', color='red', s=100, label='Sell Signal')
        
        # Add trades if provided
        if trades:
            entry_times = []
            entry_prices = []
            exit_times = []
            exit_prices = []
            profits = []
            
            for trade in trades:
                # Convert timestamps to datetime if they are strings
                entry_time = trade['entry_time']
                exit_time = trade['exit_time'] if 'exit_time' in trade else None
                
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                
                if exit_time and isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                
                # Extract prices
                entry_price = trade['entry_price']
                exit_price = trade.get('exit_price')
                
                # Store for plotting
                entry_times.append(entry_time)
                entry_prices.append(entry_price)
                
                if exit_time and exit_price:
                    exit_times.append(exit_time)
                    exit_prices.append(exit_price)
                    profits.append(trade.get('profit_pct', 0))
            
            # Plot entries
            ax1.scatter(entry_times, entry_prices, marker='o', color='blue', s=80, label='Entry')
            
            # Plot exits
            if exit_times:
                # Color exits based on profit/loss
                colors = ['green' if p > 0 else 'red' for p in profits]
                ax1.scatter(exit_times, exit_prices, marker='x', c=colors, s=80, label='Exit')
                
                # Draw lines connecting entries and exits
                for i in range(len(exit_times)):
                    ax1.plot([entry_times[i], exit_times[i]], [entry_prices[i], exit_prices[i]], 
                          color=colors[i], linestyle='--', alpha=0.7)
        
        # Add legend
        ax1.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved OHLC chart to {save_path}")
        
        return fig
    
    def _plot_ohlc_plotly(self, df: pd.DataFrame, symbol: str, signals: Optional[pd.DataFrame] = None,
                       trades: Optional[List[Dict[str, Any]]] = None, 
                       save_path: Optional[str] = None) -> go.Figure:
        """
        Plot interactive OHLC chart using Plotly.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            signals: DataFrame with signal data (optional)
            trades: List of trade dictionaries (optional)
            save_path: Path to save chart (optional)
            
        Returns:
            Plotly Figure object
        """
        # Create subplot with 2 rows
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, row_heights=[0.7, 0.3],
                          subplot_titles=(f'{symbol} Price', 'Volume'))
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)
        
        # Add volume trace
        colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.3
        ), row=2, col=1)
        
        # Add signals if provided
        if signals is not None and 'signal' in signals.columns:
            # Plot buy signals
            buy_signals = signals[signals['signal'] > 0]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=df.loc[buy_signals.index, 'low'] * 0.99,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name='Buy Signal'
                ), row=1, col=1)
            
            # Plot sell signals
            sell_signals = signals[signals['signal'] < 0]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=df.loc[sell_signals.index, 'high'] * 1.01,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name='Sell Signal'
                ), row=1, col=1)
        
        # Add trades if provided
        if trades:
            entry_times = []
            entry_prices = []
            exit_times = []
            exit_prices = []
            profit_colors = []
            hover_texts = []
            
            for trade in trades:
                # Convert timestamps to datetime if they are strings
                entry_time = trade['entry_time']
                exit_time = trade.get('exit_time')
                
                if isinstance(entry_time, str):
                    entry_time = pd.to_datetime(entry_time)
                
                if exit_time and isinstance(exit_time, str):
                    exit_time = pd.to_datetime(exit_time)
                
                # Extract prices and profit
                entry_price = trade['entry_price']
                exit_price = trade.get('exit_price')
                profit_pct = trade.get('profit_pct', 0)
                
                # Store for plotting
                entry_times.append(entry_time)
                entry_prices.append(entry_price)
                
                if exit_time and exit_price:
                    exit_times.append(exit_time)
                    exit_prices.append(exit_price)
                    color = 'green' if profit_pct > 0 else 'red'
                    profit_colors.append(color)
                    hover_texts.append(f"Profit: {profit_pct:.2f}%")
            
            # Plot entries
            fig.add_trace(go.Scatter(
                x=entry_times,
                y=entry_prices,
                mode='markers',
                marker=dict(symbol='circle', size=10, color='blue'),
                name='Entry'
            ), row=1, col=1)
            
            # Plot exits
            if exit_times:
                fig.add_trace(go.Scatter(
                    x=exit_times,
                    y=exit_prices,
                    mode='markers',
                    marker=dict(symbol='x', size=10, color=profit_colors),
                    text=hover_texts,
                    hoverinfo='text+x+y',
                    name='Exit'
                ), row=1, col=1)
                
                # Draw lines connecting entries and exits
                for i in range(len(exit_times)):
                    fig.add_trace(go.Scatter(
                        x=[entry_times[i], exit_times[i]],
                        y=[entry_prices[i], exit_prices[i]],
                        mode='lines',
                        line=dict(color=profit_colors[i], width=1, dash='dash'),
                        showlegend=False
                    ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Trading Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Saved interactive OHLC chart to {save_path}")
        
        return fig
    
    def plot_equity_curve(self, equity_curve: List[float], benchmark: Optional[List[float]] = None,
                       title: str = 'Equity Curve', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot equity curve with optional benchmark comparison.
        
        Args:
            equity_curve: List of equity values
            benchmark: List of benchmark values (optional)
            title: Chart title
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Plot equity curve
        ax.plot(equity_curve, label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark:
            # Truncate to same length if necessary
            min_length = min(len(equity_curve), len(benchmark))
            benchmark = benchmark[:min_length]
            
            # Normalize benchmark to start at same value as equity
            benchmark_norm = [benchmark[i] * equity_curve[0] / benchmark[0] for i in range(min_length)]
            
            ax.plot(benchmark_norm, label='Benchmark', linewidth=2, alpha=0.7)
        
        # Calculate drawdown
        if equity_curve:
            peaks = np.maximum.accumulate(equity_curve)
            drawdown = [(equity_curve[i] - peaks[i]) / peaks[i] * 100 for i in range(len(equity_curve))]
            
            # Create secondary y-axis for drawdown
            ax2 = ax.twinx()
            ax2.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red', label='Drawdown')
            ax2.set_ylabel('Drawdown (%)', color='red')
            ax2.tick_params(axis='y', colors='red')
            ax2.set_ylim(min(drawdown) * 1.1, 5)  # Set y-axis limit with some padding
        
        # Set labels and title
        ax.set_xlabel('Bar')
        ax.set_ylabel('Equity')
        ax.set_title(title)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.2f}"))
        
        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='upper left')
        
        # Add annotations for key metrics
        if equity_curve:
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            max_dd = min(drawdown) if drawdown else 0
            
            metrics_text = f"Total Return: {total_return:.2f}%\nMax Drawdown: {max_dd:.2f}%"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved equity curve to {save_path}")
        
        return fig
    
    def plot_drawdowns(self, equity_curve: List[float], 
                    title: str = 'Drawdown Analysis', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            equity_curve: List of equity values
            title: Chart title
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.default_figsize, sharex=True)
        
        # Calculate drawdown
        equity = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity)
        drawdown = (equity - peaks) / peaks * 100
        
        # Plot equity curve
        ax1.plot(equity, label='Equity', color='blue')
        ax1.plot(peaks, label='Peak Equity', color='green', linestyle='--')
        
        # Shade underwater periods
        underwater = equity < peaks
        for i in range(len(underwater)):
            if underwater[i] and (i == 0 or not underwater[i-1]):
                start_idx = i
            elif not underwater[i] and i > 0 and underwater[i-1]:
                end_idx = i
                ax1.axvspan(start_idx, end_idx, color='red', alpha=0.2)
        
        # Plot drawdown
        ax2.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.5)
        
        # Find worst drawdowns
        worst_dd_idx = np.argmin(drawdown)
        worst_dd = drawdown[worst_dd_idx]
        
        # Highlight worst drawdown
        recovery_idx = worst_dd_idx
        while recovery_idx < len(drawdown) - 1 and drawdown[recovery_idx] < 0:
            recovery_idx += 1
        
        # Find peak before worst drawdown
        peak_idx = worst_dd_idx
        while peak_idx > 0 and equity[peak_idx-1] <= equity[peak_idx]:
            peak_idx -= 1
        
        # Highlight worst drawdown period
        ax1.axvspan(peak_idx, recovery_idx, color='red', alpha=0.3)
        ax2.axvspan(peak_idx, recovery_idx, color='red', alpha=0.3)
        
        # Mark worst drawdown point
        ax1.scatter(worst_dd_idx, equity[worst_dd_idx], color='red', s=100, zorder=5)
        ax2.scatter(worst_dd_idx, drawdown[worst_dd_idx], color='red', s=100, zorder=5)
        
        # Add text annotations
        ax2.annotate(f"Max DD: {worst_dd:.2f}%", 
                   xy=(worst_dd_idx, worst_dd), 
                   xytext=(worst_dd_idx+5, worst_dd*0.8),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                   fontsize=10)
        
        # Add recovery line if recovered
        if recovery_idx < len(drawdown) - 1:
            ax1.plot([worst_dd_idx, recovery_idx], [equity[worst_dd_idx], equity[recovery_idx]], 
                   'g--', alpha=0.7, linewidth=2)
            
            recovery_time = recovery_idx - worst_dd_idx
            ax1.annotate(f"Recovery: {recovery_time} bars", 
                       xy=(recovery_idx, equity[recovery_idx]),
                       xytext=(recovery_idx, equity[recovery_idx]*1.05),
                       ha='right')
        
        # Set labels and title
        ax1.set_title(title)
        ax1.set_ylabel('Equity')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Bar')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Add horizontal lines at common drawdown levels
        for level in [-5, -10, -15, -20, -25]:
            ax2.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
            ax2.text(len(drawdown)*0.02, level*1.1, f"{level}%", va='bottom', ha='left')
        
        # Format y-axis of equity curve as currency
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.2f}"))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved drawdown analysis to {save_path}")
        
        return fig
    
    def plot_monthly_returns(self, equity_curve: List[float], dates: Optional[List[datetime]] = None,
                          title: str = 'Monthly Returns', save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            equity_curve: List of equity values
            dates: List of dates corresponding to equity values (optional)
            title: Chart title
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        # Create dates if not provided
        if dates is None:
            dates = pd.date_range(end=datetime.now(), periods=len(equity_curve), freq='D')
        
        # Convert to DataFrame
        df = pd.DataFrame({'equity': equity_curve}, index=dates)
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Group by year and month
        monthly_returns = df['daily_return'].groupby([df.index.year, df.index.month]).sum() * 100
        
        # Reshape for heatmap
        returns_matrix = []
        years = sorted(set(df.index.year))
        
        for year in years:
            year_returns = []
            for month in range(1, 13):
                try:
                    ret = monthly_returns.loc[(year, month)]
                    year_returns.append(ret)
                except KeyError:
                    year_returns.append(np.nan)
            returns_matrix.append(year_returns)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Define colormap
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        
        # Create heatmap
        im = ax.imshow(returns_matrix, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(12))
        ax.set_yticks(range(len(years)))
        
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        ax.set_yticklabels(years)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Return (%)')
        
        # Add text annotations with return values
        for i in range(len(years)):
            for j in range(12):
                value = returns_matrix[i][j]
                if not np.isnan(value):
                    text_color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f"{value:.1f}%", ha='center', va='center', color=text_color)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved monthly returns to {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades: List[Dict[str, Any]], 
                         title: str = 'Trade Analysis', save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plot comprehensive trade analysis.
        
        Args:
            trades: List of trade dictionaries
            title: Chart title
            save_path: Path to save chart (optional)
            
        Returns:
            Tuple of (distribution_fig, time_series_fig)
        """
        if not trades:
            self.logger.warning("No trades to analyze")
            return None, None
        
        # Extract trade data
        profits = [t.get('profit_pct', 0) for t in trades]
        entry_times = []
        durations = []
        
        for trade in trades:
            # Extract entry time
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            
            if isinstance(entry_time, str):
                entry_time = pd.to_datetime(entry_time)
            
            if exit_time and isinstance(exit_time, str):
                exit_time = pd.to_datetime(exit_time)
            
            entry_times.append(entry_time)
            
            # Calculate duration if possible
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        # Figure 1: Distribution analysis
        dist_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.default_figsize)
        
        # Profit distribution histogram
        sns.histplot(profits, kde=True, ax=ax1, color='skyblue')
        ax1.axvline(0, color='red', linestyle='--')
        ax1.set_title('Profit/Loss Distribution')
        ax1.set_xlabel('Profit/Loss (%)')
        ax1.set_ylabel('Frequency')
        
        # Add mean, median lines
        mean_profit = np.mean(profits)
        median_profit = np.median(profits)
        
        ax1.axvline(mean_profit, color='green', linestyle='--', alpha=0.7)
        ax1.axvline(median_profit, color='blue', linestyle='--', alpha=0.7)
        
        ax1.text(mean_profit, ax1.get_ylim()[1]*0.9, f' Mean: {mean_profit:.2f}%', 
               color='green', ha='left', va='center')
        ax1.text(median_profit, ax1.get_ylim()[1]*0.8, f' Median: {median_profit:.2f}%', 
               color='blue', ha='left', va='center')
        
        # Duration distribution if available
        if durations:
            sns.histplot(durations, kde=True, ax=ax2, color='lightgreen')
            ax2.set_title('Trade Duration Distribution')
            ax2.set_xlabel('Duration (hours)')
            ax2.set_ylabel('Frequency')
            
            # Add mean, median lines
            mean_duration = np.mean(durations)
            median_duration = np.median(durations)
            
            ax2.axvline(mean_duration, color='green', linestyle='--', alpha=0.7)
            ax2.axvline(median_duration, color='blue', linestyle='--', alpha=0.7)
            
            ax2.text(mean_duration, ax2.get_ylim()[1]*0.9, f' Mean: {mean_duration:.2f}h', 
                   color='green', ha='left', va='center')
            ax2.text(median_duration, ax2.get_ylim()[1]*0.8, f' Median: {median_duration:.2f}h', 
                   color='blue', ha='left', va='center')
        
        dist_fig.suptitle(f"{title} - Distribution Analysis")
        plt.tight_layout()
        
        # Figure 2: Time series analysis
        ts_fig, (ax3, ax4) = plt.subplots(2, 1, figsize=self.default_figsize, sharex=True)
        
        # Create DataFrame for time series analysis
        df = pd.DataFrame({
            'entry_time': entry_times,
            'profit_pct': profits
        })
        df.set_index('entry_time', inplace=True)
        df = df.sort_index()
        
        # Individual trade profit/loss
        colors = ['green' if p >= 0 else 'red' for p in df['profit_pct']]
        ax3.bar(df.index, df['profit_pct'], color=colors)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Individual Trade Profit/Loss')
        ax3.set_ylabel('Profit/Loss (%)')
        
        # Cumulative profit
        df['cum_profit'] = df['profit_pct'].cumsum()
        ax4.plot(df.index, df['cum_profit'], color='blue', linewidth=2)
        ax4.set_title('Cumulative Profit/Loss')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Profit (%)')
        
        # Add win/loss streak annotations
        streak_threshold = 3  # Minimum streak length to annotate
        
        # Find winning streaks
        win_mask = df['profit_pct'] > 0
        win_streaks = []
        current_streak = 0
        streak_start = None
        
        for i, (date, is_win) in enumerate(zip(df.index, win_mask)):
            if is_win:
                if current_streak == 0:
                    streak_start = date
                current_streak += 1
            else:
                if current_streak >= streak_threshold:
                    win_streaks.append((streak_start, date, current_streak))
                current_streak = 0
        
        # Add last streak if it's still ongoing
        if current_streak >= streak_threshold:
            win_streaks.append((streak_start, df.index[-1], current_streak))
        
        # Annotate winning streaks
        for start, end, length in win_streaks:
            ax3.axvspan(start, end, color='green', alpha=0.2)
            mid_point = df.index[df.index.get_indexer([start])[0] + length // 2]
            ax3.annotate(f"{length} wins", xy=(mid_point, ax3.get_ylim()[1]*0.8),
                      ha='center', va='center', color='green', fontweight='bold')
        
        # Find losing streaks
        lose_mask = df['profit_pct'] < 0
        lose_streaks = []
        current_streak = 0
        streak_start = None
        
        for i, (date, is_loss) in enumerate(zip(df.index, lose_mask)):
            if is_loss:
                if current_streak == 0:
                    streak_start = date
                current_streak += 1
            else:
                if current_streak >= streak_threshold:
                    lose_streaks.append((streak_start, date, current_streak))
                current_streak = 0
        
        # Add last streak if it's still ongoing
        if current_streak >= streak_threshold:
            lose_streaks.append((streak_start, df.index[-1], current_streak))
        
        # Annotate losing streaks
        for start, end, length in lose_streaks:
            ax3.axvspan(start, end, color='red', alpha=0.2)
            mid_point = df.index[df.index.get_indexer([start])[0] + length // 2]
            ax3.annotate(f"{length} losses", xy=(mid_point, ax3.get_ylim()[0]*0.8),
                      ha='center', va='center', color='red', fontweight='bold')
        
        # Format x-axis
        if isinstance(df.index, pd.DatetimeIndex):
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        ts_fig.suptitle(f"{title} - Time Series Analysis")
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            base_path = os.path.splitext(save_path)[0]
            dist_path = f"{base_path}_distribution.png"
            ts_path = f"{base_path}_timeseries.png"
            
            dist_fig.savefig(dist_path)
            ts_fig.savefig(ts_path)
            
            self.logger.info(f"Saved trade analysis to {dist_path} and {ts_path}")
        
        return dist_fig, ts_fig
    
    def plot_feature_importance(self, feature_importances: Dict[str, float], 
                             title: str = 'Feature Importance', 
                             top_n: int = 20,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importances: Dictionary of feature importances
            title: Chart title
            top_n: Number of top features to display
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        # Sort features by importance
        sorted_features = {k: v for k, v in sorted(feature_importances.items(), 
                                                 key=lambda item: item[1], reverse=True)}
        
        # Take top N features
        top_features = dict(list(sorted_features.items())[:top_n])
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Plot bars
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Normalize importances
        importances_norm = [i / sum(importances) * 100 for i in importances]
        
        # Create color gradient based on importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances_norm, color=colors)
        
        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        
        ax.set_xlabel('Importance (%)')
        ax.set_title(title)
        
        # Add value labels
        for i, v in enumerate(importances_norm):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
        
        # Add grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved feature importance to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                             title: str = 'Feature Correlation Matrix', 
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            columns: List of columns to include (optional)
            title: Chart title
            save_path: Path to save chart (optional)
            
        Returns:
            Matplotlib Figure object
        """
        # Select columns if specified, otherwise use all columns
        if columns:
            df = df[columns]
        
        # Calculate correlation matrix
        corr = df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Create heatmap
        im = sns.heatmap(corr, annot=True, fmt='.2f', ax=ax, cmap='coolwarm', 
                       linewidths=0.5, vmin=-1, vmax=1)
        
        # Set title
        ax.set_title(title)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved correlation matrix to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, backtest_results: Dict[str, Any], 
                             trades: List[Dict[str, Any]],
                             feature_importances: Optional[Dict[str, float]] = None,
                             save_path: Optional[str] = None) -> List[plt.Figure]:
        """
        Create a comprehensive backtest summary dashboard.
        
        Args:
            backtest_results: Dictionary of backtest results
            trades: List of trade dictionaries
            feature_importances: Dictionary of feature importances (optional)
            save_path: Path to save dashboard (optional)
            
        Returns:
            List of Matplotlib Figure objects
        """
        # Extract data
        equity_curve = backtest_results.get('equity_curve', [])
        performance = backtest_results.get('performance', {})
        
        # Create output directory if saving
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Generate charts
        figures = []
        
        # 1. Equity curve with drawdown
        equity_fig = self.plot_equity_curve(
            equity_curve=equity_curve,
            title='Equity Curve with Drawdown',
            save_path=os.path.join(os.path.dirname(save_path), 'equity_curve.png') if save_path else None
        )
        figures.append(equity_fig)
        
        # 2. Detailed drawdown analysis
        drawdown_fig = self.plot_drawdowns(
            equity_curve=equity_curve,
            title='Drawdown Analysis',
            save_path=os.path.join(os.path.dirname(save_path), 'drawdown_analysis.png') if save_path else None
        )
        figures.append(drawdown_fig)
        
        # 3. Trade analysis
        if trades:
            dist_fig, ts_fig = self.plot_trade_analysis(
                trades=trades,
                title='Trade Analysis',
                save_path=os.path.join(os.path.dirname(save_path), 'trade_analysis.png') if save_path else None
            )
            figures.extend([dist_fig, ts_fig])
        
        # 4. Monthly returns
        if isinstance(equity_curve, pd.Series) or len(equity_curve) > 20:
            monthly_fig = self.plot_monthly_returns(
                equity_curve=equity_curve,
                title='Monthly Returns',
                save_path=os.path.join(os.path.dirname(save_path), 'monthly_returns.png') if save_path else None
            )
            figures.append(monthly_fig)
        
        # 5. Feature importance
        if feature_importances:
            feat_fig = self.plot_feature_importance(
                feature_importances=feature_importances,
                title='Feature Importance',
                save_path=os.path.join(os.path.dirname(save_path), 'feature_importance.png') if save_path else None
            )
            figures.append(feat_fig)
        
        # Create summary page with key metrics if saving
        if save_path:
            self._create_summary_page(
                performance=performance,
                save_path=save_path,
                figure_paths=[os.path.join(os.path.dirname(save_path), f) for f in os.listdir(os.path.dirname(save_path)) if f.endswith('.png')]
            )
        
        return figures
    
    def _create_summary_page(self, performance: Dict[str, Any], save_path: str, figure_paths: List[str]) -> None:
        """
        Create HTML summary page with key metrics and charts.
        
        Args:
            performance: Dictionary of performance metrics
            save_path: Path to save HTML
            figure_paths: List of paths to figures
        """
        # HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Strategy Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric-card {{ background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .metric-title {{ font-size: 0.9em; color: #6c757d; margin-bottom: 5px; }}
                .metric-value {{ font-size: 1.8em; font-weight: bold; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .charts {{ display: flex; flex-direction: column; gap: 30px; }}
                .chart {{ background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .chart img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Trading Strategy Dashboard</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
        """
        
        # Add key metrics
        metrics = [
            ('Total Return', f"{performance.get('total_return', 0):.2f}%", performance.get('total_return', 0) >= 0),
            ('Sharpe Ratio', f"{performance.get('sharpe_ratio', 0):.2f}", performance.get('sharpe_ratio', 0) >= 1),
            ('Sortino Ratio', f"{performance.get('sortino_ratio', 0):.2f}", performance.get('sortino_ratio', 0) >= 1),
            ('Max Drawdown', f"{performance.get('max_drawdown', 0):.2f}%", False),
            ('Win Rate', f"{performance.get('win_rate', 0):.2f}%", True),
            ('Profit Factor', f"{performance.get('profit_factor', 0):.2f}", performance.get('profit_factor', 0) >= 1),
            ('Number of Trades', str(performance.get('num_trades', 0)), True),
            ('Avg Profit/Trade', f"{performance.get('avg_profit_per_trade', 0):.2f}%", performance.get('avg_profit_per_trade', 0) >= 0),
        ]
        
        for title, value, is_positive in metrics:
            css_class = 'positive' if is_positive else 'negative'
            html += f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value {css_class}">{value}</div>
                </div>
            """
        
        html += """
            </div>
            
            <div class="charts">
        """
        
        # Add charts
        for path in figure_paths:
            filename = os.path.basename(path)
            title = ' '.join(os.path.splitext(filename)[0].split('_')).title()
            
            html += f"""
                <div class="chart">
                    <h2>{title}</h2>
                    <img src="{filename}" alt="{title}">
                </div>
            """
        
        html += """
            </div>
            
            <div class="footer">
                <p>Generated by the Trading Strategy Visualizer</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(save_path, 'w') as f:
            f.write(html)
        
        self.logger.info(f"Created summary dashboard at {save_path}")
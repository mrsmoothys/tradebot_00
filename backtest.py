#!/usr/bin/env python
"""
Trading Bot - Backtesting

This script handles backtesting of trading strategies.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from config.config_manager import ConfigManager
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.feature_selection import FeatureSelector
from models.universal_model import UniversalModel
from models.rl_model import RLModel
from backtest.backtest_engine import BacktestEngine
from reporting.performance_report import PerformanceReport
from utils.logger import setup_trading_logger
import time
from datetime import timedelta
from utils.profiling import print_profiling_stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest Trading Strategies')
    
    parser.add_argument('--config', type=str, default='config/user_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model')
    parser.add_argument('--output-dir', type=str, help='Directory to save backtest results')
    parser.add_argument('--compare', action='store_true', 
                      help='Compare with benchmark (symbol price)')
    parser.add_argument('--initial-capital', type=float, 
                      help='Initial capital for backtesting')
    
    return parser.parse_args()

def load_model(model_path: str, config_manager) -> Any:
    """
    Load pre-trained model from disk.
    
    Args:
        model_path: Path to model directory
        config_manager: Configuration manager instance
        
    Returns:
        Loaded model
    """
    logger = logging.getLogger(__name__)
    
    # Check model type from metadata
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_type = metadata.get('model_type', None)
    else:
        # Try to infer model type from directory structure
        if os.path.exists(os.path.join(model_path, 'model.h5')):
            model_type = 'rl'
        else:
            model_type = config_manager.get('model.architecture', 'lstm')
    
    # Load model based on type
    if model_type == 'rl':
        logger.info("Loading RL model")
        model = RLModel(config_manager)
        model.load(model_path)
    else:
        logger.info("Loading universal model")
        model = UniversalModel(config_manager)
        model.load(model_path)
    
    logger.info(f"Loaded model from {model_path}")
    
    return model

def run_backtest(config_manager=None, args=None):
    """
    Run backtesting process.
    
    Args:
        config_manager: Configuration manager instance (optional)
        args: Command line arguments (optional)
    """

    # Record the start time for overall process
    total_start_time = time.time()

    # Parse args if not provided
    if args is None:
        args = parse_args()
    
    # Load config if not provided
    if config_manager is None:
        config_manager = ConfigManager(args.config)
    
    # Set up logger
    logger = setup_trading_logger(config_manager)
    
    # Get symbols and timeframes to backtest
    symbols = [args.symbol] if args.symbol else config_manager.get('data.symbols', ['BTCUSDT'])
    timeframes = [args.timeframe] if args.timeframe else config_manager.get('data.timeframes', ['1h'])
    
    # Get date range
    start_date = args.start_date or config_manager.get('data.start_date')
    end_date = args.end_date or config_manager.get('data.end_date')
    
    # Get output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config_manager.get('backtest.output_directory', 'backtest_results')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f"backtest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set initial capital if provided
    if args.initial_capital:
        config_manager.set('backtest.initial_capital', args.initial_capital)
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config_manager)
    
    # Load model if path provided
    model = None
    if args.model_path:
        model = load_model(args.model_path, config_manager)
    
    # Print backtest configuration
    logger.info("Backtest Configuration:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Date Range: {start_date or 'earliest'} to {end_date or 'latest'}")
    logger.info(f"  Initial Capital: ${config_manager.get('backtest.initial_capital', 10000)}")
    logger.info(f"  Fee Rate: {config_manager.get('backtest.fee_rate', 0.0002) * 100:.4f}%")
    
    # Run backtest
    results = backtest_engine.run_backtest(
        symbols=symbols,
        timeframes=timeframes,
        start_date=start_date,
        end_date=end_date,
        model=model
    )
    
    # After all backtests are complete, log the total time and print profiling stats
    total_elapsed = time.time() - total_start_time
    logger.info(f"Total backtesting time: {str(timedelta(seconds=int(total_elapsed)))}")

    # Save backtest results
    backtest_engine.save_results(output_dir)
    
    # Generate performance report
    report_generator = PerformanceReport(config_manager)
    report_path = os.path.join(output_dir, 'backtest_report.html')
    report = report_generator.generate_report(results, report_name=f"backtest_{timestamp}")
    
    logger.info(f"Backtest report generated: {report}")
    
    # Print summary for each symbol/timeframe pair
    logger.info("Backtest Summary:")
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                # Get performance metrics
                performance_metrics = results[symbol][timeframe]['performance_metrics']
                trades = results[symbol][timeframe]['trades']
                
                logger.info(f"  {symbol} {timeframe}:")
                logger.info(f"    Total Return: {performance_metrics.get('total_return', 0):.2f}%")
                logger.info(f"    Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"    Sortino Ratio: {performance_metrics.get('sortino_ratio', 0):.2f}")
                logger.info(f"    Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2f}%")
                logger.info(f"    Win Rate: {performance_metrics.get('win_rate', 0):.2f}%")
                logger.info(f"    Profit Factor: {performance_metrics.get('profit_factor', 0):.2f}")
                logger.info(f"    Number of Trades: {len(trades)}")
            
            except KeyError:
                logger.warning(f"  No results available for {symbol} {timeframe}")
    
    logger.info(f"Backtest completed successfully. Results saved to {output_dir}")
    
    # Compare with benchmark if requested
    if args.compare:
        logger.info("Comparing with benchmark (symbol price)")
        
        # Get benchmark data
        benchmark_results = {}
        
        for symbol in symbols:
            benchmark_results[symbol] = {}
            
            for timeframe in timeframes:
                # Load price data
                data_loader = DataLoader(config_manager)
                df = data_loader.load_data(symbol, timeframe, start_date, end_date)
                
                # Calculate price return
                initial_price = df['close'].iloc[0]
                final_price = df['close'].iloc[-1]
                price_return = (final_price / initial_price - 1) * 100
                
                # Create benchmark results
                benchmark_results[symbol][timeframe] = {
                    'performance_metrics': {
                        'total_return': price_return,
                        'sharpe_ratio': 0,  # Need to calculate
                        'max_drawdown': 0,  # Need to calculate
                        'win_rate': 0,
                        'profit_factor': 0
                    },
                    'equity_curve': df['close'].values
                }
        
        # Create comparison report
        comparison_report_path = os.path.join(output_dir, 'benchmark_comparison.html')
        report_generator.generate_comparison_report(
            results_list=[results, benchmark_results],
            names=['Strategy', 'Buy & Hold'],
            report_name=f"benchmark_comparison_{timestamp}"
        )
        
        logger.info(f"Benchmark comparison report generated: {comparison_report_path}")
        
        # Print benchmark summary
        logger.info("Benchmark Summary (Buy & Hold):")
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    benchmark_return = benchmark_results[symbol][timeframe]['performance_metrics']['total_return']
                    strategy_return = results[symbol][timeframe]['performance_metrics']['total_return']
                    
                    outperformance = strategy_return - benchmark_return
                    
                    logger.info(f"  {symbol} {timeframe}:")
                    logger.info(f"    Benchmark Return: {benchmark_return:.2f}%")
                    logger.info(f"    Strategy Return: {strategy_return:.2f}%")
                    logger.info(f"    Outperformance: {outperformance:.2f}%")
                
                except KeyError:
                    logger.warning(f"  No benchmark results available for {symbol} {timeframe}")
    
    # After all backtests are complete, log the total time and print profiling stats
    total_elapsed = time.time() - total_start_time
    logger.info(f"Total backtesting time: {str(timedelta(seconds=int(total_elapsed)))}")
    
    # Print profiling statistics
    print_profiling_stats()

if __name__ == '__main__':
    run_backtest()
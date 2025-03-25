#!/usr/bin/env python
"""
Trading Bot - Main Entry Point

This script serves as the main entry point for the trading bot system.
It provides a command-line interface for running different components
of the system such as backtesting, training, live trading, etc.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from config.config_manager import ConfigManager
from utils.logger import setup_trading_logger
import train
import backtest
import trade
import time
from datetime import timedelta
from utils.profiling import print_profiling_stats

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the trading model')
    train_parser.add_argument('--config', type=str, default='config/user_config.yaml',
                            help='Path to configuration file')
    train_parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    train_parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    train_parser.add_argument('--start-date', type=str, help='Start date for training (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', type=str, help='End date for training (YYYY-MM-DD)')
    # In main.py, add to the argument parser for the train command
    train_parser.add_argument('--continue-training', type=str, help='Path to existing model to continue training')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest the trading strategy')
    backtest_parser.add_argument('--config', type=str, default='config/user_config.yaml',
                               help='Path to configuration file')
    backtest_parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    backtest_parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    backtest_parser.add_argument('--start-date', type=str, help='Start date for backtesting (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, help='End date for backtesting (YYYY-MM-DD)')
    backtest_parser.add_argument('--model-path', type=str, help='Path to pre-trained model')
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Run live trading')
    trade_parser.add_argument('--config', type=str, default='config/user_config.yaml',
                            help='Path to configuration file')
    trade_parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    trade_parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    trade_parser.add_argument('--model-path', type=str, help='Path to pre-trained model')
    trade_parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    optimize_parser.add_argument('--config', type=str, default='config/user_config.yaml',
                               help='Path to configuration file')
    optimize_parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    optimize_parser.add_argument('--timeframe', type=str, help='Trading timeframe (e.g., 1h)')
    optimize_parser.add_argument('--start-date', type=str, help='Start date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', type=str, help='End date for optimization (YYYY-MM-DD)')
    optimize_parser.add_argument('--param-file', type=str, help='Path to parameter space file')
    
    return parser.parse_args()

def main():
    """Main entry point."""

    # Record the start time for overall process
    total_start_time = time.time()

    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config_path = args.config if hasattr(args, 'config') else 'config/user_config.yaml'
    config_manager = ConfigManager(config_path)
    
    # Set up logger
    logger = setup_trading_logger(config_manager)
    
    # Get version info
    version = "1.0.0"  # Replace with actual version
    
    # Print header
    logger.info("=" * 80)
    logger.info(f"ML Trading Bot v{version}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Override config with command line arguments
    if hasattr(args, 'symbol') and args.symbol:
        config_manager.set('data.symbols', [args.symbol])
    
    if hasattr(args, 'timeframe') and args.timeframe:
        config_manager.set('data.timeframes', [args.timeframe])
    
    if hasattr(args, 'start_date') and args.start_date:
        config_manager.set('data.start_date', args.start_date)
    
    if hasattr(args, 'end_date') and args.end_date:
        config_manager.set('data.end_date', args.end_date)
    
    # Execute command
    if args.command == 'train':
        logger.info("Running model training")
        train.run_training(config_manager, args)
    
    elif args.command == 'backtest':
        logger.info("Running strategy backtesting")
        backtest.run_backtest(config_manager, args)
    
    elif args.command == 'trade':
        logger.info("Running live trading")
        if args.paper:
            logger.info("Paper trading mode enabled")
            config_manager.set('exchange.testnet', True)
        
        trade.run_trading(config_manager, args)
    
    elif args.command == 'optimize':
        logger.info("Running strategy optimization")
        from optimization.hyperparameter_tunning import HyperparameterTuner
        from optimization.feature_optimizer import FeatureOptimizer
        
        # Determine what to optimize (default: strategy parameters)
        optimize_type = config_manager.get('optimization.type', 'hyperparameters')
        
        if optimize_type == 'features':
            logger.info("Optimizing features")
            feature_optimizer = FeatureOptimizer(config_manager)
            
            # Get first symbol and timeframe
            symbols = config_manager.get('data.symbols', ['BTCUSDT'])
            timeframes = config_manager.get('data.timeframes', ['1h'])
            symbol = symbols[0]
            timeframe = timeframes[0]
            
            # Run feature optimization
            feature_optimizer.optimize_features(symbol, timeframe)
            
        else:  # hyperparameters
            logger.info("Optimizing hyperparameters")
            hyperparameter_tuner = HyperparameterTuner(config_manager)
            
            # Load parameter space
            if args.param_file:
                with open(args.param_file, 'r') as f:
                    import yaml
                    param_space = yaml.safe_load(f)
            else:
                # Use default parameter space
                param_space = config_manager.get('optimization.param_space', {})
            
            # Get first symbol and timeframe
            symbols = config_manager.get('data.symbols', ['BTCUSDT'])
            timeframes = config_manager.get('data.timeframes', ['1h'])
            symbol = symbols[0]
            timeframe = timeframes[0]
            
            # Run hyperparameter optimization
            best_params = hyperparameter_tuner.optimize(param_space, symbol, timeframe)
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {hyperparameter_tuner.best_score}")
    
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    # After everything is done, log the total time
    total_elapsed = time.time() - total_start_time
    logger.info(f"Total execution time: {str(timedelta(seconds=int(total_elapsed)))}")
    
    # Print footer
    logger.info("=" * 80)
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    

    # Print profiling statistics
    print_profiling_stats()
    
    return 0

def generate_quality_report(config_manager):
    """Generate comprehensive data quality report."""
    from utils.data_quality import DataQualityMonitor
    monitor = DataQualityMonitor(logging.getLogger("data_quality"))
    
    # Collect metrics from trading session
    report = monitor.generate_report()
    
    # Save report
    report_path = os.path.join(
        config_manager.get('reporting.output_directory', 'reports'),
        f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(report_path, 'w') as f:
        import json
        json.dump(report, f, indent=2, default=str)
    
    logging.getLogger(__name__).info(f"Data quality report generated: {report_path}")
    
    return report

if __name__ == '__main__':
    sys.exit(main())
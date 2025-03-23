# ML Trading Bot

This is a comprehensive machine learning-based trading bot system designed for algorithmic trading. The project includes modules for data processing, feature engineering, model training, backtesting, optimization, and live trading execution.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Workflow](#workflow)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Backtesting](#backtesting)
  - [Optimization](#optimization)
  - [Live Trading](#live-trading)
- [Command Line Interface](#command-line-interface)
- [Performance Reporting](#performance-reporting)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Features

- **Data Processing**: Load and preprocess OHLCV data from various sources
- **Feature Engineering**: Generate technical indicators and select important features
- **Model Training**: Train machine learning models (LSTM, GRU, CNN, Transformer) and reinforcement learning models (DQN)
- **Backtesting**: Test strategies on historical data with realistic trading conditions
- **Optimization**: Tune hyperparameters and strategy parameters using grid search, random search, or Bayesian optimization
- **Live Trading**: Execute strategies in real-time or paper trading mode
- **Performance Analysis**: Generate comprehensive performance reports with visualizations
- **Risk Management**: Adaptive position sizing, stop-loss, and take-profit mechanisms

## Project Structure

```
tradebot/
├── backtest/                 # Backtesting engine and performance metrics
├── config/                   # Configuration files
├── data/                     # Data loading and feature engineering
├── execution/                # Order execution and exchange connections
├── models/                   # ML and RL model implementations
├── optimization/             # Hyperparameter tuning and optimization
├── reporting/                # Performance reporting and visualization
├── strategy/                 # Trading strategy implementations
├── utils/                    # Utility functions and logging
├── main.py                   # Main entry point
├── train.py                  # Model training script
├── backtest.py               # Backtesting script
└── trade.py                  # Live trading script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tradebot.git
   cd tradebot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   The following packages are required:
   - pandas
   - numpy
   - tensorflow
   - scikit-learn
   - matplotlib
   - seaborn
   - ta (technical analysis library)
   - pyyaml
   - requests
   - plotly (for interactive charts)

## Configuration

The bot uses YAML files for configuration. The main configuration file is located at `config/user_config.yaml`. You can also create custom configuration files and specify them using the `--config` parameter.

Key configuration sections include:

- `data`: Data sources, symbols, and timeframes
- `features`: Technical indicators and feature selection methods
- `model`: Model architecture and training parameters
- `strategy`: Trading strategy parameters and risk management
- `backtest`: Backtesting parameters
- `optimization`: Hyperparameter optimization settings
- `execution`: Exchange connection and order execution settings
- `logging`: Logging configuration

Example configuration:

```yaml
data:
  directory: "/path/to/data"
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["1h", "4h", "1d"]
  start_date: "2021-01-01"
  end_date: "2022-12-31"

model:
  architecture: "lstm"
  lookback_window: 60
  prediction_horizon: 5
  hidden_layers: [128, 64, 32]
  dropout_rate: 0.2
```

## Workflow

This section outlines the recommended workflow for using the trading bot.

### Data Preparation

Before training models or running backtests, you need data. The bot expects data in CSV format with columns for OHLCV (Open, High, Low, Close, Volume).

1. Place your data files in the directory specified in the configuration (`data.directory`).
   - File naming convention: `{symbol}_{timeframe}_data_{date}.csv`
   - Example: `BTCUSDT_1h_data_20210101.csv`

2. Alternatively, you can use the exchange connector to download data:
   ```python
   from execution.exchange_connector import BinanceConnector
   from config.config_manager import ConfigManager

   config = ConfigManager('config/user_config.yaml')
   connector = BinanceConnector(config)
   
   # Download 1 year of hourly data for BTCUSDT
   data = connector.get_historical_data(
       symbol='BTCUSDT',
       timeframe='1h',
       start_time='2022-01-01',
       end_time='2022-12-31',
       limit=1000
   )
   
   # Save to CSV
   data.to_csv('data/BTCUSDT_1h_data_20220101.csv')
   ```

### Model Training

Train a model with the following command:

```bash
python main.py train --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h
```

Or use the training script directly with more options:

```bash
python train.py --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --start-date 2021-01-01 --end-date 2022-12-31 --epochs 100 --batch-size 32
```

The training process will:
1. Load and preprocess the data
2. Generate features (technical indicators)
3. Train the model (LSTM, GRU, etc. based on configuration)
4. Save the trained model to the `models` directory
5. Run a validation backtest
6. Generate a performance report

Trained models are saved in the `models` directory with the naming convention `{symbol}_{timeframe}_{timestamp}`.

### Backtesting

Backtest a strategy with a trained model:

```bash
python main.py backtest --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --model-path models/BTCUSDT_1h_20230101_120000
```

Or use the backtesting script directly with more options:

```bash
python backtest.py --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --start-date 2022-01-01 --end-date 2022-12-31 --model-path models/BTCUSDT_1h_20230101_120000 --initial-capital 10000 --compare
```

The backtest will:
1. Load and preprocess the data
2. Generate features (technical indicators)
3. Load the trained model
4. Execute the trading strategy on historical data
5. Calculate performance metrics
6. Generate a performance report

Results are saved in the `backtest_results` directory.

### Optimization

Optimize strategy parameters or model hyperparameters:

```bash
python main.py optimize --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --param-file optimization/params.yaml
```

The optimization process will:
1. Load and preprocess the data
2. Set up the parameter space to explore
3. Search for the best parameters using the specified method (grid search, random search, or Bayesian optimization)
4. Save the optimization results

Example parameter file (`optimization/params.yaml`):

```yaml
model.learning_rate: [0.0001, 0.001, 0.01]
model.dropout_rate: [0.1, 0.2, 0.3, 0.4]
model.hidden_layers: [[64, 32], [128, 64], [256, 128, 64]]
strategy.risk_per_trade: [1.0, 2.0, 3.0]
strategy.stop_loss.atr_multiplier: [1.5, 2.0, 2.5, 3.0]
```

For random search or Bayesian optimization, specify ranges instead:

```yaml
model.learning_rate: [0.0001, 0.01]  # min, max
model.dropout_rate: [0.1, 0.5]       # min, max
strategy.risk_per_trade: [1.0, 5.0]  # min, max
```

After finding the best parameters, update your configuration file to use them.

### Live Trading

Run live trading with a trained model:

```bash
python main.py trade --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --model-path models/BTCUSDT_1h_20230101_120000 --paper
```

The `--paper` flag enables paper trading mode, which uses the exchange's testnet API instead of real funds.

For real trading, you need to:
1. Set up API keys in your configuration file
2. Remove the `--paper` flag
3. Set `exchange.testnet` to `false` in your configuration

Live trading will:
1. Connect to the exchange
2. Load market data
3. Generate features
4. Load the trained model
5. Execute the trading strategy in real-time
6. Log trades and performance metrics

## Command Line Interface

The bot provides a command-line interface for all major functions:

```bash
python main.py [command] [options]
```

Available commands:
- `train`: Train a model
- `backtest`: Run backtesting
- `trade`: Run live trading
- `optimize`: Optimize hyperparameters

Common options:
- `--config`: Path to configuration file
- `--symbol`: Trading symbol (e.g., BTCUSDT)
- `--timeframe`: Trading timeframe (e.g., 1h)
- `--start-date`: Start date for data (YYYY-MM-DD)
- `--end-date`: End date for data (YYYY-MM-DD)
- `--model-path`: Path to trained model directory

For help on available options:

```bash
python main.py [command] --help
```

## Performance Reporting

The bot generates performance reports in HTML format. Reports include:

- Equity curve
- Drawdown analysis
- Trade statistics
- Monthly returns
- Performance metrics (Sharpe ratio, Sortino ratio, win rate, etc.)

Reports are saved in the `reports` directory for backtesting and in the `models` directory for training.

To compare multiple strategies or parameters, use the benchmark comparison feature:

```bash
python backtest.py --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h --compare
```

This will generate a comparison report between your strategy and a buy-and-hold benchmark.

## Advanced Usage

### Using Multiple Timeframes

You can use multiple timeframes in your strategy by configuring them in your config file:

```yaml
data:
  symbols: ["BTCUSDT"]
  timeframes: ["1h", "4h", "1d"]
```

Then run training, backtesting, or optimization as usual. The system will process data for all specified timeframes.

### Feature Engineering

You can customize the technical indicators generated by editing the `features.indicators` section in your config:

```yaml
features:
  indicators:
    - name: "rsi"
      params: {window: 14}
    - name: "macd"
      params: {fast: 12, slow: 26, signal: 9}
    - name: "bollinger_bands"
      params: {window: 20, num_std: 2}
```

### Using Reinforcement Learning

To use reinforcement learning instead of supervised learning:

```yaml
model:
  architecture: "rl"
  rl:
    algorithm: "dqn"  # or "dueling_dqn"
    gamma: 0.99
    epsilon: 1.0
    epsilon_decay: 0.995
```

Then train as usual:

```bash
python main.py train --config config/user_config.yaml --symbol BTCUSDT --timeframe 1h
```

### Customizing Risk Management

The risk management system is customizable via the `strategy` section:

```yaml
strategy:
  risk_per_trade: 2.0  # % of capital risked per trade
  adaptive_sl_tp: true
  trailing_stop: true
  stop_loss:
    atr_multiplier: 2.0
  take_profit:
    min_risk_reward_ratio: 1.5
```

## Troubleshooting

### Data Loading Issues

- Check that your data directory matches the one in your configuration
- Verify that your CSV files follow the naming convention: `{symbol}_{timeframe}_data_{date}.csv`
- Ensure that your CSV files have the required columns: timestamp, open, high, low, close, volume

### Model Training Issues

- Check your GPU configuration if training is slow (TensorFlow should use GPU if available)
- Adjust batch size if you encounter memory errors
- Try reducing the model complexity if training is unstable

### Backtest Performance Issues

- If backtesting is slow, consider using a smaller date range or fewer symbols/timeframes
- Check for data leakage in feature generation (using future data to predict past events)
- Verify that your strategy parameters make sense (stop-loss, take-profit, etc.)

### Exchange Connection Issues

- Verify your API keys are valid and have the necessary permissions
- Check your internet connection and firewall settings
- Ensure the exchange is operational and not in maintenance mode

If you encounter any other issues, check the log files in the `logs` directory for more detailed error messages.



python main.py backtest --config config/my_first_run.yaml --symbol BTCUSDT --timeframe 1h --model-path /Users/mrsmoothy/Desktop/rsidtrade/tradebot_0/models/BTCUSDT_1h_20250321_145524



/Users/mrsmoothy/Desktop/rsidtrade/tradebot_0/models/BTCUSDT_1h_20250321_145524

python main.py train --config config/default_config.yaml --symbol BTCUSDT --timeframe 1h  
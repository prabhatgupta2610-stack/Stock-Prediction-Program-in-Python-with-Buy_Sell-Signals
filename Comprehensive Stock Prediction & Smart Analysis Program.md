# Comprehensive Stock Prediction & Smart Analysis Program

This is the final, most advanced version of the Python-based stock prediction program. It integrates a full suite of price features, technical indicators, and "Smart Features" to provide a holistic market analysis and prediction.

## Features

- **Full Price Features**: Open, High, Low, Close, and Volume.
- **Advanced Technical Indicators**:
  - **RSI**: Relative Strength Index (14-day).
  - **MACD**: Moving Average Convergence Divergence.
  - **SMA 20 & SMA 50**: Simple Moving Averages for short and medium-term trends.
  - **EMA**: Exponential Moving Average.
  - **Bollinger Bands**: High, Low, and Mid bands for volatility and price levels.
  - **ATR**: Average True Range for volatility measurement.
- **Smart Features**:
  - **News Sentiment Score**: Analyzes the latest news headlines for the ticker using `TextBlob`.
  - **Fear & Greed Index**: Real-time index value from CNN Business (Extreme Fear to Extreme Greed).
  - **Sector Trend**: 1-month trend analysis of the relevant sector ETF (e.g., XLK for Tech).
  - **Market Index Trend**: 1-month trend analysis of the S&P 500 (SPY).
- **Multi-Model Comparison**: Evaluates Linear Regression, Random Forest, and XGBoost using MAE, RMSE, and R² Score.
- **Final Signal Logic**: A weighted decision engine that outputs **BUY**, **SELL**, or **HOLD** based on the convergence of technical and smart features.

## Setup and Installation

To run this program, you need Python 3 and the following libraries:

```bash
sudo pip3 install yfinance pandas numpy matplotlib ta textblob fear-and-greed xgboost
```

## How to Run

1. Save the provided Python code into two files: `stock_predictor_final_lib.py` and `main_predictor_final.py`.
2. Run the `main_predictor_final.py` script from your terminal:

```bash
python3 main_predictor_final.py [TICKER_SYMBOL]
```

**Example:**

```bash
python3 main_predictor_final.py MSFT
```

## Output

The program will display a comprehensive console output and save a final analysis chart:

- `[TICKER_SYMBOL]_final_analysis.png`: A 3-panel chart including:
  - Price action with Bollinger Bands and the Final Signal.
  - RSI and ATR (Volatility) overlay.
  - Model prediction comparison on the test set.
  - Smart Feature metrics (Sentiment, Fear & Greed, Market Trend).

## Program Structure

- `stock_predictor_final_lib.py`: The core library with all data fetching, indicator calculations, smart feature integration, and plotting functions.
- `main_predictor_final.py`: The main entry point that orchestrates the full analysis and displays results.

**Disclaimer**: This program is for educational and practice purposes only and should not be used for real financial trading decisions. Stock market investments involve risks, and past performance is not indicative of future results.

# Stock Prediction Program

This program is a Python-based stock prediction tool that leverages historical stock price data, technical indicators, and a simple machine learning model to predict future stock price movements and suggest potential buy/sell timings. It also includes a basic sentiment analysis component using simulated news headlines and visualizes the results.

## Features

- **Historical Stock Price Data**: Fetches Open, High, Low, Close, and Volume data using `yfinance`.
- **Technical Indicators**: Calculates and incorporates:
  - Moving Average (MA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Volume Trend
  - Support and Resistance levels (simplified)
- **Market Sentiment**: Basic sentiment analysis from simulated news headlines using `TextBlob`.
- **Trend Prediction**: Predicts stock price movement for the next 1, 5, and 30 days using a `RandomForestRegressor`.
- **Buy/Sell Signals**: Generates signals based on RSI and MA crossovers.
- **Accuracy Score**: Provides an R-squared score for the 5-day prediction using a train/test split.
- **Visualization**: Generates charts for predicted vs. actual prices and historical prices with buy/sell markers.

## Setup and Installation

To run this program, you need Python 3 and the following libraries. You can install them using pip:

```bash
sudo pip3 install yfinance pandas numpy scikit-learn matplotlib ta textblob
```

## How to Run

1. Save the provided Python code into two files: `stock_predictor_lib.py` and `main_predictor.py`.
2. Run the `main_predictor.py` script from your terminal. You can optionally provide a stock ticker symbol as a command-line argument. If no ticker is provided, it defaults to AAPL.

```bash
python3 main_predictor.py [TICKER_SYMBOL]
```

**Example:**

```bash
python3 main_predictor.py AAPL
```

## Output

The program will print prediction results to the console and save two charts in the same directory where the script is executed:

- `[TICKER_SYMBOL]_analysis.png`: Visualizes historical prices, moving averages, Bollinger Bands, and buy/sell signals.
- `[TICKER_SYMBOL]_accuracy.png`: Compares actual vs. predicted prices for the 5-day forecast.

## Program Structure

- `stock_predictor_lib.py`: Contains functions for data fetching, technical indicator calculation, sentiment analysis, machine learning model training/prediction, signal generation, and plotting.
- `main_predictor.py`: The main script that orchestrates the execution flow, calls functions from `stock_predictor_lib.py`, and prints the results.

## Limitations

- **Sentiment Analysis**: Currently uses simulated headlines for demonstration purposes. For real-world applications, integration with a news API would be required.
- **Prediction Accuracy**: The accuracy of stock price prediction is inherently challenging. The model used here is a basic example and should not be used for actual financial decisions.
- **Support and Resistance**: The implementation of support and resistance levels is a simplified rolling min/max and may not reflect true market dynamics.
- **Technical Indicators**: While several indicators are included, advanced traders may use more complex or proprietary indicators.

**Disclaimer**: This program is for educational and practice purposes only and should not be used for real financial trading decisions. Stock market investments involve risks, and past performance is not indicative of future results.

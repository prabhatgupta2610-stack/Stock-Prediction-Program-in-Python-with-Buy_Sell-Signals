import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from textblob import TextBlob

def get_stock_data(ticker, period="2y", interval="1d"):
    """Fetch historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    # Moving Averages
    df['MA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['MA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    # Volume Trend (Simple moving average of volume)
    df['Vol_MA20'] = SMAIndicator(close=df['Volume'], window=20).sma_indicator()
    
    # Support and Resistance (Simplified using rolling min/max)
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    
    return df

def get_sentiment(headlines):
    """Calculate average sentiment from a list of headlines."""
    if not headlines:
        return 0.5  # Neutral
    sentiments = [TextBlob(h).sentiment.polarity for h in headlines]
    # Map -1 to 1 into 0 to 1 range
    avg_sentiment = np.mean(sentiments)
    return (avg_sentiment + 1) / 2

def prepare_features(df):
    """Prepare features for machine learning."""
    df = df.copy()
    # Drop rows with NaN values from indicators
    df.dropna(inplace=True)
    
    # Define features
    features = ['Close', 'MA20', 'MA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 'Vol_MA20', 'BB_High', 'BB_Low']
    X = df[features]
    
    return X, df

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_and_predict(df, forecast_days):
    """Train a model and predict future prices."""
    # Create target: price shifted by forecast_days
    df = df.copy()
    df['Target'] = df['Close'].shift(-forecast_days)
    
    # Prepare features and targets
    features = ['Close', 'MA20', 'MA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 'Vol_MA20', 'BB_High', 'BB_Low']
    
    # Valid data for training (exclude last forecast_days)
    train_df = df.dropna(subset=['Target'])
    X = train_df[features]
    y = train_df['Target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    # Predict future (last row)
    last_row = df[features].tail(1)
    prediction = model.predict(last_row)[0]
    
    return prediction, score, y_test, y_pred

def generate_signals(df):
    """Generate buy/sell signals based on technical indicators."""
    df = df.copy()
    df['Signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
    
    # Buy signal: RSI < 30 (oversold) and price crosses above MA20 (bullish)
    df.loc[(df['RSI'] < 30) & (df['Close'] > df['MA20']), 'Signal'] = 1
    
    # Sell signal: RSI > 70 (overbought) and price crosses below MA20 (bearish)
    df.loc[(df['RSI'] > 70) & (df['Close'] < df['MA20']), 'Signal'] = -1
    
    return df

import matplotlib.pyplot as plt

def plot_predictions(df, ticker):
    """Visualize historical prices and buy/sell signals."""
    plt.figure(figsize=(14, 7))
    
    # Plot price and moving averages
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    plt.plot(df.index, df['MA20'], label='MA 20', color='orange', linestyle='--')
    plt.plot(df.index, df['MA50'], label='MA 50', color='green', linestyle='--')
    
    # Plot Bollinger Bands
    plt.fill_between(df.index, df['BB_High'], df['BB_Low'], color='lightgrey', alpha=0.3, label='Bollinger Bands')
    
    # Plot Buy/Sell signals
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)
    
    plt.title(f'Stock Price Analysis for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_analysis.png')
    plt.close()

def plot_accuracy(y_test, y_pred, ticker):
    """Plot predicted vs actual prices."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Price', color='blue')
    plt.plot(y_pred, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Actual vs Predicted Prices for {ticker}')
    plt.xlabel('Sample')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_accuracy.png')
    plt.close()

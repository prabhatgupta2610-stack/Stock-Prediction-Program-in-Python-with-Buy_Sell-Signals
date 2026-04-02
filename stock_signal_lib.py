import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator

def get_stock_data(ticker, period="2y"):
    """Fetch historical stock data."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

def calculate_signals(df):
    """Calculate refined buy and sell signals based on specific conditions."""
    df = df.copy()
    
    # 1. Indicators
    df['MA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Volume metrics
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Vol_MA20'] * 1.5)
    
    # Resistance (20-day high)
    df['Resistance'] = df['High'].rolling(window=20).max()
    
    # 2. Condition checks
    # RSI rising check
    df['RSI_Rising'] = df['RSI'] > df['RSI'].shift(1)
    
    # MACD Crossovers
    df['MACD_Bullish_Crossover'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    df['MACD_Bearish_Crossover'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    # Volume Divergence (Simplified: Price rising, Volume falling over 5 days)
    df['Price_Rising_5d'] = df['Close'] > df['Close'].shift(5)
    df['Vol_Falling_5d'] = df['Volume'] < df['Volume'].shift(5)
    df['Neg_Vol_Divergence'] = df['Price_Rising_5d'] & df['Vol_Falling_5d']
    
    # Resistance Proximity (within 2% of 20-day high)
    df['Near_Resistance'] = df['Close'] >= (df['Resistance'] * 0.98)
    
    # 3. Signal Logic
    df['Signal'] = 'HOLD'
    
    # Buy conditions:
    # - RSI below 30 and rising
    # - MACD crossover bullish
    # - Price above 50-day moving average
    # - Volume spike confirmation
    buy_mask = (
        (df['RSI'] < 30) & (df['RSI_Rising']) & 
        (df['MACD_Bullish_Crossover']) & 
        (df['Close'] > df['MA50']) & 
        (df['Volume_Spike'])
    )
    
    # Sell conditions:
    # - RSI above 70
    # - MACD bearish crossover
    # - Price near resistance
    # - Negative volume divergence
    sell_mask = (
        (df['RSI'] > 70) | 
        (df['MACD_Bearish_Crossover']) | 
        (df['Near_Resistance']) | 
        (df['Neg_Vol_Divergence'])
    )
    
    df.loc[buy_mask, 'Signal'] = 'BUY'
    df.loc[sell_mask, 'Signal'] = 'SELL'
    
    return df

import matplotlib.pyplot as plt

def plot_signals(df, ticker):
    """Visualize the stock signals and indicators."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Price, MA50, and signals
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    ax1.plot(df.index, df['MA50'], label='MA 50', color='orange', linestyle='--')
    
    buy_signals = df[df['Signal'] == 'BUY']
    sell_signals = df[df['Signal'] == 'SELL']
    
    ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='BUY Signal', s=100)
    ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='SELL Signal', s=100)
    
    ax1.set_title(f'Stock Signal Analysis for {ticker}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Volume and Spikes
    ax3.bar(df.index, df['Volume'], label='Volume', color='gray', alpha=0.5)
    ax3.plot(df.index, df['Vol_MA20'], label='Vol MA 20', color='black', linestyle=':')
    
    spikes = df[df['Volume_Spike']]
    ax3.scatter(spikes.index, spikes['Volume'], marker='o', color='red', label='Volume Spike', s=20)
    
    ax3.set_ylabel('Volume')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_refined_signals.png')
    plt.close()

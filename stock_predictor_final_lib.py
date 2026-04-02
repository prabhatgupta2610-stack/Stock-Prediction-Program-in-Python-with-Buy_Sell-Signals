import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from textblob import TextBlob
import fear_and_greed
import datetime

def get_stock_data(ticker, period="5y"):
    """Fetch historical stock data including all price features."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    # Ensure all required price features are present
    # yfinance history usually includes: Open, High, Low, Close, Volume, Dividends, Stock Splits
    # Adjusted Close is often same as Close for yf.history unless auto_adjust=False
    return df

def get_market_trends():
    """Fetch market and sector trends."""
    # Market Index: S&P 500
    spy = yf.Ticker("SPY").history(period="1mo")
    market_trend = (spy['Close'].iloc[-1] / spy['Close'].iloc[0]) - 1
    
    # Sector Trends (Example: Tech sector XLK)
    xlk = yf.Ticker("XLK").history(period="1mo")
    sector_trend = (xlk['Close'].iloc[-1] / xlk['Close'].iloc[0]) - 1
    
    return market_trend, sector_trend

def get_fear_greed():
    """Fetch current Fear & Greed Index."""
    try:
        data = fear_and_greed.get()
        return data.value, data.description
    except Exception:
        return 50, "neutral"

def add_comprehensive_indicators(df):
    """Add all requested technical indicators."""
    df = df.copy()
    
    # Technical Indicators
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    
    bb = BollingerBands(close=df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Drop NaNs from indicators
    df.dropna(inplace=True)
    return df

def get_sentiment_score(ticker):
    """Fetch news and calculate sentiment score."""
    stock = yf.Ticker(ticker)
    news = stock.news
    if not news:
        return 0.5
    
    sentiments = []
    for item in news:
        title = item.get('title', '')
        blob = TextBlob(title)
        sentiments.append((blob.sentiment.polarity + 1) / 2) # Map -1,1 to 0,1
        
    return np.mean(sentiments) if sentiments else 0.5

def generate_final_signal(df, sentiment, fear_greed_val, market_trend, sector_trend):
    """Generate final signal using all smart features."""
    latest = df.iloc[-1]
    
    # Buy Score Components
    rsi_buy = latest['RSI'] < 35
    macd_buy = latest['MACD'] > latest['MACD_Signal']
    price_above_ma = latest['Close'] > latest['SMA50']
    sentiment_buy = sentiment > 0.6
    fear_buy = fear_greed_val < 30 # Extreme fear often a buy signal
    market_buy = market_trend > 0
    
    score = sum([rsi_buy, macd_buy, price_above_ma, sentiment_buy, fear_buy, market_buy])
    
    if score >= 4:
        return "BUY"
    elif score <= 1:
        return "SELL"
    else:
        return "HOLD"

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_models(df):
    """Train and evaluate multiple models with comprehensive features."""
    df = df.copy()
    df['Target'] = df['Close'].shift(-1)
    
    # Define features
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 
        'SMA20', 'SMA50', 'EMA', 'BB_High', 'BB_Low', 'BB_Mid', 'ATR'
    ]
    
    # Prepare data
    data = df.dropna(subset=['Target'])
    X = data[features]
    y = data['Target']
    
    # Train/Test Split (Time series aware)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        predictions[name] = y_pred
        
    return pd.DataFrame(results), predictions, y_test

def plot_final_analysis(df, ticker, predictions, y_test, signal, sentiment, fear_greed_val):
    """Create final comprehensive visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    # Plot 1: Price and Bollinger Bands
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    ax1.plot(df.index, df['BB_High'], label='BB High', color='gray', linestyle='--', alpha=0.3)
    ax1.plot(df.index, df['BB_Low'], label='BB Low', color='gray', linestyle='--', alpha=0.3)
    ax1.fill_between(df.index, df['BB_High'], df['BB_Low'], color='lightgray', alpha=0.2)
    ax1.set_title(f'Comprehensive Analysis for {ticker} | Final Signal: {signal}')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI and ATR
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2_right = ax2.twinx()
    ax2_right.plot(df.index, df['ATR'], label='ATR (Volatility)', color='orange', alpha=0.5)
    ax2.set_ylabel('RSI')
    ax2_right.set_ylabel('ATR')
    ax2.legend(loc='upper left')
    ax2_right.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot 3: Prediction Comparison (Test Set)
    test_indices = df.index[-len(y_test):]
    ax3.plot(test_indices, y_test.values, label='Actual Price', color='black', linewidth=2)
    for name, y_pred in predictions.items():
        ax3.plot(test_indices, y_pred, label=f'{name} Predicted', alpha=0.7)
    ax3.set_title('Model Prediction Comparison (Test Set)')
    ax3.set_ylabel('Price')
    ax3.legend()
    ax3.grid(True)
    
    # Smart Features Annotation
    plt.figtext(0.1, 0.01, f"Sentiment Score: {sentiment:.2f} | Fear & Greed: {fear_greed_val:.2f} | Market Index Trend: {sentiment:.2%}", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_final_analysis.png')
    plt.close()

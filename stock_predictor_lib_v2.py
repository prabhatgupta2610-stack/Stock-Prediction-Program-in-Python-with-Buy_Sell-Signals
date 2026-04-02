import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

def get_stock_data(ticker, period="5y", interval="1d"):
    """Fetch historical stock data."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def add_advanced_features(df):
    """Add advanced features for better prediction accuracy."""
    df = df.copy()
    
    # 1. Basic Technical Indicators
    df['MA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['MA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    bb = BollingerBands(close=df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    # 2. Lag Features (previous 5, 10, 20 day prices)
    for lag in [5, 10, 20]:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    # 3. Rolling Averages & Volatility
    df['Rolling_Std_20'] = df['Close'].rolling(window=20).std()
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # 4. Momentum Score (Rate of Change)
    df['Momentum_10'] = df['Close'].pct_change(periods=10)
    
    # 5. Daily Return Percentage
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 6. Trend Strength Score (ADX)
    adx = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ADX'] = adx.adx()
    
    # 7. Price Breakout Detection
    df['Upper_Channel'] = df['High'].rolling(window=20).max()
    df['Lower_Channel'] = df['Low'].rolling(window=20).min()
    df['Breakout_High'] = (df['Close'] > df['Upper_Channel'].shift(1)).astype(int)
    df['Breakout_Low'] = (df['Close'] < df['Lower_Channel'].shift(1)).astype(int)
    
    # Drop rows with NaN values created by indicators/lags
    df.dropna(inplace=True)
    
    return df

def prepare_ml_data(df, target_col='Close', forecast_out=1):
    """Prepare features and target for ML models."""
    df = df.copy()
    df['Target'] = df[target_col].shift(-forecast_out)
    
    # Define feature columns
    features = [
        'Close', 'MA20', 'MA50', 'EMA20', 'RSI', 'MACD', 'MACD_Signal', 
        'BB_High', 'BB_Low', 'Lag_5', 'Lag_10', 'Lag_20', 
        'Rolling_Std_20', 'Volatility', 'Momentum_10', 'Daily_Return', 
        'ADX', 'Breakout_High', 'Breakout_Low'
    ]
    
    # Data for training (rows where Target is not NaN)
    data = df.dropna(subset=['Target'])
    X = data[features]
    y = data['Target']
    
    # Most recent data for future prediction
    X_latest = df[features].tail(1)
    
    return X, y, X_latest, features

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_linear_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_random_forest(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_xgboost(X_train, y_train, X_test):
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_lstm(X_train, y_train, X_test, input_shape):
    # Reshape for LSTM: (samples, time_steps, features)
    X_train_reshaped = np.array(X_train).reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = np.array(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_reshaped, y_train, batch_size=32, epochs=10, verbose=0)
    
    y_pred = model.predict(X_test_reshaped).flatten()
    return model, y_pred

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

import matplotlib.pyplot as plt

def plot_comparison(results_df, ticker):
    """Plot performance metrics comparison across models."""
    metrics = ['MAE', 'RMSE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        results_df.plot(kind='bar', x='Model', y=metric, ax=axes[i], legend=False, color='skyblue')
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Model Performance Comparison for {ticker}')
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_model_comparison.png')
    plt.close()

def plot_predictions_comparison(y_test, predictions, ticker):
    """Plot actual vs predicted prices for each model."""
    plt.figure(figsize=(14, 8))
    plt.plot(y_test.values, label='Actual Price', color='black', linewidth=2)
    
    colors = ['blue', 'green', 'red', 'orange']
    for (model_name, y_pred), color in zip(predictions.items(), colors):
        plt.plot(y_pred, label=f'{model_name} Predicted', color=color, alpha=0.7)
    
    plt.title(f'Actual vs Predicted Prices for {ticker} (Test Set)')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/{ticker}_predictions_overlay.png')
    plt.close()

import stock_predictor_lib_v2 as lib
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main(ticker="AAPL"):
    print(f"--- Improved Stock Prediction & Model Comparison for {ticker} ---")
    
    # 1. Fetch and process data
    print("Fetching historical data...")
    df = lib.get_stock_data(ticker)
    
    print("Performing advanced feature engineering...")
    df = lib.add_advanced_features(df)
    
    # 2. Prepare ML data
    print("Preparing training and test datasets...")
    X, y, X_latest, features = lib.prepare_ml_data(df)
    
    # Train/Test Split (Time series aware)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scaling (Essential for Linear Regression and LSTM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_latest_scaled = scaler.transform(X_latest)
    
    # 3. Model Comparison
    models_results = []
    predictions_overlay = {}
    
    # --- Linear Regression ---
    print("Training Linear Regression...")
    lr_model, lr_pred = lib.train_linear_regression(X_train_scaled, y_train, X_test_scaled)
    mae, rmse, r2 = lib.evaluate_model(y_test, lr_pred)
    models_results.append({'Model': 'Linear Regression', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    predictions_overlay['Linear Regression'] = lr_pred
    
    # --- Random Forest ---
    print("Training Random Forest...")
    rf_model, rf_pred = lib.train_random_forest(X_train, y_train, X_test)
    mae, rmse, r2 = lib.evaluate_model(y_test, rf_pred)
    models_results.append({'Model': 'Random Forest', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    predictions_overlay['Random Forest'] = rf_pred
    
    # --- XGBoost ---
    print("Training XGBoost...")
    xgb_model, xgb_pred = lib.train_xgboost(X_train, y_train, X_test)
    mae, rmse, r2 = lib.evaluate_model(y_test, xgb_pred)
    models_results.append({'Model': 'XGBoost', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    predictions_overlay['XGBoost'] = xgb_pred
    
    # --- LSTM ---
    print("Training LSTM...")
    input_shape = (1, len(features))
    lstm_model, lstm_pred = lib.train_lstm(X_train_scaled, y_train, X_test_scaled, input_shape)
    mae, rmse, r2 = lib.evaluate_model(y_test, lstm_pred)
    models_results.append({'Model': 'LSTM', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    predictions_overlay['LSTM'] = lstm_pred
    
    # 4. Results Comparison
    results_df = pd.DataFrame(models_results)
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))
    
    # 5. Visualization
    print("\nGenerating comparison charts...")
    lib.plot_comparison(results_df, ticker)
    lib.plot_predictions_comparison(y_test, predictions_overlay, ticker)
    
    # 6. Future Prediction (Using the best model based on R2)
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    print(f"\nBest Model: {best_model_name}")
    
    current_price = df['Close'].iloc[-1]
    if best_model_name == 'Linear Regression':
        future_pred = lr_model.predict(X_latest_scaled)[0]
    elif best_model_name == 'Random Forest':
        future_pred = rf_model.predict(X_latest)[0]
    elif best_model_name == 'XGBoost':
        future_pred = xgb_model.predict(X_latest)[0]
    else: # LSTM
        X_latest_reshaped = np.array(X_latest_scaled).reshape((1, 1, len(features)))
        future_pred = lstm_model.predict(X_latest_reshaped).flatten()[0]
    
    trend = "Bullish" if future_pred > current_price else "Bearish"
    print(f"Current Price: ${current_price:.2f}")
    print(f"Next Day Prediction ({best_model_name}): ${future_pred:.2f} ({trend})")
    
    print(f"\nCharts saved to: /home/ubuntu/{ticker}_model_comparison.png and /home/ubuntu/{ticker}_predictions_overlay.png")

if __name__ == "__main__":
    ticker = "AAPL"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    main(ticker)

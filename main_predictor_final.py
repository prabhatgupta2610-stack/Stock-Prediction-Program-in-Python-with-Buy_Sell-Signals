import stock_predictor_final_lib as lib
import sys
import pandas as pd
import numpy as np

def main(ticker="MSFT"):
    print(f"--- Comprehensive Stock Prediction & Smart Analysis for {ticker} ---")
    
    # 1. Fetch data
    print("Fetching historical data and smart features...")
    df = lib.get_stock_data(ticker)
    market_trend, sector_trend = lib.get_market_trends()
    fear_greed_val, fear_greed_desc = lib.get_fear_greed()
    sentiment = lib.get_sentiment_score(ticker)
    
    print(f"Market Trend (SPY): {market_trend:.2%}")
    print(f"Sector Trend (XLK): {sector_trend:.2%}")
    print(f"Fear & Greed: {fear_greed_val:.2f} ({fear_greed_desc})")
    print(f"News Sentiment Score: {sentiment:.2f}")
    
    # 2. Add indicators
    print("Calculating technical indicators (RSI, MACD, SMA, EMA, BB, ATR)...")
    df = lib.add_comprehensive_indicators(df)
    
    # 3. Model Training & Comparison
    print("Training and evaluating models (LR, RF, XGB)...")
    results_df, predictions, y_test = lib.train_and_evaluate_models(df)
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))
    
    # 4. Signal Generation
    print("\nGenerating final signal using all smart features...")
    signal = lib.generate_final_signal(df, sentiment, fear_greed_val, market_trend, sector_trend)
    print(f"FINAL SIGNAL: {signal}")
    
    # 5. Visualization
    print("Generating final comprehensive analysis charts...")
    lib.plot_final_analysis(df, ticker, predictions, y_test, signal, sentiment, fear_greed_val)
    
    # 6. Summary Output
    current_price = df['Close'].iloc[-1]
    print(f"\n--- Current Summary for {ticker} ---")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Final Signal: {signal}")
    print(f"Analysis chart saved to: /home/ubuntu/{ticker}_final_analysis.png")

if __name__ == "__main__":
    ticker = "MSFT"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    main(ticker)

import stock_predictor_lib as lib
import pandas as pd
import sys

def main(ticker="AAPL"):
    print(f"--- Stock Prediction Analysis for {ticker} ---")
    
    # 1. Fetch data
    print("Fetching historical data...")
    df = lib.get_stock_data(ticker)
    
    # 2. Add technical indicators
    print("Calculating technical indicators...")
    df = lib.add_technical_indicators(df)
    
    # 3. Add buy/sell signals
    print("Generating buy/sell signals...")
    df = lib.generate_signals(df)
    
    # 4. Market sentiment (simulated headlines for practice purposes)
    print("Analyzing market sentiment...")
    headlines = [
        f"{ticker} reports strong quarterly growth and record profits.",
        f"Analysts upgrade {ticker} to 'Strong Buy' as market share increases.",
        f"Tech sector sees broad rally, {ticker} leads the way.",
        f"Concerns over global supply chain may impact {ticker} in short term."
    ]
    sentiment = lib.get_sentiment(headlines)
    print(f"Market Sentiment Score: {sentiment:.2f} (0=Bearish, 1=Bullish)")
    
    # 5. Trend predictions
    print("Training models for trend prediction...")
    forecast_periods = [1, 5, 30]
    predictions = {}
    scores = {}
    
    for days in forecast_periods:
        pred, score, y_test, y_pred = lib.train_and_predict(df, days)
        predictions[days] = pred
        scores[days] = score
        # Save visualization for 5-day prediction
        if days == 5:
            lib.plot_accuracy(y_test, y_pred, ticker)
    
    # 6. Visualization
    print("Generating analysis charts...")
    lib.plot_predictions(df, ticker)
    
    # 7. Output results
    print("\n--- Results ---")
    current_price = df['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    for days in forecast_periods:
        trend = "Bullish" if predictions[days] > current_price else "Bearish"
        print(f"Next {days} Day(s) Prediction: ${predictions[days]:.2f} ({trend}, Accuracy: {scores[days]:.2f})")
    
    last_signal = df['Signal'].iloc[-1]
    if last_signal == 1:
        print("Recommendation: BUY (RSI Oversold Reversal + Trend Bullish)")
    elif last_signal == -1:
        print("Recommendation: SELL (RSI Overbought Reversal + Trend Bearish)")
    else:
        print("Recommendation: HOLD")
    
    print(f"\nCharts saved to: /home/ubuntu/{ticker}_analysis.png and /home/ubuntu/{ticker}_accuracy.png")

if __name__ == "__main__":
    ticker = "AAPL"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    main(ticker)

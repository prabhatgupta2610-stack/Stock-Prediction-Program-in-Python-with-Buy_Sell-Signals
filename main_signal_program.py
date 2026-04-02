import stock_signal_lib as lib
import sys

def main(ticker="TSLA"):
    print(f"--- Refined Stock Signal Generation for {ticker} ---")
    
    # 1. Fetch data
    print("Fetching historical data...")
    df = lib.get_stock_data(ticker)
    
    # 2. Calculate signals
    print("Calculating refined signals based on specific conditions...")
    df = lib.calculate_signals(df)
    
    # 3. Visualization
    print("Generating refined signal analysis charts...")
    lib.plot_signals(df, ticker)
    
    # 4. Results
    print("\n--- Current Signal Analysis ---")
    current_data = df.iloc[-1]
    print(f"Ticker: {ticker}")
    print(f"Current Price: ${current_data['Close']:.2f}")
    print(f"RSI: {current_data['RSI']:.2f}")
    print(f"MA 50: ${current_data['MA50']:.2f}")
    print(f"Volume Spike: {'Yes' if current_data['Volume_Spike'] else 'No'}")
    print(f"Near Resistance: {'Yes' if current_data['Near_Resistance'] else 'No'}")
    print(f"Negative Volume Divergence: {'Yes' if current_data['Neg_Vol_Divergence'] else 'No'}")
    
    print(f"\nFINAL SIGNAL: {current_data['Signal']}")
    
    print(f"\nRefined signal analysis chart saved to: /home/ubuntu/{ticker}_refined_signals.png")

if __name__ == "__main__":
    ticker = "TSLA"
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    main(ticker)

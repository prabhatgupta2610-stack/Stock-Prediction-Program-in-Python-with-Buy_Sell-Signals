# Refined Stock Signal Generation Program

This program is a Python-based tool that generates specific BUY, SELL, and HOLD signals for stocks based on a combination of technical indicators, volume analysis, and price action.

## Features

- **Buy Conditions**:
  - RSI below 30 and rising (oversold reversal).
  - MACD crossover bullish (momentum shift).
  - Price above 50-day moving average (trend confirmation).
  - Volume spike confirmation (stronger conviction).

- **Sell Conditions**:
  - RSI above 70 (overbought).
  - MACD bearish crossover (momentum weakening).
  - Price near resistance (within 2% of 20-day high).
  - Negative volume divergence (price rising while volume is falling over 5 days).

- **Display Signal**:
  - Clear output of **BUY**, **HOLD**, or **SELL** for the current trading day.

- **Visualization**:
  - Multi-panel chart showing:
    - Price action with MA 50 and Buy/Sell markers.
    - RSI with overbought (70) and oversold (30) levels.
    - Volume with 20-day moving average and volume spike markers.

## Setup and Installation

To run this program, you need Python 3 and the following libraries:

```bash
sudo pip3 install yfinance pandas numpy matplotlib ta
```

## How to Run

1. Save the provided Python code into two files: `stock_signal_lib.py` and `main_signal_program.py`.
2. Run the `main_signal_program.py` script from your terminal:

```bash
python3 main_signal_program.py [TICKER_SYMBOL]
```

**Example:**

```bash
python3 main_signal_program.py TSLA
```

## Output

The program will print the current signal analysis to the console and save a visualization chart:

- `[TICKER_SYMBOL]_refined_signals.png`: A comprehensive chart showing the indicators and signal markers.

## Program Structure

- `stock_signal_lib.py`: Contains functions for data fetching, technical indicator calculation, refined signal logic, and plotting.
- `main_signal_program.py`: The main script that orchestrates the execution and displays the results.

**Disclaimer**: This program is for educational and practice purposes only and should not be used for real financial trading decisions. Stock market investments involve risks, and past performance is not indicative of future results.

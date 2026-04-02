# Improved Stock Prediction Program

This is an enhanced Python-based stock prediction tool that incorporates advanced feature engineering and compares multiple machine learning models to improve prediction accuracy. It provides comprehensive analysis, model evaluation, and visualization of results.

## Features

- **Advanced Feature Engineering**: Includes:
  - **Lag Features**: Previous 5, 10, and 20-day closing prices.
  - **Rolling Averages**: Standard deviation of closing prices over a 20-day window.
  - **Volatility Calculation**: Annualized volatility based on daily returns.
  - **Momentum Score**: 10-day Rate of Change (ROC).
  - **Daily Return Percentage**.
  - **Trend Strength Score**: Average Directional Index (ADX).
  - **Price Breakout Detection**: Identifies when the price breaks above a 20-day high or below a 20-day low.

- **Multiple Prediction Models**: Compares the performance of:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - Long Short-Term Memory (LSTM) Neural Network

- **Model Evaluation**: Evaluates models using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) Score

- **Visualization**: Generates charts for:
  - Comparison of MAE, RMSE, and R² across all models.
  - Overlay of actual prices vs. predicted prices from each model on the test set.

## Setup and Installation

To run this program, you need Python 3 and the following libraries. You can install them using pip:

```bash
sudo pip3 install yfinance pandas numpy scikit-learn matplotlib ta xgboost tensorflow
```

## How to Run

1. Save the provided Python code into two files: `stock_predictor_lib_v2.py` and `main_predictor_v2.py`.
2. Run the `main_predictor_v2.py` script from your terminal. You can optionally provide a stock ticker symbol as a command-line argument. If no ticker is provided, it defaults to AAPL.

```bash
python3 main_predictor_v2.py [TICKER_SYMBOL]
```

**Example:**

```bash
python3 main_predictor_v2.py NVDA
```

## Output

The program will print model performance metrics and the next day's price prediction to the console. It will also save two charts in the same directory where the script is executed:

- `[TICKER_SYMBOL]_model_comparison.png`: Bar charts comparing MAE, RMSE, and R² for all models.
- `[TICKER_SYMBOL]_predictions_overlay.png`: A line chart showing actual prices and predicted prices from each model on the test set.

## Program Structure

- `stock_predictor_lib_v2.py`: Contains functions for data fetching, advanced feature engineering, training and predicting with different ML models, and plotting utilities.
- `main_predictor_v2.py`: The main script that orchestrates the execution flow, calls functions from `stock_predictor_lib_v2.py`, and prints the results and generates visualizations.

## Model Performance Summary (Example for NVDA)

| Model             | MAE       | RMSE      | R2        |
|:------------------|:----------|:----------|:----------|
| Linear Regression | 2.896504  | 3.729176  | 0.976117  |
| Random Forest     | 31.440558 | 35.602247 | -1.176757 |
| XGBoost           | 34.460047 | 38.912066 | -1.600301 |
| LSTM              | 46.133859 | 50.187306 | -3.325565 |

*Note: The R² score for Random Forest, XGBoost, and LSTM models is negative, indicating that these models perform worse than a simple horizontal line average of the target variable on this specific dataset and split. This could be due to the inherent volatility of stock data, the chosen features, or the model hyperparameters. Linear Regression performed significantly better in this instance.*

## Limitations

- **Data Period**: The `get_stock_data` function now fetches 5 years of data, which is more robust for training complex models.
- **Sentiment Analysis**: The sentiment analysis component from the previous version was removed to focus on numerical prediction and model comparison. It can be re-integrated if needed.
- **Prediction Accuracy**: Stock price prediction remains a challenging task. The models and features used here are for demonstration and practice. Real-world financial decisions should involve more sophisticated analysis and expert advice.
- **Hyperparameter Tuning**: The models use default or basic hyperparameters. Further tuning could potentially improve performance.

**Disclaimer**: This program is for educational and practice purposes only and should not be used for real financial trading decisions. Stock market investments involve risks, and past performance is not indicative of future results.

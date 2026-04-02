# 📈 Stock Price Prediction

A machine learning project that predicts stock prices using historical market data and technical indicators. Built with Python and popular ML libraries like scikit-learn and XGBoost.

---

## 🚀 Features

- Fetches historical stock data (e.g., via yfinance)
- Feature engineering with technical indicators (Moving Average, RSI, MACD, etc.)
- Trains ML models: Linear Regression, Random Forest, XGBoost
- Evaluates model performance (RMSE, MAE, R²)
- Visualizes actual vs predicted prices

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| scikit-learn | ML models & preprocessing |
| XGBoost | Gradient boosting model |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualization |
| yfinance | Stock data fetching |

---

## 📁 Project Structure
```
stock-price-prediction/
│
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA & experiments
├── src/
│   ├── fetch_data.py   # Data collection
│   ├── features.py     # Feature engineering
│   ├── train.py        # Model training
│   └── predict.py      # Prediction logic
├── models/             # Saved trained models
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Usage
```bash
# Fetch stock data
python src/fetch_data.py --ticker AAPL --period 2y

# Train the model
python src/train.py

# Make predictions
python src/predict.py --ticker AAPL
```

---

## 📉 Models Used

- **Linear Regression** — baseline model
- **Random Forest Regressor** — handles non-linearity
- **XGBoost** — best performance on tabular data

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock market predictions are inherently uncertain and should **not** be used as financial advice.

---

## 📌 Roadmap

- [ ] Data collection pipeline
- [ ] Feature engineering module
- [ ] Baseline model training
- [ ] XGBoost model tuning
- [ ] Backtesting module
- [ ] (Optional) Web dashboard

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE)

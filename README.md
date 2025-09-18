# 🏦 StockVision - AI-Powered Stock Analysis Platform

An intelligent stock analysis platform that combines LSTM, XGBoost, and Random Forest models to predict Indian stock prices using technical indicators.

## 🚀 Features

- **AI-Powered Predictions**: Ensemble model combining LSTM, XGBoost, and Random Forest
- **Technical Analysis**: RSI, ADX, ROC, and Stochastic indicators
- **Interactive Charts**: Candlestick charts with Plotly
- **Personal Watchlist**: Track your favorite stocks
- **User Authentication**: Secure login system with SQLite

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, SQLite
- **ML Models**: TensorFlow (LSTM), XGBoost, Random Forest
- **Data**: Yahoo Finance API
- **Visualization**: Matplotlib, Plotly

## 📊 Technical Indicators

- **RSI (Relative Strength Index)**: Momentum oscillator
- **ADX (Average Directional Index)**: Trend strength indicator
- **ROC (Rate of Change)**: Price momentum indicator
- **Stochastic Oscillator**: Momentum indicator

## 🚀 Quick Start

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Train models**: `python train_models.py`
4. **Run the app**: `streamlit run app.py`

## 📈 Supported Stocks

- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- RELIANCE.NS (Reliance Industries)
- HDFCBANK.NS (HDFC Bank)
- WIPRO.NS (Wipro)
- And more Indian stocks...

## 🔧 Model Training

The ensemble model combines:
- **LSTM**: For sequential pattern recognition
- **XGBoost**: For gradient boosting
- **Random Forest**: For ensemble learning

## 📱 Deployment

This app is optimized for Streamlit Cloud deployment with automatic model fallback for demo purposes.

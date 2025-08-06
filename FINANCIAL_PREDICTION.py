import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import streamlit as st
st.set_page_config(page_title="StockVision", page_icon="📈", layout="wide")

import yfinance as yf
import pickle
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import random
import yfinance as yf
import time
from requests.exceptions import RequestException
import re

# Initialize session state for user
if 'user' not in st.session_state:
    st.session_state['user'] = None

# Database setup
def create_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            username TEXT,
            stock TEXT,
            PRIMARY KEY (username, stock),
            FOREIGN KEY (username) REFERENCES users(username)
        )
    """)
    conn.commit()
    conn.close()

create_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        st.success("Account created! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists. Try another.")
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    if user and user[0] == hash_password(password):
        return True
    return False

def add_to_watchlist(username, stock):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO watchlist (username, stock) VALUES (?, ?)", (username, stock))
    conn.commit()
    conn.close()

def get_watchlist(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT stock FROM watchlist WHERE username = ?", (username,))
    stocks = [row[0] for row in c.fetchall()]
    conn.close()
    return stocks

@st.cache_data(show_spinner=False)
def get_cached_stock_history(ticker, period="60d", interval="1d"):
    stock = yf.Ticker(ticker)
    return stock.history(period=period, interval=interval)

@st.cache_resource
def load_models():
    try:
        with open("stock_models.pkl", "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please train models first.")
        return {}
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}

def get_technical_indicators(stock_data):
    df = stock_data.copy()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    df['Stochastic'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                         (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
    
    # Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Average Directional Index (simplified)
    df['ADX'] = abs(df['High'] - df['Low']).rolling(14).mean()
    
    df.fillna(0, inplace=True)
    return df

def predict_next_day(ticker, model_dict):
    # Get latest stock data (cached)
    hist = get_cached_stock_history(ticker, period="60d")

    if hist.empty:
        return None, "No data available for this stock"

    # Calculate technical indicators
    data = get_technical_indicators(hist)

    # Extract features
    features = ['Close', 'RSI', 'Stochastic', 'ROC', 'ADX']
    scaler = model_dict['scaler']
    time_step = model_dict['time_step']

    # Scale the data
    data_scaled = scaler.transform(data[features].tail(time_step))

    # Prepare input for LSTM
    X_lstm = data_scaled.reshape(1, time_step, len(features))

    # Prepare input for XGBoost and RF
    X_tabular = data_scaled.reshape(1, -1)

    # Make predictions
    lstm_pred = model_dict['lstm'].predict(X_lstm)[0][0]
    xgb_pred = model_dict['xgb'].predict(X_tabular)[0]
    rf_pred = model_dict['rf'].predict(X_tabular)[0]

    # Ensemble prediction
    ensemble_pred = (lstm_pred + xgb_pred + rf_pred) / 3

    # Inverse transform to get the actual price
    pred_scaled = np.zeros((1, len(features)))
    pred_scaled[0, 0] = ensemble_pred
    predicted_price = scaler.inverse_transform(pred_scaled)[0, 0]

    # Calculate the date for tomorrow
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    return predicted_price, tomorrow


# def login_page():
#     st.sidebar.header("Login / Sign Up")
#     option = st.sidebar.radio("Select Option", ["Login", "Register"])
#     username = st.sidebar.text_input("Username:")
#     password = st.sidebar.text_input("Password:", type="password")
#     if option == "Register":
#         if st.sidebar.button("Create Account"):
#             register_user(username, password)
#     else:
#         if st.sidebar.button("Login"):
#             if login_user(username, password):
#                 st.session_state['user'] = username
#                 st.sidebar.success(f"Logged in as {username}")
#             else:
#                 st.sidebar.error("Invalid credentials.")

def login_page():
    st.sidebar.header("Login / Register")

    tab1, tab2 = st.sidebar.tabs(["🔐 Login", "🆕 Register"])

    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if not username or not password:
                st.warning("Please fill in both fields.")
            elif login_user(username, password):
                st.session_state['user'] = username
                st.sidebar.success(f"Logged in as {username}")
                st.rerun()

            else:
                st.error("Invalid username or password.")

    with tab2:
        new_username = st.text_input("New Username", key="reg_username")
        new_password = st.text_input("New Password", type="password", key="reg_password")
        if st.button("Register"):
            if not new_username or not new_password:
                st.warning("Please fill in both fields.")
            else:
                register_user(new_username, new_password)


if st.session_state['user'] is None:
    login_page()
    st.stop()
else:
    st.sidebar.write(f"Logged in as: {st.session_state['user']}")
    if st.sidebar.button("Logout"):
        st.session_state['user'] = None
        st.rerun()


# Load all models
all_models = load_models()

# Available stocks (those with trained models)
available_stocks = list(all_models.keys()) if all_models else []
if not available_stocks:
    available_stocks = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]  # Default options

# Modified navigation without Historical Performance
# page = st.sidebar.radio("Navigation", ["Stock Graph & Prediction", "Custom Analysis", "Watchlist"])
page = st.sidebar.radio("🧰 Analysis Tools", ["📊 AI-Powered Prediction", "📈 Technical Insights", "⭐ My Watchlist", "🧭 Interactive Charting"])

if page == "📊 AI-Powered Prediction":
    st.title("AI/ML Finance Market Predictor")
    
    selected_stock = st.selectbox("Choose a stock:", available_stocks)

    if selected_stock:
        # Display stock info
        stock = yf.Ticker(selected_stock)
        try:
            info = stock.info
            
            # Create two columns for layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"{info.get('shortName', selected_stock)}")
                
                # Get historical data
                hist = stock.history(period="1y")
                
                if not hist.empty:
                    # Plot the stock price
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(hist.index, hist['Close'], label='Close Price')
                    
                    ax.set_title(f"{selected_stock} Stock Price (Last Year)")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (INR)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display recent price information
                    st.write("### Recent Price Information")
                    last_price = hist['Close'].iloc[-1]
                    price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                    price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                    
                    col_price1, col_price2, col_price3 = st.columns(3)
                    col_price1.metric("Last Close Price", f"₹{last_price:.2f}")
                    col_price2.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
                    col_price3.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
                    
                    # Make prediction if model exists
                    if selected_stock in all_models:
                        predicted_price, prediction_date = predict_next_day(selected_stock, all_models[selected_stock])
                        if predicted_price is not None:
                            # Calculate prediction change
                            pred_change = predicted_price - last_price
                            pred_change_pct = (pred_change / last_price) * 100
                            
                            st.subheader(f"Prediction for {prediction_date}")
                            st.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{pred_change_pct:.2f}%")
                    else:
                        st.warning(f"No trained model available for {selected_stock}")
                else:
                    st.error("No historical data available for this stock.")
            
            with col2:
                # Display company info
                st.subheader("Company Info")
                if 'sector' in info:
                    st.write(f"**Sector:** {info['sector']}")
                if 'industry' in info:
                    st.write(f"**Industry:** {info['industry']}")
                if 'marketCap' in info:
                    st.write(f"**Market Cap:** ₹{info['marketCap']:,.0f}")
                if 'trailingPE' in info:
                    st.write(f"**P/E Ratio:** {info['trailingPE']:.2f}")
                if 'dividendYield' in info and info['dividendYield'] is not None:
                    st.write(f"**Dividend Yield:** {info['dividendYield']*100:.2f}%")
                
                # Add to watchlist button
                if st.button("Add to Watchlist"):
                    add_to_watchlist(st.session_state['user'], selected_stock)
                    st.success(f"{selected_stock} added to watchlist!")
                
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")

elif page == "📈 Technical Insights":
    st.title("Custom Technical Analysis")
    
    selected_stock = st.selectbox("Choose a stock:", available_stocks)
    
    if selected_stock:
        st.subheader(f"Technical Indicators for {selected_stock}")
        
        # Get latest stock data
        stock = yf.Ticker(selected_stock)
        hist = stock.history(period="60d")
        
        if not hist.empty:
            # Calculate technical indicators
            data = get_technical_indicators(hist)
            
            # Display the technical indicators
            st.dataframe(data[['Close', 'RSI', 'Stochastic', 'ROC', 'ADX']].tail())
            
            # Technical indicator plots
            tab1, tab2, tab3 = st.tabs(["Price & Volume", "Momentum Indicators", "Volatility"])
            
            with tab1:
                fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
                ax[0].plot(data.index, data['Close'], label='Close Price')
                ax[0].set_title(f"{selected_stock} Price")
                ax[0].grid(True, alpha=0.3)
                ax[0].legend()
                
                ax[1].bar(data.index, data['Volume'], label='Volume')
                ax[1].set_title("Volume")
                ax[1].grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # RSI Plot
                ax1.plot(data.index, data['RSI'], color='purple')
                ax1.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                ax1.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                ax1.fill_between(data.index, data['RSI'], 70, where=(data['RSI'] >= 70), color='r', alpha=0.3)
                ax1.fill_between(data.index, data['RSI'], 30, where=(data['RSI'] <= 30), color='g', alpha=0.3)
                ax1.set_title("Relative Strength Index (RSI)")
                ax1.grid(True, alpha=0.3)
                
                # Stochastic Plot
                ax2.plot(data.index, data['Stochastic'], color='blue')
                ax2.axhline(y=80, color='r', linestyle='-', alpha=0.3)
                ax2.axhline(y=20, color='g', linestyle='-', alpha=0.3)
                ax2.fill_between(data.index, data['Stochastic'], 80, where=(data['Stochastic'] >= 80), color='r', alpha=0.3)
                ax2.fill_between(data.index, data['Stochastic'], 20, where=(data['Stochastic'] <= 20), color='g', alpha=0.3)
                ax2.set_title("Stochastic Oscillator")
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab3:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(data.index, data['ROC'], label='Rate of Change', color='orange')
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.set_title("Rate of Change (ROC)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                st.pyplot(fig)
        else:
            st.error("No data available for this stock.")

elif page == "⭐ My Watchlist":
    st.title("Your Watchlist")
    
    watchlist = get_watchlist(st.session_state['user'])
    
    if not watchlist:
        st.info("Your watchlist is empty. Add stocks from the Stock Graph & Prediction page!")
    else:
        # Display watchlist stocks in a grid
        cols = st.columns(3)
        
        for i, stock in enumerate(watchlist):
            with cols[i % 3]:
                stock_ticker = yf.Ticker(stock)
                hist = stock_ticker.history(period="5d")
                
                if not hist.empty:
                    last_price = hist['Close'].iloc[-1]
                    price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
                    price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                    
                    st.subheader(stock)
                    st.metric("Price", f"₹{last_price:.2f}", f"{price_change_pct:.2f}%")
                    
                    # Calculate prediction if model exists
                    if stock in all_models:
                        predicted_price, prediction_date = predict_next_day(stock, all_models[stock])
                        if predicted_price is not None:
                            pred_change_pct = ((predicted_price - last_price) / last_price) * 100
                            direction = "↑" if pred_change_pct > 0 else "↓"
                            st.write(f"Prediction: {direction} ₹{predicted_price:.2f} ({pred_change_pct:.2f}%)")
                    
                    # Mini chart
                    st.line_chart(hist['Close'])
                else:
                    st.subheader(stock)
                    st.warning("No data available")

elif page == "🧭 Interactive Charting":
    st.title("Interactive Stock Chart")
    selected_stock = st.selectbox("Choose a stock to visualize:", available_stocks)

    if selected_stock:
        stock = yf.Ticker(selected_stock)
        data = stock.history(period="6mo", interval="1d")
        if not data.empty:
            import plotly.graph_objs as go

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            ))
            fig.update_layout(
                title=f'{selected_stock} - Interactive Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price (INR)',
                xaxis_rangeslider_visible=True,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for this stock.")


# Add your watchlist to the sidebar
st.sidebar.write("### 🔖 Quick Watchlist")
for stock in get_watchlist(st.session_state['user']):
    st.sidebar.write(f"- {stock}")



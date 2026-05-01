import streamlit as st  
import yfinance as yf 
import pandas as pd  
import joblib  
import datetime
import shap 
import matplotlib.pyplot as plt  
from indicators import compute_rsi, compute_macd, compute_sma
from sentiments import get_sentiment_score
from dotenv import load_dotenv  
import os
import numpy as np

load_dotenv()
api_key = os.getenv("api_key")

model = joblib.load('xgb_model.pkl')

# UI
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Pattern Predictor")
st.markdown("Predict **Buy / Sell / Hold** using technical indicators + news sentiment")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TCS.BO, INFY.BO)", "AAPL")
st.write(f"Selected Ticker: `{ticker}`")

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan


if st.button("Predict"):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=100)

    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        st.error("Couldn't fetch stock data. Check ticker symbol.")
    else:
        with st.spinner("Analyzing stock data..."):

            df['rsi'] = compute_rsi(df)
            df['macd'] = compute_macd(df)
            df['sma'] = compute_sma(df)

            sent = get_sentiment_score(ticker)
            try:
                sent = float(sent)
            except:
                sent = 0.0
            df['sentiment'] = sent

            latest = df.iloc[-1]

            input_features = pd.DataFrame([[ 
                safe_float(latest['rsi']),
                safe_float(latest['macd']),
                safe_float(latest['sma']),
                safe_float(latest['sentiment'])
            ]], columns=['rsi', 'macd', 'sma', 'sentiment'])

            input_features = input_features.apply(pd.to_numeric, errors='coerce')

            if input_features.isnull().values.any():
                st.warning("⚠️ Some indicators missing. Using safe fallback values.")
                input_features = input_features.fillna(input_features.mean()).fillna(0)

            # Prediction
            prediction = model.predict(input_features)[0]
            proba = model.predict_proba(input_features)[0]
            label_map = {0: 'Buy 🟢', 1: 'Sell 🔴', 2: 'Hold ⚪'}

            st.subheader("📊 Prediction")
            st.success(label_map.get(prediction, "Unknown"))
            st.write("**Input Features**")
            st.dataframe(input_features)
            st.write("**Prediction Probabilities (Buy, Sell, Hold)**", proba)

            # SHAP explainability
            st.subheader("🔍 Model Explainability (SHAP)")
            explainer = shap.Explainer(model)
            shap_values = explainer(input_features)

            shap_exp = shap.Explanation(
                values=shap_values.values[0][prediction],
                base_values=shap_values.base_values[0][prediction],
                data=input_features.values[0],
                feature_names=input_features.columns.tolist()
            )

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig)

            # Charts
            st.subheader("📉 Stock Price Trend")
            st.line_chart(df['Close'])

            st.subheader("📋 Recent Data")
            st.dataframe(df.tail())

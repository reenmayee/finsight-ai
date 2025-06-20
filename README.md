# FinSight-AI - AI Stock Pattern Predictor

A machine learning-powered web app that predicts whether to **Buy**, **Sell**, or **Hold** a stock using technical indicators and real-time news sentiment analysis.

## 🚀 Live App

[View Deployment](https://your-deployment-link)  
_Replace this link with your actual Streamlit Cloud or HuggingFace Space URL._

---

## 🔍 Features

- Real-time stock data via Yahoo Finance
- Technical indicators: RSI, MACD, SMA
- News sentiment analysis using NewsAPI + VADER
- XGBoost model for prediction
- SHAP explainability plots
- Interactive Streamlit interface

---

## 🛠️ How to Run Locally

```bash
git clone https://github.com/yourusername/ai-stock-predictor.git
cd ai-stock-predictor
pip install -r requirements.txt

Create a .env file in the root directory :
NEWS_API_KEY=your_news_api_key_here

Then run the app :
streamlit run app.py

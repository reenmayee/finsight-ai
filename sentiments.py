# News sentiment score function
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import requests  # type: ignore
from dotenv import load_dotenv # type: ignore
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("NEWS_API_KEY")

# Sentiment analyzer setup
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(ticker_symbol):
    url = f"https://newsapi.org/v2/everything?q={ticker_symbol}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        return 0
    articles = response.json().get('articles', [])
    sentiments = []
    for article in articles:
        text = article.get('title', '') + " " + article.get('description', '')
        score = analyzer.polarity_scores(text)['compound']
        sentiments.append(score)
    return sum(sentiments) / len(sentiments) if sentiments else 0

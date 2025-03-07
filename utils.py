import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import ta
import streamlit as st
import time
from textblob import TextBlob
import requests

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st


@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, timeframe: str) -> pd.DataFrame:
    timeframe_dict = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
    period = timeframe_dict[timeframe]
    stock = yf.Ticker(symbol)
    try:
        data = stock.history(period=period, interval="1d")
    except Exception as e:
        if "Rate" in str(e):
            st.error("Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            data = stock.history(period=period, interval="1d")
        else:
            raise e
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
    data = add_technical_indicators(data)
    return data

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    """
    # MACD
    data['macd'] = ta.trend.macd_diff(data['Close'])

    # RSI
    data['rsi'] = ta.momentum.rsi(data['Close'])

    # Moving averages
    data['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50)

    # Bollinger Bands
    data['bb_high'] = ta.volatility.bollinger_hband(data['Close'])
    data['bb_low'] = ta.volatility.bollinger_lband(data['Close'])

    # Momentum
    data['momentum'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])

     # VWAP (Volume Weighted Average Price)
    data['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

    # Fibonacci Retracement Levels
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    diff = max_price - min_price
    data['fib_0'] = min_price
    data['fib_0.236'] = min_price + 0.236 * diff
    data['fib_0.382'] = min_price + 0.382 * diff
    data['fib_0.5'] = min_price + 0.5 * diff
    data['fib_0.618'] = min_price + 0.618 * diff
    data['fib_0.786'] = min_price + 0.786 * diff
    data['fib_1'] = max_price
    
    # Average True Range (ATR) - Volatility indicator
    data['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    
    # On-Balance Volume (OBV) - Volume indicator
    data['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
    
    # Ichimoku Cloud
    data['ichimoku_a'] = ta.trend.ichimoku_a(data['High'], data['Low'])
    data['ichimoku_b'] = ta.trend.ichimoku_b(data['High'], data['Low'])

    return data

def predict_next_day_price(data: pd.DataFrame) -> tuple:
    """
    Predict the next day's closing price using technical indicators
    Returns: (predicted_price, confidence_score)
    """
    if len(data) < 20:  # Need at least 20 days for technical indicators
        return None, None

    # Prepare features for prediction
    features = ['macd', 'rsi', 'sma_20', 'sma_50', 'momentum', 'vwap']

    # Remove any rows with NaN values
    analysis_data = data.dropna().copy()
    if len(analysis_data) < 5:
        return None, None

    # Prepare X (features) and y (target)
    X = analysis_data[features].values[-20:]  # Use last 20 days
    y = analysis_data['Close'].values[-20:]

    # Reshape X to 2D array if necessary
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Prepare latest features for prediction
    latest_features = analysis_data[features].values[-1:] 

    # Predict next day
    predicted_price = model.predict(latest_features)[0]

    # Calculate confidence score (R² score)
    confidence = model.score(X, y)

    return round(predicted_price, 2), round(confidence * 100, 2)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_info(symbol: str) -> dict:
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        'name': info.get('longName', symbol),
        'marketCap': info.get('marketCap', 0),
        'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
        'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
        'trailingPE': info.get('trailingPE', None),
        'forwardPE': info.get('forwardPE', None),
        'pegRatio': info.get('pegRatio', None),
        'priceToBook': info.get('priceToBook', None),
        'trailingEps': info.get('trailingEps', None),
        'dividendYield': info.get('dividendYield', None),
        'beta': info.get('beta', None),
        'volume': info.get('volume', 0)
    }

def format_large_number(number: float) -> str:
    """
    Format large numbers to human-readable format
    """
    if number is None or number == 0:
        return "N/A"

    billion = 1_000_000_000
    million = 1_000_000

    if number >= billion:
        return f"${number / billion:.2f}B"
    elif number >= million:
        return f"${number / million:.2f}M"
    else:
        return f"${number:,.0f}"
    

def get_stock_news(symbol: str, days_back: int = 7, max_items: int = 5) -> list:
    API_KEY = '16f43921b0d8438c84c39b1aab364c61'
    query = symbol
    language = 'en'
    page_size = max_items
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q={query}&language={language}&from={from_date}&pageSize={page_size}&apiKey={API_KEY}'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            articles = data['articles']
            news_items = [
                {
                    'title': article['title'],
                    'publisher': article['source']['name'],
                    'link': article['url'],
                    'providerPublishTime': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
                }
                for article in articles
            ]
            return news_items
        else:
            st.warning(f"Failed to fetch news for {symbol}: {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Error fetching news for {symbol}: {e}")
        return []
    


def get_news_sentiment(news: list) -> float:
    """
    Compute the average sentiment score from a list of news items.
    
    Args:
        news (list): List of news items from get_stock_news
    
    Returns:
        float: Average sentiment score (-1 to 1), 0 if no valid news
    """
    sentiments = []
    for item in news:
        title = item.get('title', '')
        if title:
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0.0


# Function to fetch article content (simplified for demo)
def fetch_article_content(url: str, title: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = ' '.join(p.get_text() for p in paragraphs)[:1000]  # Limit to 1000 chars
            return content
        else:
            print(f"Failed to fetch {url}, using title only")
            return title
    except Exception as e:
        print(f"Error fetching {url}: {e}, using title only")
        return title


@st.cache_resource  # Cache the model loading for performance
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def get_advanced_news_sentiment(news: list, days_back: int = 7) -> float:
    tokenizer, model = load_finbert_model()
    sentiments = []
    current_time = datetime.now().timestamp()

    for item in news:
        url = item.get('link', '')
        publish_time = item.get('providerPublishTime', current_time)
        title = item.get('title', '')

        # Fetch article content
        content = fetch_article_content(url,title) if url else title
        if not content:
            content = title  # Fallback to title if content fetch fails

        # Tokenize and analyze sentiment
        inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]  # [Negative, Positive, Neutral]
        
        # Map to a sentiment score: Positive (+1), Neutral (0), Negative (-1)
        sentiment_score = probs[1] - probs[0]  # Positive prob - Negative prob (range: -1 to 1)

        # Temporal weighting: Linear decay over days_back
        age_days = (current_time - publish_time) / (24 * 3600)  # Age in days
        if age_days > days_back:
            weight = 0  # Ignore articles older than days_back
        else:
            weight = 1 - (age_days / days_back)  # Linear decay: 1 (now) to 0 (days_back ago)
        
        sentiments.append(sentiment_score * weight)

    # Average weighted sentiments
    return sum(sentiments) / len(sentiments) if sentiments else 0.0


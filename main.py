import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from utils import fetch_stock_data, get_company_info, format_large_number, predict_next_day_price,get_news_sentiment,get_stock_news
from styles import apply_styles
import time
from StockPredictionModel import predict_with_models
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Stock Data Analyzer",
    page_icon="📈",
    layout="wide"
)

# Apply custom styles
apply_styles()

# Initialize session state for auto refresh
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5

# Header
st.title("📈 Stock Data Analyzer")
st.markdown("Enter a stock symbol to view financial data, technical indicators, and AI predictions")

# Input section
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1,1])
with col1:
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", "").upper()
with col2:
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1M", "3M", "6M", "1Y", "2Y", "5Y"],
        index=2
    )
with col3:
    st.session_state.auto_refresh = st.checkbox("Auto Refresh", value=False)  # Default off
with col4:
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.number_input(
            "Refresh Interval (seconds)",
            min_value=60,  # Increase minimum to 60
            max_value=3600,
            value=60
        )

with col5:
        horizon_options = {"Next Day": 1, "Next Week": 5, "Next Month": 20}
        selected_horizons = st.multiselect("Select Prediction Horizons", list(horizon_options.keys()), default=["Next Day"])
        horizons = [horizon_options[h] for h in selected_horizons]

# Display last updated time
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if symbol:
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(symbol, timeframe)
        company_info = get_company_info(symbol)

        # Calculate volatility based on daily returns
        daily_returns = stock_data['Close'].pct_change().dropna()
        volatility = daily_returns.std()

        # Fetch news and compute sentiment
        news = get_stock_news(symbol)
        sentiment_score = get_news_sentiment(news)

        # Display company info
        st.header(f"{company_info['name']} ({symbol})")

        # Current stock price and stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Current Price",
                f"${stock_data['Close'].iloc[-1]:.2f}",
                f"{((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2] * 100):.2f}%"
            )
        with col2:
            st.metric("Market Cap", format_large_number(company_info['marketCap']))
        with col3:
            st.metric("52W High", f"${company_info['fiftyTwoWeekHigh']:.2f}")
        with col4:
            st.metric("52W Low", f"${company_info['fiftyTwoWeekLow']:.2f}")

        # Technical Indicators
        st.subheader("Technical Indicators")
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        with tech_col1:
            st.metric("RSI", f"{stock_data['rsi'].iloc[-1]:.2f}")
        with tech_col2:
            st.metric("MACD", f"{stock_data['macd'].iloc[-1]:.2f}")
        with tech_col3:
            st.metric(
                "Momentum", 
                f"{stock_data['momentum'].iloc[-1]:.2f}"
            )

        # Price Prediction
        # pred_price, confidence = predict_next_day_price(stock_data)
        # pred_price, en_pred_price,confidence = predict_with_models(stock_data)
        # if pred_price and confidence:
        #     st.subheader("AI Price Prediction")
        #     pred_col1,pred_col2, pred_col3 = st.columns(3)
        #     with pred_col1:
        #         st.metric(
        #             "Predicted Next Day Price",
        #             f"${pred_price:.2f}",
        #             f"{((pred_price - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1] * 100):.2f}%"
        #         )
        #     with pred_col2:
        #         st.metric(
        #              "Ensemble Predicted Next Day Price",
        #             f"${en_pred_price:.2f}",
        #             f"{((en_pred_price - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1] * 100):.2f}%"
        #         )
        #     with pred_col3:
        #         st.metric("Prediction Confidence", f"{confidence}%")

        #     st.info("⚠️ This prediction is based on technical analysis and historical data. It should not be used as financial advice.")

        # Price Prediction
        predictions, ensemble_predictions, confidences = predict_with_models(stock_data, timeframe, horizons)
        if ensemble_predictions:
            # Adjust predictions with news sentiment
            for horizon in horizons:
                adjustment = sentiment_score * volatility / np.sqrt(horizon)
                ensemble_predictions[horizon] *= (1 + adjustment)

            st.subheader("AI Price Predictions")
            cols = st.columns(len(horizons))
            horizon_names = {1: "Next Day", 5: "Next Week", 20: "Next Month"}
            for i, horizon in enumerate(horizons):
                horizon_name = horizon_names.get(horizon, f"{horizon} Days")
                with cols[i]:
                    ensemble_price = ensemble_predictions[horizon]
                    current_price = stock_data['Close'].iloc[-1]
                    st.metric(
                        f"Predicted {horizon_name} Price",
                        f"${ensemble_price:.2f}",
                        f"{((ensemble_price - current_price) / current_price * 100):.2f}%"
                    )
                    st.write(f"Confidence: {confidences[horizon] * 100:.2f}%")
            
            st.info("⚠️ Predictions are based on historical data, technical analysis, and recent news sentiment. Not financial advice.")
            
            # Display news sentiment
            sentiment_label = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"
            st.write(f"Current News Sentiment: {sentiment_label} ({sentiment_score:.2f} on scale -1 to 1)")
            st.metric(f"Predicted {horizon_name} Price", f"${ensemble_price:.2f}", f"{((ensemble_price - current_price) / current_price * 100):.2f}%")
            st.write("This sentiment adjusts the price predictions based on recent news.")
            
            if st.button("Show Recent News"):
                if news:
                    for item in news:
                        st.write(f"- {item['title']} (Source: {item['publisher']})")
                else:
                    st.write("No recent news available.")

        # Stock price chart with technical indicators
        st.subheader("Stock Price Chart with Technical Indicators")
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name="OHLC"
            )
        )

        # Add Moving Averages
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['sma_20'],
                name="20-day SMA",
                line=dict(color='orange')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['sma_50'],
                name="50-day SMA",
                line=dict(color='blue')
            )
        )

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['bb_high'],
                name="Bollinger High",
                line=dict(color='gray', dash='dash')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['bb_low'],
                name="Bollinger Low",
                line=dict(color='gray', dash='dash'),
                fill='tonexty'
            )
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Financial metrics table
        st.subheader("Key Financial Metrics")
        metrics_df = pd.DataFrame({
            'Metric': [
                'P/E Ratio', 'Forward P/E', 'PEG Ratio', 'Price/Book',
                'EPS (TTM)', 'Dividend Yield', 'Beta', 'Volume'
            ],
            'Value': [
                company_info.get('trailingPE', 'N/A'),
                company_info.get('forwardPE', 'N/A'),
                company_info.get('pegRatio', 'N/A'),
                company_info.get('priceToBook', 'N/A'),
                company_info.get('trailingEps', 'N/A'),
                f"{company_info.get('dividendYield', 0) * 100:.2f}%" if company_info.get('dividendYield') else 'N/A',
                company_info.get('beta', 'N/A'),
                format_large_number(company_info.get('volume', 0))
            ]
        })
        st.dataframe(metrics_df)

        # Download button for CSV
        csv = stock_data.to_csv()
        st.download_button(
            label="Download Stock Data as CSV",
            data=csv,
            file_name=f"{symbol}_stock_data.csv",
            mime="text/csv"
        )

        # Auto refresh
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()

    except Exception as e:
        st.error(f"Error: Unable to fetch data for {symbol}. Please check the symbol and try again.")
        st.exception(e)

else:
    st.info("👆 Enter a stock symbol above to get started!")
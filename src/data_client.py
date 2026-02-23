import os
import pandas as pd
from datetime import datetime, timedelta
from eodhd import APIClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class MarketDataClient:
    def __init__(self):
        self.api_key = os.getenv("EODHD_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ EODHD_API_KEY not found in .env file.")
        
        # Initialize the EODHD client
        self.client = APIClient(self.api_key)

    def get_technicals(self, ticker: str) -> dict:
        """Fetches 1 year of historical data and calculates TA metrics."""
        try:
            # EODHD expects tickers like AAPL.US for US stocks
            symbol = f"{ticker}.US" if "." not in ticker else ticker
            
            # Fetch historical data
            df = self.client.get_historical_data(symbol, period='d')
            if df is None or df.empty:
                return None

            # Calculate Moving Averages
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()

            # Calculate RSI (Wilder's Smoothing)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Get latest values
            latest = df.iloc[-1]
            return {
                "Ticker": ticker,
                "Price": round(latest['close'], 2),
                "SMA_50": round(latest['SMA_50'], 2) if not pd.isna(latest['SMA_50']) else 0,
                "SMA_200": round(latest['SMA_200'], 2) if not pd.isna(latest['SMA_200']) else 0,
                "RSI": round(latest['RSI'], 1) if not pd.isna(latest['RSI']) else 50,
                "Volume": int(latest['volume'])
            }
        except Exception as e:
            print(f"Error fetching technicals for {ticker}: {e}")
            return None

    def get_news_sentiment(self, ticker: str) -> dict:
        """Fetches the latest news and EODHD's built-in AI sentiment score."""
        try:
            symbol = f"{ticker}.US" if "." not in ticker else ticker
            # Fetch news from the last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            news = self.client.get_financial_news(s=symbol, from_date=from_date, limit=5)
            
            if not news:
                return {"headlines": "No recent news.", "avg_sentiment": 0.0}

            headlines = []
            total_sentiment = 0.0
            valid_scores = 0

            for article in news:
                title = article.get('title', 'No Title')
                sentiment = article.get('sentiment', {})
                polarity = sentiment.get('polarity', 0) # Score between -1 (Bad) and 1 (Good)
                
                headlines.append(f"- {title} (Score: {polarity})")
                total_sentiment += polarity
                valid_scores += 1

            avg_sentiment = round(total_sentiment / valid_scores, 2) if valid_scores > 0 else 0.0

            return {
                "headlines": "\n".join(headlines),
                "avg_sentiment": avg_sentiment
            }
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return {"headlines": f"Error: {e}", "avg_sentiment": 0.0}

    def get_earnings_date(self, ticker: str) -> str:
        """Fetches highly accurate upcoming earnings dates."""
        try:
            symbol = f"{ticker}.US" if "." not in ticker else ticker
            from_date = datetime.now().strftime("%Y-%m-%d")
            to_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
            
            earnings = self.client.get_upcoming_earnings(from_date=from_date, to_date=to_date, symbols=symbol)
            
            if not earnings or len(earnings) == 0:
                return "Safe (No earnings in next 90 days)"
            
            next_date_str = earnings[0].get('report_date')
            next_date = datetime.strptime(next_date_str, "%Y-%m-%d").date()
            days_until = (next_date - datetime.now().date()).days
            
            if days_until <= 7:
                return f"⚠️ EARNINGS IN {days_until} DAYS ({next_date_str})"
            return f"Safe (Earnings in {days_until} days)"
            
        except Exception as e:
            return "Safe (Data Error)"

# --- Quick Test Block ---
if __name__ == "__main__":
    client = MarketDataClient()
    print("Testing EODHD Data Client...")
    
    test_ticker = "AAPL"
    
    print(f"\n--- Technicals for {test_ticker} ---")
    print(client.get_technicals(test_ticker))
    
    print(f"\n--- Earnings for {test_ticker} ---")
    print(client.get_earnings_date(test_ticker))
    
    print(f"\n--- News & Sentiment for {test_ticker} ---")
    news_data = client.get_news_sentiment(test_ticker)
    print(f"Average Sentiment: {news_data['avg_sentiment']}")
    print(news_data['headlines'])
import os
import requests
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MarketDataClient:
    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ TIINGO_API_KEY not found in .env file.")
        
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }

    def get_technicals(self, ticker: str) -> dict:
        """Fetches 1 year of daily prices from Tiingo and calculates TA metrics."""
        try:
            # Get data for the last 1 year (roughly 252 trading days + buffer)
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Tiingo API Error ({ticker}): {response.text}")
                return None
                
            data = response.json()
            if not data:
                return None

            # Convert to Pandas DataFrame for easy math
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

            # Calculate Moving Averages based on the 'close' price
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()

            # Calculate RSI (Wilder's Smoothing)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Extract the most recent day's data
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
        
    def get_smart_momentum(self, ticker):
        """
        Fetches historical data and calculates Volatility-Adjusted 12-minus-1 Momentum.
        """
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import requests

        try:
            # We look back 380 calendar days to guarantee we get at least 252 trading days
            end_date = datetime.today()
            start_date = end_date - timedelta(days=380)
            
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {
                'startDate': start_date.strftime('%Y-%m-%d'),
                'endDate': end_date.strftime('%Y-%m-%d'),
                'token': self.api_key # Ensure this matches your Tiingo initialization!
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # If a stock IPO'd recently, it won't have enough data. We skip it.
            if len(data) < 252:
                return None 
                
            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            
            # 1. Volatility Calculation
            daily_returns = df['close'].pct_change()
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            # 2. 12-Minus-1 Momentum Calculation (Using .iloc to grab exact days from the end)
            price_t_21 = df['close'].iloc[-21]
            price_t_252 = df['close'].iloc[-252]
            momentum_12m_1m = (price_t_21 - price_t_252) / price_t_252
            
            # 3. The Smoothness Score
            if annual_volatility == 0:
                smooth_score = 0
            else:
                smooth_score = momentum_12m_1m / annual_volatility
                
            return {
                'Current_Price': df['close'].iloc[-1],
                'Momentum_12m_1m': round(momentum_12m_1m, 4),
                'Annual_Volatility': round(annual_volatility, 4),
                'Smooth_Score': round(smooth_score, 4)
            }
            
        except Exception as e:
            # If Tiingo fails or ticker is delisted, silently skip it
            return None

    def get_news(self, ticker: str) -> str:
        """Fetches the latest 5 news headlines from Tiingo."""
        try:
            url = f"https://api.tiingo.com/tiingo/news?tickers={ticker}&limit=5"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                return "News API Error."

            articles = response.json()
            if not articles:
                return "No recent news."

            headlines = []
            for article in articles:
                title = article.get('title', 'No Title')
                source = article.get('source', 'Unknown')
                pub_date = article.get('publishedDate', '')[:10] # Just grab the YYYY-MM-DD
                headlines.append(f"- [{pub_date}] {title} ({source})")

            return "\n".join(headlines)
        except Exception as e:
            return f"Error fetching news: {e}"

    def get_earnings_date(self, ticker: str) -> str:
        """Fallback to yfinance to grab the upcoming earnings date."""
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar
            
            if cal is None or 'Earnings Date' not in cal or not cal['Earnings Date']:
                return "Safe (No Data)"

            reported_date = pd.to_datetime(cal['Earnings Date'][0]).date()
            today = datetime.now().date()

            if reported_date >= today:
                days_until = (reported_date - today).days
                if days_until <= 7:
                    return f"⚠️ EARNINGS IN {days_until} DAYS"
                return f"Safe (Earnings in {days_until} days)"
            else:
                # If the date is in the past, Yahoo hasn't updated it yet
                return "Safe (Awaiting next quarter date)"

        except Exception:
            return "Safe (Data Error)"

# --- Quick Test Block ---
if __name__ == "__main__":
    client = MarketDataClient()
    print("Testing Tiingo Data Client...")
    
    test_ticker = "AAPL"
    
    print(f"\n--- Technicals for {test_ticker} ---")
    print(client.get_technicals(test_ticker))
    
    print(f"\n--- Earnings for {test_ticker} ---")
    print(client.get_earnings_date(test_ticker))
    
    print(f"\n--- News for {test_ticker} ---")
    print(client.get_news(test_ticker))
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
    # 1. Sector Map defined as a class-level constant
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'TXN', 'INTC', 'MU', 'LRCX', 'ADI', 'AMAT', 'KLAC'],
        'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'BKNG', 'TJX', 'ORLY', 'MAR', 'F', 'GM', 'DG', 'EBAY'],
        'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C', 'CB', 'PGR', 'SCHW'],
        'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'HAL', 'DVN', 'HES', 'OXY'],
        'XLV': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN', 'ISRG', 'BMY', 'MRNA', 'GILD'],
        'XLI': ['GE', 'CAT', 'UNP', 'HON', 'RTX', 'LMT', 'DE', 'BA', 'UPS', 'MMM', 'HII', 'PCAR', 'FEDEX', 'GEV'],
        'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'EL', 'MO', 'CL', 'KMB', 'STZ', 'SYY'],
        'XLB': ['LIN', 'SHW', 'APD', 'FCX', 'NEM', 'ECL', 'ALB', 'DOW', 'CTVA'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'TMUS', 'VZ', 'T', 'CHTR', 'WBD'],
        'XLU': ['NEE', 'SO', 'DUK', 'AEP', 'D', 'EXC', 'PCG', 'SRE', 'ED'],
        'XLRE': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SBAC', 'WELL']
    }

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
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate={start_date}"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                return None
                
            data = response.json()
            if not data: return None

            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            latest = df.iloc[-1]
            return {
                "Ticker": ticker,
                "Price": round(latest['close'], 2),
                "SMA_50": round(latest['SMA_50'], 2) if not pd.isna(latest['SMA_50']) else 0,
                "SMA_200": round(latest['SMA_200'], 2) if not pd.isna(latest['SMA_200']) else 0,
                "RSI": round(latest['RSI'], 1) if not pd.isna(latest['RSI']) else 50,
                "Volume": int(latest['volume'])
            }
        except Exception:
            return None

    def get_regime(self, ticker='SPY'):
        """Fetches trend regime for a specific ticker (SPY or Sector ETF)."""
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=380)
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': start_date.strftime('%Y-%m-%d'), 'token': self.api_key}
            
            res = requests.get(url, params=params)
            data = res.json()
            if not data: return None

            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            
            current_price = df['close'].iloc[-1]
            sma_200 = df['close'].rolling(window=200).mean().iloc[-1]
            status = "RISK ON" if current_price > sma_200 else "RISK OFF"
            
            return {
                'status': status,
                'price': round(current_price, 2),
                'sma': round(sma_200, 2),
                'bullish': current_price > sma_200
            }
        except Exception:
            return None

    def get_sector_for_ticker(self, ticker):
        """Finds which XL-ETF a ticker belongs to."""
        for sector_etf, members in self.SECTOR_MAP.items():
            if ticker in members:
                return sector_etf
        return 'SPY' 

    def get_smart_momentum(self, ticker):
        """Calculates Volatility-Adjusted 12-minus-1 Momentum."""
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=380)
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': start_date.strftime('%Y-%m-%d'), 'token': self.api_key}
            
            response = requests.get(url, params=params)
            data = response.json()
            if len(data) < 252: return None 
                
            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            
            daily_returns = df['close'].pct_change()
            annual_volatility = daily_returns.std() * np.sqrt(252)
            
            price_t_21 = df['close'].iloc[-21]
            price_t_252 = df['close'].iloc[-252]
            momentum_12m_1m = (price_t_21 - price_t_252) / price_t_252
            
            smooth_score = momentum_12m_1m / annual_volatility if annual_volatility != 0 else 0
                
            return {
                'Current_Price': df['close'].iloc[-1],
                'Momentum_12m_1m': round(momentum_12m_1m, 4),
                'Annual_Volatility': round(annual_volatility, 4),
                'Smooth_Score': round(smooth_score, 4)
            }
        except Exception:
            return None

    def get_atr_and_sizing(self, ticker, account_value, risk_pct=0.01):
        """Calculates 14-day ATR and recommended position size."""
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=40)
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': start_date.strftime('%Y-%m-%d'), 'token': self.api_key}
            
            response = requests.get(url, params=params)
            data = response.json()
            if len(data) < 15: return None
                
            df = pd.DataFrame(data)
            df['prev_close'] = df['close'].shift(1)
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['prev_close'])
            df['tr3'] = abs(df['low'] - df['prev_close'])
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            atr = df['true_range'].rolling(window=14).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            stop_distance = 2 * atr
            stop_price = current_price - stop_distance
            max_loss_dollars = account_value * risk_pct
            shares_to_buy = int(max_loss_dollars / stop_distance) if stop_distance > 0 else 0
                
            return {
                'Current_Price': round(current_price, 2),
                'ATR': round(atr, 2),
                'Stop_Loss': round(stop_price, 2),
                'Shares': shares_to_buy,
                'Total_Investment': round(shares_to_buy * current_price, 2),
                'Max_Loss_Risk': round(max_loss_dollars, 2)
            }
        except Exception:
            return None

    def get_news(self, ticker: str) -> str:
        """Fetches the latest 5 news headlines from Tiingo."""
        try:
            url = f"https://api.tiingo.com/tiingo/news?tickers={ticker}&limit=5"
            response = requests.get(url, headers=self.headers)
            articles = response.json()
            if not articles: return "No recent news."

            headlines = []
            for article in articles:
                title = article.get('title', 'No Title')
                source = article.get('source', 'Unknown')
                pub_date = article.get('publishedDate', '')[:10]
                headlines.append(f"- [{pub_date}] {title} ({source})")
            return "\n".join(headlines)
        except Exception:
            return "Error fetching news."

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
                return f"⚠️ EARNINGS IN {days_until} DAYS" if days_until <= 7 else f"Safe (Earnings in {days_until} days)"
            return "Safe (Awaiting next quarter date)"
        except Exception:
            return "Safe (Data Error)"
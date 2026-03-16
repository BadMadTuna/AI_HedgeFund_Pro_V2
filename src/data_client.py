import os
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import time
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

    def get_market_regime(self, ticker='SPY'):
        """
        Advanced Layer 1: Trend-Volatility Matrix Classifier.
        Evaluates Trend (Moving Averages) and Realized Volatility to classify the market state.
        """
        try:
            # Fetch 1 year of data to calculate long-term baselines
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty:
                return None

            close_prices = hist['Close']
            
            # 1. Calculate Trend (Fast vs Slow SMA Crossover is more robust than just Price vs SMA)
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            current_price = close_prices.iloc[-1]
            
            # Is the structural trend up or down?
            is_uptrend = sma_50 > sma_200 and current_price > sma_200

            # 2. Calculate Realized Volatility (Rolling 20-day annualized std dev)
            daily_returns = close_prices.pct_change()
            rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
            
            current_vol = rolling_vol.iloc[-1]
            # Calculate the median volatility over the last year to establish a "normal" baseline
            baseline_vol = rolling_vol.median() 
            
            is_high_vol = current_vol > baseline_vol

            # 3. Classify the Regime
            if is_uptrend and not is_high_vol:
                regime = "QUIET_BULL"
                action = "Aggressive Trend/Momentum"
            elif is_uptrend and is_high_vol:
                regime = "VOLATILE_BULL"
                action = "Reduce Size, Blend Momentum with Mean Reversion"
            elif not is_uptrend and not is_high_vol:
                regime = "QUIET_BEAR"
                action = "Defensive Value, High Yield, Cash"
            else: # not is_uptrend and is_high_vol
                regime = "VOLATILE_BEAR"
                action = "Maximum Cash, Trade Extreme Oversold Bounces Only"

            return {
                'regime': regime,
                'recommended_action': action,
                'metrics': {
                    'current_price': round(current_price, 2),
                    'sma_50': round(sma_50, 2),
                    'sma_200': round(sma_200, 2),
                    'current_volatility': round(current_vol * 100, 2), # As percentage
                    'baseline_volatility': round(baseline_vol * 100, 2)
                }
            }
        except Exception as e:
            print(f"Regime Engine Error: {e}")
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
        
    def get_fundamentals(self, ticker: str) -> dict:
        """Fetches fundamental data with a 12-hour Time-to-Live (TTL) cache."""
        
        # 1. Initialize cache if it doesn't exist
        if not hasattr(self, 'fund_cache'):
            self.fund_cache = {}
            
        # 2. Set Expiration: 12 hours (in seconds)
        CACHE_TTL = 12 * 3600 
        current_time = time.time()
            
        # 3. Check if we have valid, unexpired data
        if ticker in self.fund_cache:
            cached_data, timestamp = self.fund_cache[ticker]
            
            # If the data is younger than 12 hours, use it
            if (current_time - timestamp) < CACHE_TTL:
                return cached_data
            else:
                # The data is stale. Delete it so we fetch fresh data.
                del self.fund_cache[ticker]
                
        try:
            info = yf.Ticker(ticker).info
            
            roe = info.get('returnOnEquity', 0)
            gross_margin = info.get('grossMargins', 0)
            ev_ebitda = info.get('enterpriseToEbitda', 0)
            fcf = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 1)
            
            fcf_yield = fcf / market_cap if market_cap and fcf else 0
            
            result = {
                'ROE': roe if roe else 0,
                'Gross_Margin': gross_margin if gross_margin else 0,
                'EV_EBITDA': ev_ebitda if ev_ebitda else 0,
                'FCF_Yield': fcf_yield if fcf_yield else 0
            }
            
            # 4. Save the successful result AND the current timestamp to memory
            if result['FCF_Yield'] != 0 or result['ROE'] != 0:
                self.fund_cache[ticker] = (result, current_time)
                
            return result
            
        except Exception:
            return {'ROE': 0, 'Gross_Margin': 0, 'EV_EBITDA': 0, 'FCF_Yield': 0}

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
        
    def get_mean_reversion_metrics(self, ticker):
        """
        Layer 2 (Engine B): Mean Reversion Calculator.
        Finds extreme oversold conditions in choppy/volatile regimes using Bollinger Bands.
        """
        try:
            # We need ~60 days to calculate a clean 20-day moving average and standard deviation
            end_date = datetime.today()
            start_date = end_date - timedelta(days=90)
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {'startDate': start_date.strftime('%Y-%m-%d'), 'token': self.api_key}
            
            response = requests.get(url, params=params)
            data = response.json()
            if len(data) < 25: return None
                
            df = pd.DataFrame(data)
            df['close'] = df['close'].astype(float)
            
            # 1. Calculate Bollinger Bands (20-day SMA, 2 Standard Deviations)
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['STD_20'] = df['close'].rolling(window=20).std()
            df['Lower_BB'] = df['SMA_20'] - (2 * df['STD_20'])
            
            # 2. Calculate 14-day RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            current_price = df['close'].iloc[-1]
            lower_bb = df['Lower_BB'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            
            # 3. The Mean Reversion Trigger
            # Is the price within 1% of the lower band (or below it) AND RSI extremely low?
            is_oversold = current_price <= (lower_bb * 1.01) and rsi < 35.0
            
            # Calculate how far we are from the "Mean" (the 20-day SMA) for potential profit target
            sma_20 = df['SMA_20'].iloc[-1]
            upside_to_mean = (sma_20 - current_price) / current_price
            
            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'Lower_BB': round(lower_bb, 2),
                'RSI': round(rsi, 1),
                'Upside_to_Mean': round(upside_to_mean, 4),
                'Is_Oversold_Setup': is_oversold
            }
        except Exception as e:
            return None
        
    def get_deep_value_metrics(self, ticker: str) -> dict:
        """
        Layer 2 (Engine C): Deep Value & Yield Calculator.
        Hunts for dividend yield, low multiples, and low debt in Quiet Bear regimes.
        """
        import yfinance as yf
        try:
            info = yf.Ticker(ticker).info
            
            # 1. Shareholder Yield (Dividends)
            div_yield = info.get('dividendYield', 0)
            if div_yield is None: div_yield = 0
                
            # 2. Valuation Multiples
            ev_ebitda = info.get('enterpriseToEbitda', 99) # Default high if missing
            price_to_book = info.get('priceToBook', 99)
            
            # 3. Balance Sheet Safety (Crucial for Bear Markets)
            debt_to_equity = info.get('debtToEquity', 999) 
            
            # 4. Free Cash Flow
            fcf = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 1)
            fcf_yield = fcf / market_cap if market_cap and fcf else 0
            
            # 5. Build the Value Score (Higher is better)
            # Reward high yield and FCF, penalize high debt and high multiples
            score = (div_yield * 200) + (fcf_yield * 100)
            
            if ev_ebitda < 10: score += 20
            elif ev_ebitda > 20: score -= 30
                
            if debt_to_equity < 50: score += 20 # Low debt is a premium in a bear market
            
            current_price = info.get('currentPrice', info.get('previousClose', 0))
            
            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'Dividend_Yield': round(div_yield, 4),
                'EV_EBITDA': round(ev_ebitda, 2),
                'Debt_to_Equity': round(debt_to_equity, 2),
                'FCF_Yield': round(fcf_yield, 4),
                'Value_Score': round(score, 2)
            }
        except Exception as e:
            return None
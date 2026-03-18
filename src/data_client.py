import os
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MarketDataClient:
    # 1. Sector Map defined as a class-level constant to prevent 500+ YF API calls
    SECTOR_MAP = {
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'ADBE', 'CRM', 'AMD', 'QCOM', 'TXN', 'INTC', 'MU', 'LRCX', 'ADI', 'AMAT', 'KLAC'],
        'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'BKNG', 'TJX', 'ORLY', 'MAR', 'F', 'GM', 'DG', 'EBAY'],
        'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C', 'CB', 'PGR', 'SCHW'],
        'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'HAL', 'DVN', 'HES', 'OXY'],
        'XLV': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'PFE', 'DHR', 'AMGN', 'ISRG', 'SYK', 'VRTX', 'BMY', 'CVS'],
        'XLC': ['GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'EA', 'TTWO'],
        'XLI': ['CAT', 'GE', 'UNP', 'BA', 'RTX', 'HON', 'UPS', 'LMT', 'DE', 'ADP', 'CSX', 'MMM', 'NSC', 'FDX', 'ETN'],
        'XLP': ['PG', 'COST', 'WMT', 'PEP', 'KO', 'PM', 'MO', 'TGT', 'CL', 'KMB', 'MDLZ', 'SYY', 'EL', 'GIS'],
        'XLU': ['NEE', 'SO', 'DUK', 'SRE', 'AEP', 'D', 'EXC', 'XEL', 'ED', 'PCG', 'PEG', 'WEC', 'EIX'],
        'XLRE': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB', 'EQR', 'VTR'],
        'XLB': ['LIN', 'SHW', 'APD', 'ECL', 'NEM', 'FCX', 'DD', 'DOW', 'CTVA', 'NUE', 'VMC', 'MLM', 'ALB']
    }

    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        self.headers = {'Content-Type': 'application/json'}
        
        # Initialize Universal Memory Banks
        if 'mdc_caches' not in st.session_state:
            st.session_state.mdc_caches = {
                'regime': {}, 'fund': {}, 'tech': {}, 
                'mom': {}, 'rev': {}, 'val': {}, 'sector': {}
            }
        self.caches = st.session_state.mdc_caches

    # ==========================================
    # ROBUST RETRY WRAPPERS (Prevents YF Dropouts)
    # ==========================================
    def _get_history_with_retry(self, ticker: str, period: str, retries=3):
        """Fetches YF history with exponential backoff to bypass rate limits."""
        for attempt in range(retries):
            try:
                hist = yf.Ticker(ticker).history(period=period)
                if not hist.empty:
                    return hist
            except Exception:
                pass
            time.sleep(0.3 * (attempt + 1)) # Sleep and try again
        return pd.DataFrame()

    def _get_info_with_retry(self, ticker: str, retries=3):
        """Fetches YF fundamental info with backoff."""
        for attempt in range(retries):
            try:
                info = yf.Ticker(ticker).info
                if info and ('returnOnEquity' in info or 'marketCap' in info):
                    return info
            except Exception:
                pass
            time.sleep(0.3 * (attempt + 1))
        return {}

    # ==========================================
    # UNIVERSAL CACHE MANAGER
    # ==========================================
    def _check_cache(self, cache_name: str, key: str, ttl_seconds: int = 3600):
        if key in self.caches[cache_name]:
            data, timestamp = self.caches[cache_name][key]
            if time.time() - timestamp < ttl_seconds:
                return data
        return None

    def _save_cache(self, cache_name: str, key: str, data):
        if data is not None:
            self.caches[cache_name][key] = (data, time.time())

    # ==========================================
    # LAYER 1: THE BRAIN (DUAL-HYSTERESIS)
    # ==========================================
    def get_market_regime(self, ticker="SPY") -> dict:
        cached = self._check_cache('regime', ticker, ttl_seconds=3600)
        if cached: return cached

        try:
            hist = self._get_history_with_retry(ticker, "1y")
            if hist.empty: return None
            
            close_prices = hist['Close']
            current_price = close_prices.iloc[-1]
            sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            
            daily_returns = close_prices.pct_change().dropna()
            current_vol = daily_returns.tail(20).std() * np.sqrt(252) * 100
            baseline_vol = daily_returns.std() * np.sqrt(252) * 100
            
            last_trend = st.session_state.get('last_trend', 'BEAR')
            last_vol_state = st.session_state.get('last_vol_state', 'VOLATILE')

            trend_upper_band = sma_200 * 1.015
            trend_lower_band = sma_200 * 0.985
            if current_price > trend_upper_band: current_trend = 'BULL'
            elif current_price < trend_lower_band: current_trend = 'BEAR'
            else: current_trend = last_trend 

            vol_upper_band = baseline_vol + 1.0
            vol_lower_band = baseline_vol - 1.0
            if current_vol > vol_upper_band: current_vol_state = 'VOLATILE'
            elif current_vol < vol_lower_band: current_vol_state = 'QUIET'
            else: current_vol_state = last_vol_state 
            
            st.session_state.last_trend = current_trend
            st.session_state.last_vol_state = current_vol_state
            
            regime_name = f"{current_vol_state}_{current_trend}"
            directives = {
                "QUIET_BULL": "Aggressive Trend/Momentum",
                "VOLATILE_BULL": "Mean Reversion / Profit Taking",
                "QUIET_BEAR": "Deep Value / Dividend Yield",
                "VOLATILE_BEAR": "Maximum Defense / Cash Preservation"
            }
            
            result = {
                'regime': regime_name,
                'recommended_action': directives.get(regime_name, "Unknown"),
                'metrics': {
                    'current_price': round(current_price, 2),
                    'sma_50': round(sma_50, 2),
                    'sma_200': round(sma_200, 2),
                    'current_volatility': round(current_vol, 2),
                    'baseline_volatility': round(baseline_vol, 2)
                }
            }
            self._save_cache('regime', ticker, result)
            return result
        except Exception:
            return None

    # ==========================================
    # FUNDAMENTALS
    # ==========================================
    def get_fundamentals(self, ticker: str) -> dict:
        # 12-hour cache for fundamentals
        cached = self._check_cache('fund', ticker, ttl_seconds=43200)
        if cached: return cached
                
        try:
            info = self._get_info_with_retry(ticker)
            
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
            
            # Only cache if data isn't empty 0s
            if result['FCF_Yield'] != 0 or result['ROE'] != 0:
                self._save_cache('fund', ticker, result)
                
            return result
        except Exception:
            return {'ROE': 0, 'Gross_Margin': 0, 'EV_EBITDA': 0, 'FCF_Yield': 0}

    # ==========================================
    # GENERAL TECHNICALS
    # ==========================================
    def get_technicals(self, ticker: str) -> dict:
        cached = self._check_cache('tech', ticker, ttl_seconds=3600)
        if cached: return cached

        try:
            hist = self._get_history_with_retry(ticker, "3mo")
            if hist.empty: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
            
            result = {
                'Current_Price': round(current_price, 2),
                'Price': round(current_price, 2), 
                'RSI_14': round(rsi, 2),
                'RSI': round(rsi, 2) 
            }
            self._save_cache('tech', ticker, result)
            return result
        except Exception:
            return None

    # ==========================================
    # ENGINE A: MOMENTUM (QUIET BULL)
    # ==========================================
    def get_smart_momentum(self, ticker: str) -> dict:
        cached = self._check_cache('mom', ticker, ttl_seconds=3600)
        if cached: return cached

        try:
            hist = self._get_history_with_retry(ticker, "6mo")
            if len(hist) < 50: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            six_mo_return = (current_price - close.iloc[0]) / close.iloc[0]
            
            x = np.arange(len(close))
            y = close.values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            smooth_score = six_mo_return * r_squared
            
            result = {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                '6m_Return': round(six_mo_return, 4),
                'Trend_Smoothness': round(r_squared, 4),
                'Smooth_Score': round(smooth_score, 4)
            }
            self._save_cache('mom', ticker, result)
            return result
        except Exception:
            return None

    # ==========================================
    # ENGINE B: MEAN REVERSION (VOLATILE REGIMES)
    # ==========================================
    def get_mean_reversion_metrics(self, ticker: str) -> dict:
        cached = self._check_cache('rev', ticker, ttl_seconds=3600)
        if cached: return cached

        try:
            hist = self._get_history_with_retry(ticker, "3mo")
            if len(hist) < 20: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            std_20 = close.rolling(window=20).std().iloc[-1]
            lower_bb = sma_20 - (2 * std_20)
            
            upside_to_mean = (sma_20 - current_price) / current_price if current_price < sma_20 else 0
            is_oversold = current_price < lower_bb
            
            result = {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'SMA_20': round(sma_20, 2),
                'Lower_BB': round(lower_bb, 2),
                'Upside_to_Mean': round(upside_to_mean, 4),
                'Is_Oversold_Setup': is_oversold
            }
            self._save_cache('rev', ticker, result)
            return result
        except Exception:
            return None

    # ==========================================
    # ENGINE C: DEEP VALUE (QUIET BEAR)
    # ==========================================
    def get_deep_value_metrics(self, ticker: str) -> dict:
        cached = self._check_cache('val', ticker, ttl_seconds=3600)
        if cached: return cached

        try:
            info = self._get_info_with_retry(ticker)
            div_yield = info.get('dividendYield', 0)
            if div_yield is None: div_yield = 0
                
            ev_ebitda = info.get('enterpriseToEbitda', 99) 
            debt_to_equity = info.get('debtToEquity', 999) 
            
            fcf = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 1)
            fcf_yield = fcf / market_cap if market_cap and fcf else 0
            
            score = (div_yield * 200) + (fcf_yield * 100)
            if ev_ebitda < 10: score += 20
            elif ev_ebitda > 20: score -= 30
            if debt_to_equity < 50: score += 20 
            
            current_price = info.get('currentPrice', info.get('previousClose', 0))
            
            result = {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'Dividend_Yield': round(div_yield, 4),
                'EV_EBITDA': round(ev_ebitda, 2),
                'Debt_to_Equity': round(debt_to_equity, 2),
                'FCF_Yield': round(fcf_yield, 4),
                'Value_Score': round(score, 2)
            }
            self._save_cache('val', ticker, result)
            return result
        except Exception:
            return None

    # ==========================================
    # RISK MANAGEMENT (ATR SIZING)
    # ==========================================
    def get_atr_and_sizing(self, ticker: str, account_value: float = 100000.0, risk_pct: float = 0.01) -> dict:
        try:
            hist = self._get_history_with_retry(ticker, "1mo")
            if len(hist) < 15: return None
            
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            
            current_price = hist['Close'].iloc[-1]
            risk_amount = account_value * risk_pct
            stop_distance = atr_14 * 2 
            shares = int(risk_amount / stop_distance)
            
            return {
                'Current_Price': current_price,
                'ATR_14': atr_14,
                'Stop_Distance': stop_distance,
                'Stop_Loss': current_price - stop_distance,
                'Shares': shares,
                'Total_Investment': shares * current_price,
                'Max_Loss_Risk': shares * stop_distance
            }
        except Exception:
            return None

    # ==========================================
    # HELPERS
    # ==========================================
    def get_sector_for_ticker(self, ticker: str) -> str:
        """Uses static map to prevent 500 YF API calls."""
        cached = self._check_cache('sector', ticker, ttl_seconds=86400)
        if cached: return cached

        # Look in our predefined map first
        for etf, tickers in self.SECTOR_MAP.items():
            if ticker in tickers:
                self._save_cache('sector', ticker, etf)
                return etf
                
        # Fallback to YF only if unknown
        try:
            info = self._get_info_with_retry(ticker, retries=1)
            sector = info.get('sector', 'Unknown')
            mapping = {
                'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Energy': 'XLE',
                'Utilities': 'XLU', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
                'Real Estate': 'XLRE', 'Communication Services': 'XLC'
            }
            mapped_sector = mapping.get(sector, 'SPY')
            self._save_cache('sector', ticker, mapped_sector)
            return mapped_sector
        except:
            return 'SPY'

    def get_news(self, ticker: str, bypass_cache: bool = False) -> str:
        """Fetches news. Uses 1-hour cache by default, but can be bypassed for live Guardian audits."""
        
        # 1. Check cache ONLY if bypass_cache is False
        if not bypass_cache:
            cached = self._check_cache('news', ticker, ttl_seconds=3600)
            if cached: return cached

        if not self.api_key: return "No News API Key provided."
        
        try:
            url = "https://api.tiingo.com/tiingo/news"
            params = {'tickers': ticker, 'limit': 3, 'token': self.api_key}
            res = requests.get(url, params=params)
            
            if res.status_code == 200:
                articles = res.json()
                if articles:
                    news_str = "\n".join([f"- {a['title']}" for a in articles])
                    
                    # 2. Only save to cache if we aren't actively bypassing it
                    if not bypass_cache:
                        self._save_cache('news', ticker, news_str)
                        
                    return news_str
                    
            if not bypass_cache:
                self._save_cache('news', ticker, "No recent major news.")
            return "No recent major news."
        except:
            return "Failed to fetch news."

    def get_earnings_date(self, ticker: str) -> str:
        try:
            t = yf.Ticker(ticker)
            calendar = t.calendar
            if calendar is not None and not calendar.empty:
                if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if len(dates) > 0:
                        next_date = dates[0]
                        if isinstance(next_date, datetime):
                            days_away = (next_date.replace(tzinfo=None) - datetime.now()).days
                        else:
                            return f"Next Earnings: {next_date}"
                            
                        if days_away < 0: return "Earnings recently passed."
                        if days_away < 7: return f"⚠️ EARNINGS IN {days_away} DAYS"
                        return f"Safe (Earnings in {days_away} days)"
            return "Unknown"
        except:
            return "Unknown"
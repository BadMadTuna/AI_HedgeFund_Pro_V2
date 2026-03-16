import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import time
import streamlit as st

class MarketDataClient:
    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        self.headers = {'Content-Type': 'application/json'}
        
    # ==========================================
    # LAYER 1: THE BRAIN (DUAL-HYSTERESIS)
    # ==========================================
    def get_market_regime(self, ticker="SPY") -> dict:
        """
        Calculates the Macro Regime using Dual-Hysteresis 
        to perfectly prevent both Trend and Volatility whipsaws.
        """
        try:
            # 1. Fetch 1 year of data
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty: return None
            
            close_prices = hist['Close']
            current_price = close_prices.iloc[-1]
            sma_200 = close_prices.rolling(window=200).mean().iloc[-1]
            sma_50 = close_prices.rolling(window=50).mean().iloc[-1]
            
            # 2. Calculate Volatility
            daily_returns = close_prices.pct_change().dropna()
            current_vol = daily_returns.tail(20).std() * np.sqrt(252) * 100
            baseline_vol = daily_returns.std() * np.sqrt(252) * 100
            
            # ==========================================
            # DUAL-HYSTERESIS LOGIC
            # ==========================================
            
            # Fetch the previous states from memory (default to safe/bearish if no memory)
            last_trend = st.session_state.get('last_trend', 'BEAR')
            last_vol_state = st.session_state.get('last_vol_state', 'VOLATILE')

            # --- TREND HYSTERESIS (1.5% Buffer) ---
            trend_upper_band = sma_200 * 1.015
            trend_lower_band = sma_200 * 0.985
            
            if current_price > trend_upper_band: current_trend = 'BULL'
            elif current_price < trend_lower_band: current_trend = 'BEAR'
            else: current_trend = last_trend # Stuck in the dead zone, keep previous state

            # --- VOLATILITY HYSTERESIS (1.0% Buffer) ---
            vol_upper_band = baseline_vol + 1.0
            vol_lower_band = baseline_vol - 1.0
            
            if current_vol > vol_upper_band: current_vol_state = 'VOLATILE'
            elif current_vol < vol_lower_band: current_vol_state = 'QUIET'
            else: current_vol_state = last_vol_state # Stuck in the dead zone, keep previous state
            
            # Save the confirmed states back to memory for tomorrow
            st.session_state.last_trend = current_trend
            st.session_state.last_vol_state = current_vol_state
            
            # 3. Combine into the Final Regime
            regime_name = f"{current_vol_state}_{current_trend}"

            # 4. Map the Directive
            directives = {
                "QUIET_BULL": "Aggressive Trend/Momentum",
                "VOLATILE_BULL": "Mean Reversion / Profit Taking",
                "QUIET_BEAR": "Deep Value / Dividend Yield",
                "VOLATILE_BEAR": "Maximum Defense / Cash Preservation"
            }
            
            return {
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
        except Exception as e:
            return None

    # ==========================================
    # FUNDAMENTALS & CACHING
    # ==========================================
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

    # ==========================================
    # GENERAL TECHNICALS
    # ==========================================
    def get_technicals(self, ticker: str) -> dict:
        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            if hist.empty: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            
            # Basic RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50
            
            return {
                'Current_Price': round(current_price, 2),
                'Price': round(current_price, 2), # Fallback key
                'RSI_14': round(rsi, 2),
                'RSI': round(rsi, 2) # Fallback key
            }
        except Exception:
            return None

    # ==========================================
    # ENGINE A: MOMENTUM (QUIET BULL)
    # ==========================================
    def get_smart_momentum(self, ticker: str) -> dict:
        """Calculates trend smoothness and raw momentum."""
        try:
            hist = yf.Ticker(ticker).history(period="6mo")
            if len(hist) < 50: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            
            # 1. Raw Return
            six_mo_return = (current_price - close.iloc[0]) / close.iloc[0]
            
            # 2. Path Smoothness (R-Squared of price vs time)
            x = np.arange(len(close))
            y = close.values
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # 3. Smooth Score (Return * Smoothness)
            smooth_score = six_mo_return * r_squared
            
            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                '6m_Return': round(six_mo_return, 4),
                'Trend_Smoothness': round(r_squared, 4),
                'Smooth_Score': round(smooth_score, 4)
            }
        except Exception:
            return None

    # ==========================================
    # ENGINE B: MEAN REVERSION (VOLATILE REGIMES)
    # ==========================================
    def get_mean_reversion_metrics(self, ticker: str) -> dict:
        """Hunts for statistically oversold conditions below the Bollinger Band."""
        try:
            hist = yf.Ticker(ticker).history(period="3mo")
            if len(hist) < 20: return None
            
            close = hist['Close']
            current_price = close.iloc[-1]
            
            # Bollinger Bands (20-day, 2 Standard Deviations)
            sma_20 = close.rolling(window=20).mean().iloc[-1]
            std_20 = close.rolling(window=20).std().iloc[-1]
            lower_bb = sma_20 - (2 * std_20)
            
            # Upside to Mean (How far is it from its 20-day baseline?)
            upside_to_mean = (sma_20 - current_price) / current_price if current_price < sma_20 else 0
            
            # Oversold Logic
            is_oversold = current_price < lower_bb
            
            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'SMA_20': round(sma_20, 2),
                'Lower_BB': round(lower_bb, 2),
                'Upside_to_Mean': round(upside_to_mean, 4),
                'Is_Oversold_Setup': is_oversold
            }
        except Exception:
            return None

    # ==========================================
    # ENGINE C: DEEP VALUE (QUIET BEAR)
    # ==========================================
    def get_deep_value_metrics(self, ticker: str) -> dict:
        """Hunts for dividend yield, low multiples, and low debt in Quiet Bear regimes."""
        try:
            info = yf.Ticker(ticker).info
            
            # 1. Shareholder Yield (Dividends)
            div_yield = info.get('dividendYield', 0)
            if div_yield is None: div_yield = 0
                
            # 2. Valuation Multiples
            ev_ebitda = info.get('enterpriseToEbitda', 99) 
            
            # 3. Balance Sheet Safety
            debt_to_equity = info.get('debtToEquity', 999) 
            
            # 4. Free Cash Flow
            fcf = info.get('freeCashflow', 0)
            market_cap = info.get('marketCap', 1)
            fcf_yield = fcf / market_cap if market_cap and fcf else 0
            
            # 5. Build the Value Score
            score = (div_yield * 200) + (fcf_yield * 100)
            
            if ev_ebitda < 10: score += 20
            elif ev_ebitda > 20: score -= 30
            if debt_to_equity < 50: score += 20 
            
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
        except Exception:
            return None

    # ==========================================
    # RISK MANAGEMENT (ATR SIZING)
    # ==========================================
    def get_atr_and_sizing(self, ticker: str, account_value: float = 100000.0, risk_pct: float = 0.01) -> dict:
        """Calculates position size using the 14-day Average True Range (ATR)."""
        try:
            hist = yf.Ticker(ticker).history(period="1mo")
            if len(hist) < 15: return None
            
            high_low = hist['High'] - hist['Low']
            high_close = np.abs(hist['High'] - hist['Close'].shift())
            low_close = np.abs(hist['Low'] - hist['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            
            current_price = hist['Close'].iloc[-1]
            
            # Risk Math
            risk_amount = account_value * risk_pct
            stop_distance = atr_14 * 2 # 2x ATR Stop Loss
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
        """Maps an S&P 500 stock to its parent sector ETF for relative strength checks."""
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            
            mapping = {
                'Technology': 'XLK', 'Healthcare': 'XLV', 'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY', 'Consumer Defensive': 'XLP', 'Energy': 'XLE',
                'Utilities': 'XLU', 'Industrials': 'XLI', 'Basic Materials': 'XLB',
                'Real Estate': 'XLRE', 'Communication Services': 'XLC'
            }
            return mapping.get(sector, 'SPY')
        except:
            return 'SPY'

    def get_news(self, ticker: str) -> str:
        """Fetches latest news from Tiingo."""
        if not self.api_key: return "No News API Key provided."
        try:
            url = "https://api.tiingo.com/tiingo/news"
            params = {'tickers': ticker, 'limit': 3, 'token': self.api_key}
            res = requests.get(url, params=params)
            
            if res.status_code == 200:
                articles = res.json()
                if articles:
                    return "\n".join([f"- {a['title']}" for a in articles])
            return "No recent major news."
        except:
            return "Failed to fetch news."

    def get_earnings_date(self, ticker: str) -> str:
        """Returns the next earnings date."""
        try:
            t = yf.Ticker(ticker)
            calendar = t.calendar
            if calendar is not None and not calendar.empty:
                # Format depends on yfinance version, usually dict or dataframe
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
import streamlit as st
import pandas as pd
import concurrent.futures
import requests
import time
import sqlite3
from datetime import datetime, timedelta
import os
import numpy as np
import yfinance as yf
from src.data_client import MarketDataClient
from src.database import get_portfolio_df, get_journal_df
from src.ai_agent import AIAgent
from src.portfolio import PortfolioManager

# --- INIT ---
st.set_page_config(page_title="🦅 AI Hedge Fund Pro", layout="wide", page_icon="📈")

@st.cache_resource
def get_clients():
    return MarketDataClient(), AIAgent(), PortfolioManager()

data_client, agent, pm = get_clients()

# --- HELPERS ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_sp500_tickers():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        
        tables = pd.read_html(response.text)
        df = tables[0]
        # Clean tickers (Wikipedia uses dots, Tiingo/YFinance use hyphens)
        return df['Symbol'].str.replace('.', '-').tolist()
    except Exception as e:
        st.error(f"Failed to fetch S&P 500: {e}. Using default tech list.")
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]

# --- HEADER ---
st.title("🦅 AI Hedge Fund Manager (Pro V2)")
st.markdown("Powered by Tiingo, Gemini 2.5 Pro, and SQLite.")

# --- TABS ---
tab_port, tab_radar, tab_analyze, tab_journal, tab_backtest = st.tabs([
    "📂 Portfolio", "🎯 Radar Scan", "🔍 Deep Analyzer", "📓 Trade Journal", "📈 Backtester"
])

# ==========================================
# TAB 1: PORTFOLIO & GUARDIAN
# ==========================================
with tab_port:
    st.header("Portfolio Overview")
    
    df_port = get_portfolio_df()
    
    if "live_port_df" not in st.session_state:
        st.session_state.live_port_df = None
        st.session_state.current_fx_rate = 1.0
    if "guardian_audit_df" not in st.session_state:
        st.session_state.guardian_audit_df = None

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        refresh_clicked = st.button("🔄 Refresh Live Prices & PnL", type="primary", use_container_width=True)

    if refresh_clicked and not df_port.empty:
        with st.spinner("Fetching live market data and FX rates from Tiingo..."):
            try:
                api_key = getattr(data_client, 'api_key', None) 
                if not api_key: api_key = os.getenv("TIINGO_API_KEY")

                fx_url = "https://api.tiingo.com/tiingo/fx/top"
                fx_res = requests.get(fx_url, params={'tickers': 'eurusd', 'token': api_key})
                fx_res.raise_for_status()
                fx_data = fx_res.json()
                
                eur_usd_rate = fx_data[0]['midPrice']
                usd_to_eur = 1.0 / eur_usd_rate
            except Exception as e:
                st.warning(f"Could not fetch live FX rate ({e}). Defaulting to 0.92.")
                usd_to_eur = 0.92

            def fetch_live_price(row):
                ticker = row['ticker']
                if ticker == 'EUR': return {'ticker': 'EUR', 'live_price': 1.0}
                try:
                    tech = data_client.get_technicals(ticker)
                    if tech and 'Price' in tech:
                        return {'ticker': ticker, 'live_price': tech['Price'] * usd_to_eur}
                    return {'ticker': ticker, 'live_price': row['cost']}
                except: return {'ticker': ticker, 'live_price': row['cost']}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_live_price, row): row for _, row in df_port.iterrows()}
                live_prices = {fut.result()['ticker']: fut.result()['live_price'] for fut in concurrent.futures.as_completed(futures)}
            
            live_df = df_port.copy()
            live_df['Live Price (€)'] = live_df['ticker'].map(live_prices)
            live_df['Current Value (€)'] = live_df['Live Price (€)'] * live_df['quantity']
            
            is_stock = live_df['ticker'] != 'EUR'
            live_df['PnL (€)'] = 0.0
            live_df['PnL (%)'] = 0.0
            live_df.loc[is_stock, 'PnL (€)'] = (live_df['Live Price (€)'] - live_df['cost']) * live_df['quantity']
            live_df.loc[is_stock, 'PnL (%)'] = ((live_df['Live Price (€)'] - live_df['cost']) / live_df['cost']) * 100
            
            # --- UPGRADED: SMART TREND & SOFT STOP TRACKER ---
            live_df['Hard Stop (€)'] = 0.0
            live_df.loc[is_stock, 'Hard Stop (€)'] = live_df['cost'] * 0.92  # The -8% threshold
            live_df['Trend Status'] = 'OK'
            
            with st.spinner("Checking technical trendlines for your holdings..."):
                for idx, row in live_df[is_stock].iterrows():
                    try:
                        t = row['ticker']
                        hist = yf.Ticker(t).history(period="1mo")
                        if len(hist) >= 20:
                            sma_20 = hist['Close'].tail(20).mean()
                            current_close = hist['Close'].iloc[-1]
                            if current_close < sma_20:
                                live_df.at[idx, 'Trend Status'] = 'BROKEN'
                    except Exception:
                        pass 
            
            def check_eod_status(row):
                if row['ticker'] == 'EUR': return '-'
                if row['Live Price (€)'] <= row['Hard Stop (€)']: 
                    return '🚨 SELL (-8% Breach)'
                elif row['Trend Status'] == 'BROKEN':
                    return '🚨 SELL (Trend Broken)'
                elif row['PnL (%)'] >= 20.0: 
                    return '🔥 TRIM (+20% Target)'
                else: 
                    return '✅ SAFE'
                
            live_df['EOD Status'] = live_df.apply(check_eod_status, axis=1)
            # -------------------------------------------------
            
            st.session_state.live_port_df = live_df
            st.session_state.current_fx_rate = usd_to_eur

    if st.session_state.live_port_df is not None:
        live_df = st.session_state.live_port_df
        cash = live_df[live_df['ticker'] == 'EUR']['Current Value (€)'].sum() if 'EUR' in live_df['ticker'].values else 0.0
        inv_cost = live_df[live_df['ticker'] != 'EUR']['cost'].multiply(live_df[live_df['ticker'] != 'EUR']['quantity']).sum()
        inv_val = live_df[live_df['ticker'] != 'EUR']['Current Value (€)'].sum()
        pnl_eur = live_df['PnL (€)'].sum()
        total_equity = cash + inv_val
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Live Total Equity", f"€{total_equity:,.2f}", f"{pnl_eur:+,.2f} €")
        c2.metric("💵 Cash Available", f"€{cash:,.2f}")
        c3.metric("📈 Total Return", f"{(pnl_eur/inv_cost*100 if inv_cost>0 else 0):+.2f}%")
        c4.metric("📊 Live Invested Value", f"€{inv_val:,.2f}")
        
        st.caption(f"🌍 *Live USD to EUR Conversion Rate:* **{st.session_state.current_fx_rate:.4f}**")

        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #10b981;' 
                elif val < 0: return 'color: #ef4444;' 
            return ''
            
        def color_status(val):
            if 'SELL' in str(val): return 'background-color: #ef4444; color: white; font-weight: bold;'
            if 'TRIM' in str(val): return 'background-color: #f59e0b; color: white; font-weight: bold;'
            return ''
            
        styled_df = live_df.style.format({
            'cost': '€{:.2f}', 'Live Price (€)': '€{:.2f}', 'Current Value (€)': '€{:.2f}', 
            'PnL (€)': '€{:.2f}', 'PnL (%)': '{:.2f}%', 'Hard Stop (€)': '€{:.2f}'
        }).map(color_pnl, subset=['PnL (€)', 'PnL (%)']).map(color_status, subset=['EOD Status'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        summary = pm.get_equity_summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Equity (Cost Basis)", f"€{summary['total_equity']:,.2f}")
        col2.metric("💵 Cash Available", f"€{summary['cash']:,.2f}")
        col3.metric("📈 Invested Capital", f"€{summary['invested']:,.2f}")
        st.dataframe(df_port, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🛠️ Management & Portfolio Injection")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        with st.expander("➕ Add Single Position / Deposit Cash"):
            with st.form("buy_form"):
                b_ticker = st.text_input("Ticker (Use 'EUR' for Cash)", "AAPL").upper()
                b_price = st.number_input("Entry Price (1.0 for EUR)", min_value=0.0, value=150.0)
                b_qty = st.number_input("Quantity", min_value=1.0, value=10.0)
                b_target = st.number_input("Target Price (Optional)", min_value=0.0, value=0.0)
                if st.form_submit_button("Execute Buy / Deposit"):
                    if pm.execute_buy(b_ticker, b_price, b_qty, b_target):
                        st.success(f"Successfully added {b_qty} of {b_ticker}!")
                        st.rerun()
                    else:
                        st.error("Failed. Check cash balance or inputs.")

        with st.expander("📥 Bulk Inject Existing Portfolio (No Cash Deduction)"):
            st.write("Format: `TICKER, QUANTITY, AVG_PRICE` (Use 'EUR' to set initial cash)")
            with st.form("bulk_inject_form"):
                bulk_data = st.text_area("Paste portfolio data here:")
                if st.form_submit_button("Force Inject"):
                    try:
                        conn = sqlite3.connect("data/hedgefund.db")
                        cursor = conn.cursor()
                        success_count = 0
                        for line in bulk_data.split('\n'):
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                ticker = parts[0].upper()
                                qty = float(parts[1])
                                price = float(parts[2])
                                date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                cursor.execute("SELECT id, quantity, cost FROM portfolio WHERE ticker = ? AND status = 'OPEN'", (ticker,))
                                row = cursor.fetchone()
                                if row:
                                    new_qty = row[1] + qty
                                    new_cost = ((row[1] * row[2]) + (qty * price)) / new_qty
                                    cursor.execute("UPDATE portfolio SET quantity = ?, cost = ? WHERE id = ?", (new_qty, new_cost, row[0]))
                                else:
                                    cursor.execute("INSERT INTO portfolio (ticker, cost, quantity, target, status, date_acquired) VALUES (?, ?, ?, 0.0, 'OPEN', ?)", 
                                                   (ticker, price, qty, date_str))
                                success_count += 1
                        conn.commit(); conn.close()
                        st.success(f"Successfully injected {success_count} positions!"); time.sleep(1); st.rerun()
                    except Exception as e: st.error(f"Failed: {e}")

    with m_col2:
        with st.expander("✂️ Sell / Trim Position"):
            with st.form("sell_form"):
                s_ticker = st.text_input("Ticker to Sell").upper()
                s_price = st.number_input("Exit Price", min_value=0.0, value=150.0)
                s_qty = st.number_input("Quantity to Sell", min_value=1.0, value=10.0)
                if st.form_submit_button("Execute Sell / Trim"):
                    if pm.execute_sell(s_ticker, s_price, s_qty):
                        st.success(f"Successfully sold {s_qty} of {s_ticker}!"); st.rerun()
                    else:
                        st.error("Failed. Ensure you own enough shares of the stock.")

    st.markdown("---")

    # --- PORTFOLIO CORRELATION MATRIX ---
    st.markdown("---")
    st.subheader("🕸️ Portfolio Correlation Matrix")
    with st.expander("Analyze Cross-Asset Risk & Diversification", expanded=False):
        active_tickers = df_port[df_port['ticker'] != 'EUR']['ticker'].unique().tolist()
        
        if len(active_tickers) < 2:
            st.info("You need at least 2 active stock positions to calculate correlation.")
        else:
            if st.button("Generate Correlation Heatmap"):
                with st.spinner("Fetching institutional data via Tiingo..."):
                    try:
                        clean_tickers = [str(t).strip().upper() for t in active_tickers if pd.notna(t) and str(t).strip() != '']
                        if len(clean_tickers) < 2:
                            st.warning("Not enough valid tickers to run correlation.")
                        else:
                            price_dict = {}
                            start_str = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
                            api_key = getattr(data_client, 'api_key', None) 
                            if not api_key: api_key = os.getenv("TIINGO_API_KEY")
                            
                            for t in clean_tickers:
                                try:
                                    if "." in t: 
                                        hist = yf.Ticker(t).history(period="6m")
                                        if not hist.empty and 'Close' in hist.columns:
                                            hist.index = hist.index.tz_localize(None) 
                                            price_dict[t] = hist['Close']
                                    else: 
                                        url = f"https://api.tiingo.com/tiingo/daily/{t}/prices"
                                        params = {'startDate': start_str, 'token': api_key}
                                        res = requests.get(url, params=params)
                                        if res.status_code == 200:
                                            j_data = res.json()
                                            if len(j_data) > 0:
                                                df_t = pd.DataFrame(j_data)
                                                df_t['date'] = pd.to_datetime(df_t['date']).dt.tz_localize(None)
                                                df_t.set_index('date', inplace=True)
                                                price_dict[t] = df_t['close']
                                except Exception:
                                    pass 
                                    
                            if len(price_dict) < 2:
                                st.error("Could not fetch enough data. Check your Tiingo API key or limit.")
                            else:
                                close_df = pd.DataFrame(price_dict)
                                close_df = close_df.apply(pd.to_numeric, errors='coerce')
                                close_df = close_df.dropna(axis=1, how='all')
                                close_df = close_df.ffill() 
                                
                                returns = close_df.pct_change()
                                corr_matrix = returns.corr(method='pearson')
                                corr_matrix = corr_matrix.fillna(0)
                                if not corr_matrix.empty:
                                    np.fill_diagonal(corr_matrix.values, 1.0) 
                                
                                st.write("**Pearson Correlation Coefficient (Last 6 Months):**")
                                st.caption("🔴 :red[**Red**] = Danger/High Correlation | 🟢 :green[**Green**] = Safe/Low Correlation")
                                
                                styled_corr = corr_matrix.style.background_gradient(cmap='RdYlGn_r', vmin=-0.5, vmax=1.0).format("{:.2f}")
                                st.dataframe(styled_corr, use_container_width=True)
                                
                                high_corr_pairs = []
                                cols = corr_matrix.columns
                                for i in range(len(cols)):
                                    for j in range(i+1, len(cols)):
                                        val = corr_matrix.iloc[i, j]
                                        if val > 0.70 and val < 0.99: 
                                            high_corr_pairs.append(f"**{cols[i]}** & **{cols[j]}** ({val:.2f})")
                                            
                                if high_corr_pairs:
                                    st.error(f"**⚠️ Concentration Risk Detected:** The following pairs are highly correlated (>0.70). If one drops, the other is highly likely to crash with it. Consider rotating one out:")
                                    for pair in high_corr_pairs:
                                        st.write(f"- {pair}")
                                else:
                                    st.success("**✅ Healthy Diversification:** No dangerously correlated pairs detected. Your portfolio risk is well-distributed!")
                                    
                    except Exception as e:
                        st.error(f"Failed to generate correlation matrix. ({e})")
                        
    st.markdown("---")

    if st.button("🛡️ Run AI Guardian Audit on Portfolio"):
        audit_df = st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port
        if audit_df.empty or len(audit_df[audit_df['ticker'] != 'EUR']) == 0:
            st.warning("No active stocks to audit.")
        else:
            with st.spinner("Guardian is analyzing your holdings..."):
                audit_results = []
                for _, row in audit_df[audit_df['ticker'] != 'EUR'].iterrows():
                    ticker = row['ticker']
                    v = agent.get_guardian_audit(ticker, row.to_dict(), data_client.get_news(ticker), data_client.get_earnings_date(ticker))
                    audit_results.append({
                        'Ticker': ticker, 
                        'Action': v.get('action', 'N/A'), 
                        'Earnings Risk': v.get('earnings_risk', 'Unknown'),
                        'AI Advice': v.get('reasoning', ''),
                        'Execution Plan': v.get('proposed_stop', '')
                    })
                st.session_state.guardian_audit_df = pd.DataFrame(audit_results)

    # Note: Placed outside the button so it survives downloads
    if st.session_state.guardian_audit_df is not None:
        st.success("Audit Complete!")
        for _, row in st.session_state.guardian_audit_df.iterrows():
            with st.container(border=True):
                action = str(row['Action']).upper()
                color = "red" if 'SELL' in action else "orange" if 'TRIM' in action else "green"
                st.markdown(f"### {row['Ticker']} | Action: :{color}[**{action}**]")
                
                risk = str(row['Earnings Risk'])
                if 'Critical' in risk or 'Elevated' in risk:
                    st.write(f"**Earnings Risk:** :red[{risk}]")
                else:
                    st.write(f"**Earnings Risk:** {risk}")
                    
                st.write(f"**Advice:** {row['AI Advice']}")
                st.write(f"**Plan:** {row['Execution Plan']}")

        export_df = pd.merge(st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port, 
                             st.session_state.guardian_audit_df, left_on='ticker', right_on='Ticker', how='left')
        st.download_button("📥 Download Full Portfolio & AI Audit Report (CSV)", 
                           export_df.to_csv(index=False).encode('utf-8'), 
                           f"portfolio_audit_{datetime.today().strftime('%Y%m%d')}.csv", 
                           "text/csv")

# ==========================================
# TAB 2: RADAR SCAN
# ==========================================
with tab_radar:
    st.header("🎯 Two-Tier AI Radar Scan")
    
    # Initialize session state variables so data survives the download button
    if "scan_df_final" not in st.session_state:
        st.session_state.scan_df_final = None
        st.session_state.scan_top_20_alpha = None
    
    with st.expander("🌍 Market Regime Check", expanded=True):
        with st.spinner("Checking market trend..."):
            regime = data_client.get_regime('SPY')
            
        if regime:
            if not regime['bullish']:
                st.error("🔴 RISK OFF: Broad Market is in a Downtrend")
                st.warning(f"SPY is trading at ${regime['price']}, BELOW its 200-day SMA of ${regime['sma']}. Institutional models suggest caution, but scanning is enabled.")
            else:
                st.success("🟢 RISK ON: Broad Market is in an Uptrend")
                st.write(f"SPY is trading at ${regime['price']}, ABOVE its 200-day SMA of ${regime['sma']}.")
        else:
            st.warning("Regime check failed.")

    tickers_to_scan = get_sp500_tickers()

    if st.button("🚀 Launch Quantamental Alpha Scan", type="primary"):
        st.subheader("⚙️ Tier 1: Momentum Filter (Broad Scan)")
        quant_results = []
        progress_bar = st.progress(0)
        
        def fetch_quant(t):
            sector_etf = data_client.get_sector_for_ticker(t)
            sector_regime = data_client.get_regime(sector_etf)
            metrics = data_client.get_smart_momentum(t)
            if not metrics: return None
            
            tech = data_client.get_technicals(t)
            if not tech: return None

            return {
                'Ticker': t, 'Sector': sector_etf,
                'Sector Health': sector_regime['status'] if sector_regime else "Unknown",
                'Price': metrics['Current_Price'], 'Smooth_Score': metrics['Smooth_Score'],
                **tech
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_quant, t): t for t in tickers_to_scan}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: quant_results.append(res)
                progress_bar.progress((i + 1) / len(tickers_to_scan))
        
        if quant_results:
            df_quant = pd.DataFrame(quant_results).sort_values(by="Smooth_Score", ascending=False)
            top_50_df = df_quant.head(50)
            
            st.subheader("🔬 Tier 2: Fundamental Alpha Score (Quality + Value)")
            st.write("Fetching corporate fundamentals for the Top 50 momentum leaders...")
            
            fund_results = []
            fund_pb = st.progress(0)
            
            def fetch_fund(row):
                t = row['Ticker']
                funds = data_client.get_fundamentals(t)
                
                smooth_pts = row['Smooth_Score'] * 10 
                quality_pts = (funds['ROE'] * 50) + (funds['Gross_Margin'] * 20)
                ev = funds['EV_EBITDA']
                ev_pts = max(0, 30 - ev) if ev > 0 else 0 
                value_pts = (funds['FCF_Yield'] * 200) + ev_pts
                
                alpha_score = smooth_pts + quality_pts + value_pts
                
                return {
                    **row,
                    'ROE': f"{funds['ROE']:.1%}",
                    'Margin': f"{funds['Gross_Margin']:.1%}",
                    'EV/EBITDA': round(funds['EV_EBITDA'], 1),
                    'Alpha_Score': round(alpha_score, 1)
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures2 = {executor.submit(fetch_fund, row.to_dict()): row for _, row in top_50_df.iterrows()}
                for i, future in enumerate(concurrent.futures.as_completed(futures2)):
                    res = future.result()
                    if res: fund_results.append(res)
                    fund_pb.progress((i + 1) / len(top_50_df))
            
            df_alpha = pd.DataFrame(fund_results).sort_values(by="Alpha_Score", ascending=False)
            top_20_alpha = df_alpha.head(20)
            
            # --- TIER 3: AI HUNTER ---
            st.subheader("🧠 Tier 3: AI Deep Dive (Top 20 Quantamental)")
            final_results = []
            ai_progress = st.progress(0)

            for i, t in enumerate(top_20_alpha['Ticker']):
                tech_fund_data = top_20_alpha.iloc[i].to_dict() 
                news = data_client.get_news(t)
                earn = data_client.get_earnings_date(t)
                
                ai_res = agent.get_hunter_verdict(t, tech_fund_data, news, earn)
                
                final_results.append({
                    "Ticker": t, "Price": tech_fund_data['Price'], "Sector": tech_fund_data['Sector'], 
                    "Alpha Score": tech_fund_data['Alpha_Score'], "Quality": f"ROE: {tech_fund_data['ROE']}",
                    "AI Score": ai_res.get('score', 0), "Verdict": ai_res.get('verdict', 'ERROR'),
                    "Earnings": earn, "Reasoning": ai_res.get('reasoning', '')
                })
                ai_progress.progress((i + 1) / len(top_20_alpha))
                time.sleep(1) 

            df_final = pd.DataFrame(final_results).sort_values(by="AI Score", ascending=False)
            
            # Save results to session state so they persist!
            st.session_state.scan_df_final = df_final
            st.session_state.scan_top_20_alpha = top_20_alpha

    # --- Render Scan UI out here so it survives the download button refresh ---
    if st.session_state.scan_df_final is not None:
        df_final = st.session_state.scan_df_final
        top_20_alpha = st.session_state.scan_top_20_alpha
        
        st.subheader("🔬 Tier 2: Fundamental Alpha Score (Quality + Value)")
        st.dataframe(top_20_alpha[['Ticker', 'Sector', 'Smooth_Score', 'ROE', 'Margin', 'EV/EBITDA', 'Alpha_Score']], use_container_width=True)

        st.subheader("🧠 Tier 3: AI Deep Dive (Top 20 Quantamental)")
        def highlight_verdict(val):
            if val == 'BUY': return 'background-color: #064e3b; color: white;' 
            elif val == 'WATCH': return 'background-color: #78350f; color: white;' 
            elif val == 'AVOID': return 'background-color: #7f1d1d; color: white;' 
            return ''
            
        st.dataframe(df_final.style.map(highlight_verdict, subset=['Verdict']), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("📝 Detailed AI Reasoning & Trade Sizing")
        
        equity_summary = pm.get_equity_summary()
        ACCOUNT_SIZE = equity_summary.get('total_equity', 100000)
        fx_rate = st.session_state.get('current_fx_rate', 0.92)

        for _, row in df_final.iterrows():
            if row['Verdict'] in ['BUY', 'WATCH']: 
                with st.expander(f"{row['Verdict']} | {row['Ticker']} (AI Score: {row['AI Score']})"):
                    st.write(f"**Sector:** {row['Sector']} | **Quantamental Alpha Score:** {row['Alpha Score']}")
                    
                    if row['Verdict'] == 'BUY':
                        is_eu = "." in row['Ticker']
                        math_equity = ACCOUNT_SIZE if is_eu else (ACCOUNT_SIZE / fx_rate)
                        sizing = data_client.get_atr_and_sizing(row['Ticker'], account_value=math_equity, risk_pct=0.01)
                        
                        if sizing:
                            price_eur = sizing['Current_Price'] if is_eu else (sizing['Current_Price'] * fx_rate)
                            invest_eur = sizing['Total_Investment'] if is_eu else (sizing['Total_Investment'] * fx_rate)
                            stop_eur = sizing['Stop_Loss'] if is_eu else (sizing['Stop_Loss'] * fx_rate)
                            risk_eur = sizing['Max_Loss_Risk'] if is_eu else (sizing['Max_Loss_Risk'] * fx_rate)
                            
                            st.success(f"**Execution Plan (1% Risk on €{ACCOUNT_SIZE:,.2f} Total Equity):**\n"
                                       f"- Buy **{sizing['Shares']} shares** at approx **€{price_eur:,.2f}**\n"
                                       f"- Total Capital Deployed: **€{invest_eur:,.2f}**\n"
                                       f"- Hard Stop Loss: **€{stop_eur:,.2f}** (2x ATR)\n"
                                       f"- Max Risk if stopped out: **€{risk_eur:,.2f}**")
                    st.markdown(row['Reasoning'])
        
        st.divider()
        st.download_button("📥 Download Complete Quantamental Report (CSV)", 
                           df_final.to_csv(index=False).encode('utf-8'), 
                           f"quantamental_scan_{datetime.today().strftime('%Y%m%d')}.csv", 
                           "text/csv")

# ==========================================
# TAB 3 & 4
# ==========================================
with tab_analyze:
    st.header("🔍 Deep Stock Analyzer")
    a_ticker = st.text_input("Ticker to Analyze", "NVDA").upper()
    
    if st.button("Analyze Stock"):
        with st.spinner("Gathering data..."):
            tech = data_client.get_technicals(a_ticker)
            if not tech:
                st.error("Could not fetch data. Check ticker.")
            else:
                news = data_client.get_news(a_ticker)
                earn = data_client.get_earnings_date(a_ticker)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${tech['Price']}")
                c2.metric("RSI (14)", tech['RSI'])
                c3.metric("Earnings Status", earn)
                
                st.markdown("### 📰 Recent News")
                st.text(news)
                
                st.markdown("### 🧠 AI Verdict")
                ai_res = agent.get_hunter_verdict(a_ticker, tech, news, earn)
                if ai_res.get('verdict') == "BUY": st.success(f"Score: {ai_res.get('score')} | Verdict: BUY")
                elif ai_res.get('verdict') == "WATCH": st.warning(f"Score: {ai_res.get('score')} | Verdict: WATCH")
                else: st.error(f"Score: {ai_res.get('score')} | Verdict: AVOID")
                st.write(ai_res.get('reasoning'))

with tab_journal:
    st.header("📓 Trade Journal")
    df_journal = get_journal_df()
    
    if not df_journal.empty:
        total_pnl = df_journal['pnl_abs'].sum()
        if total_pnl >= 0:
            st.success(f"**Total Realized PnL: €{total_pnl:,.2f}**")
        else:
            st.error(f"**Total Realized PnL: €{total_pnl:,.2f}**")
            
        st.dataframe(df_journal, use_container_width=True, hide_index=True)
    else:
        st.info("Journal is empty. Execute some trades to build history.")

# ==========================================
# TAB 5: BACKTESTER
# ==========================================
with tab_backtest:
    st.header("📈 Historical Regime Backtester")
    st.write("Test the exact mathematical edge of your **Risk-On / Risk-Off** moving average rule against pure Buy & Hold.")
    
    col1, col2, col3 = st.columns(3)
    bt_ticker = col1.text_input("Asset Ticker", "SPY").upper()
    bt_sma = col2.number_input("Regime Filter (SMA)", min_value=10, max_value=300, value=200, step=10)
    bt_years = col3.number_input("Years of History", min_value=1, max_value=10, value=5, step=1)
    
    if st.button("Run Institutional Backtest", type="primary"):
        with st.spinner(f"Simulating {bt_years} years of trading history for {bt_ticker}..."):
            try:
                api_key = getattr(data_client, 'api_key', None) 
                if not api_key: api_key = os.getenv("TIINGO_API_KEY")
                
                start_date = (datetime.today() - timedelta(days=bt_years * 365)).strftime('%Y-%m-%d')
                url = f"https://api.tiingo.com/tiingo/daily/{bt_ticker}/prices"
                params = {'startDate': start_date, 'token': api_key}
                
                res = requests.get(url, params=params)
                if res.status_code != 200 or len(res.json()) == 0:
                    st.error(f"Failed to fetch data for {bt_ticker}. Check ticker or API limits.")
                else:
                    df_bt = pd.DataFrame(res.json())
                    df_bt['date'] = pd.to_datetime(df_bt['date']).dt.tz_localize(None)
                    df_bt.set_index('date', inplace=True)
                    df_bt.sort_index(inplace=True)
                    
                    close_prices = df_bt['close']
                    
                    df_bt['Daily_Return'] = close_prices.pct_change()
                    df_bt['SMA'] = close_prices.rolling(window=bt_sma).mean()
                    
                    df_bt['Signal'] = np.where(close_prices > df_bt['SMA'], 1, 0)
                    df_bt['Strategy_Return'] = df_bt['Signal'].shift(1) * df_bt['Daily_Return']
                    
                    df_bt.dropna(inplace=True)
                    
                    df_bt['Cumulative_Buy_Hold'] = (1 + df_bt['Daily_Return']).cumprod()
                    df_bt['Cumulative_Strategy'] = (1 + df_bt['Strategy_Return']).cumprod()
                    
                    annual_trading_days = 252
                    
                    bh_cagr = (df_bt['Cumulative_Buy_Hold'].iloc[-1] ** (1 / bt_years)) - 1
                    bh_vol = df_bt['Daily_Return'].std() * np.sqrt(annual_trading_days)
                    bh_sharpe = bh_cagr / bh_vol if bh_vol != 0 else 0
                    bh_drawdown = (df_bt['Cumulative_Buy_Hold'] / df_bt['Cumulative_Buy_Hold'].cummax() - 1).min()
                    
                    strat_cagr = (df_bt['Cumulative_Strategy'].iloc[-1] ** (1 / bt_years)) - 1
                    strat_vol = df_bt['Strategy_Return'].std() * np.sqrt(annual_trading_days)
                    strat_sharpe = strat_cagr / strat_vol if strat_vol != 0 else 0
                    strat_drawdown = (df_bt['Cumulative_Strategy'] / df_bt['Cumulative_Strategy'].cummax() - 1).min()
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.markdown("### 📉 Pure Buy & Hold")
                        st.metric("Total CAGR", f"{bh_cagr:.2%}")
                        st.metric("Max Drawdown", f"{bh_drawdown:.2%}", delta_color="inverse")
                        st.metric("Sharpe Ratio (Risk-Adjusted)", f"{bh_sharpe:.2f}")
                        
                    with c2:
                        st.markdown("### 🛡️ Trend Regime Strategy")
                        st.metric("Total CAGR", f"{strat_cagr:.2%}", delta=f"{(strat_cagr - bh_cagr):.2%} vs B&H")
                        st.metric("Max Drawdown", f"{strat_drawdown:.2%}", delta=f"{(strat_drawdown - bh_drawdown):.2%} vs B&H", delta_color="inverse")
                        st.metric("Sharpe Ratio (Risk-Adjusted)", f"{strat_sharpe:.2f}", delta=f"{(strat_sharpe - bh_sharpe):.2f}")

                    st.markdown("---")
                    st.subheader("💰 Equity Curve Comparison")
                    chart_data = df_bt[['Cumulative_Buy_Hold', 'Cumulative_Strategy']].copy()
                    chart_data.columns = ['Buy & Hold', 'Trend Strategy']
                    st.line_chart(chart_data, color=["#ef4444", "#10b981"])
                    
                    st.caption("Notice how the green line (Strategy) flattens out during major market crashes because the system mathematically rotated to cash, preserving your capital.")

            except Exception as e:
                st.error(f"Backtest engine failed: {e}")
import streamlit as st
import pandas as pd
import concurrent.futures
import requests
import time
import sqlite3
from datetime import datetime
import os
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
tab_port, tab_radar, tab_analyze, tab_journal = st.tabs([
    "📂 Portfolio", "🎯 Radar Scan", "🔍 Deep Analyzer", "📓 Trade Journal"
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
            
            st.session_state.live_port_df = live_df
            st.session_state.current_fx_rate = usd_to_eur

    if st.session_state.live_port_df is not None:
        live_df = st.session_state.live_port_df
        cash = live_df[live_df['ticker'] == 'EUR']['Current Value (€)'].sum()
        inv_cost = live_df[live_df['ticker'] != 'EUR']['cost'].multiply(live_df[live_df['ticker'] != 'EUR']['quantity']).sum()
        inv_val = live_df[live_df['ticker'] != 'EUR']['Current Value (€)'].sum()
        pnl_eur = live_df['PnL (€)'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Live Total Equity", f"€{cash + inv_val:,.2f}", f"{pnl_eur:+,.2f} €")
        c2.metric("💵 Cash Available", f"€{cash:,.2f}")
        c3.metric("📈 Total Return", f"{(pnl_eur/inv_cost*100 if inv_cost>0 else 0):+.2f}%")
        c4.metric("📊 Live Invested Value", f"€{inv_val:,.2f}")
        
        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #10b981;' 
                elif val < 0: return 'color: #ef4444;' 
            return ''
            
        st.dataframe(live_df.style.format({'cost': '€{:.2f}', 'Live Price (€)': '€{:.2f}', 'PnL (%)': '{:.2f}%'}).map(color_pnl, subset=['PnL (%)']), use_container_width=True, hide_index=True)
    else:
        summary = pm.get_equity_summary()
        st.dataframe(df_port, use_container_width=True, hide_index=True)

    st.markdown("---")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        with st.expander("📥 Bulk Inject Existing Portfolio"):
            with st.form("bulk_inject_form"):
                bulk_data = st.text_area("Ticker, Quantity, Avg_Price")
                if st.form_submit_button("Force Inject"):
                    conn = sqlite3.connect("data/hedgefund.db")
                    cursor = conn.cursor()
                    for line in bulk_data.split('\n'):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            cursor.execute("INSERT INTO portfolio (ticker, cost, quantity, target, status, date_acquired) VALUES (?, ?, ?, 0.0, 'OPEN', ?)", (parts[0].upper(), float(parts[2]), float(parts[1]), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit(); conn.close(); st.rerun()

    if st.button("🛡️ Run AI Guardian Audit"):
        audit_df = st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port
        with st.spinner("Analyzing..."):
            results = []
            for _, row in audit_df[audit_df['ticker'] != 'EUR'].iterrows():
                v = agent.get_guardian_audit(row['ticker'], row.to_dict(), data_client.get_news(row['ticker']), data_client.get_earnings_date(row['ticker']))
                results.append({'Ticker': row['ticker'], 'Action': v.get('action'), 'AI Advice': v.get('reasoning')})
            st.session_state.guardian_audit_df = pd.DataFrame(results)

    if st.session_state.guardian_audit_df is not None:
        for _, row in st.session_state.guardian_audit_df.iterrows():
            with st.container(border=True):
                color = "red" if 'SELL' in row['Action'] else "orange" if 'TRIM' in row['Action'] else "green"
                st.markdown(f"### {row['Ticker']} | Action: :{color}[**{row['Action']}**]")
                st.write(row['AI Advice'])

# ==========================================
# TAB 2: RADAR SCAN
# ==========================================
with tab_radar:
    st.header("🎯 Two-Tier AI Radar Scan")
    
    with st.expander("🌍 Market Regime Check", expanded=True):
        regime = data_client.get_regime('SPY')
        if regime:
            if not regime['bullish']:
                st.error("🔴 RISK OFF: SPY below 200 SMA")
                override = st.checkbox("⚠️ Override Kill Switch")
            else:
                st.success("🟢 RISK ON: SPY above 200 SMA")
                override = True
        else:
            override = True

    if override:
        if st.button("🚀 Launch Scan", type="primary"):
            tickers = get_sp500_tickers()
            results = []
            p = st.progress(0)
            
            def fetch_quant(t):
                sec = data_client.get_sector_for_ticker(t)
                s_reg = data_client.get_regime(sec)
                met = data_client.get_smart_momentum(t)
                if not met: return None
                return {'Ticker': t, 'Sector': sec, 'Sector Health': s_reg['status'] if s_reg else "??", 'Smooth_Score': met['Smooth_Score']}

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
                futs = {ex.submit(fetch_quant, t): t for t in tickers}
                for i, f in enumerate(concurrent.futures.as_completed(futs)):
                    res = f.result()
                    if res: results.append(res)
                    p.progress((i+1)/len(tickers))
            
            if results:
                top_20 = pd.DataFrame(results).sort_values(by="Smooth_Score", ascending=False).head(20)
                st.subheader("🧠 AI Deep Dive")
                final = []
                for _, r in top_20.iterrows():
                    v = agent.get_hunter_verdict(r['Ticker'], r.to_dict(), data_client.get_news(r['Ticker']), data_client.get_earnings_date(r['Ticker']))
                    final.append({"Ticker": r['Ticker'], "Score": v.get('score'), "Verdict": v.get('verdict'), "Reason": v.get('reasoning')})
                    time.sleep(1)
                st.dataframe(pd.DataFrame(final).sort_values(by="Score", ascending=False), use_container_width=True)
    else:
        st.info("🛑 Scan disabled by Kill Switch.")

# ==========================================
# TAB 3 & 4
# ==========================================
with tab_analyze:
    st.header("🔍 Deep Stock Analyzer")
    ticker = st.text_input("Ticker", "NVDA").upper()
    if st.button("Analyze"):
        tech = data_client.get_technicals(ticker)
        if tech:
            st.metric("Price", f"${tech['Price']}")
            st.write(agent.get_hunter_verdict(ticker, tech, data_client.get_news(ticker), data_client.get_earnings_date(ticker))['reasoning'])

with tab_journal:
    st.header("📓 Trade Journal")
    st.dataframe(get_journal_df(), use_container_width=True)
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
    
    # Initialize session state to hold live market data & audit
    if "live_port_df" not in st.session_state:
        st.session_state.live_port_df = None
        st.session_state.current_fx_rate = 1.0
    if "guardian_audit_df" not in st.session_state:
        st.session_state.guardian_audit_df = None

    # --- REFRESH BUTTON & LOGIC ---
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        refresh_clicked = st.button("🔄 Refresh Live Prices & PnL", type="primary", use_container_width=True)

    if refresh_clicked and not df_port.empty:
        with st.spinner("Fetching live market data and FX rates from Tiingo..."):
            try:
                api_key = getattr(data_client, 'api_key', None) 
                if not api_key:
                    api_key = os.getenv("TIINGO_API_KEY")

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
                if ticker == 'EUR':
                    return {'ticker': 'EUR', 'live_price': 1.0}
                try:
                    tech = data_client.get_technicals(ticker)
                    if tech and 'Price' in tech:
                        eur_price = tech['Price'] * usd_to_eur 
                        return {'ticker': ticker, 'live_price': eur_price}
                    return {'ticker': ticker, 'live_price': row['cost']}
                except:
                    return {'ticker': ticker, 'live_price': row['cost']}
            
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
        cash = live_df[live_df['ticker'] == 'EUR']['Current Value (€)'].sum() if 'EUR' in live_df['ticker'].values else 0.0
        invested_cost = live_df[live_df['ticker'] != 'EUR']['cost'].multiply(live_df[live_df['ticker'] != 'EUR']['quantity']).sum()
        live_invested_val = live_df[live_df['ticker'] != 'EUR']['Current Value (€)'].sum()
        
        total_pnl_eur = live_df['PnL (€)'].sum()
        live_total_equity = cash + live_invested_val
        total_pnl_pct = (total_pnl_eur / invested_cost * 100) if invested_cost > 0 else 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Live Total Equity", f"€{live_total_equity:,.2f}", f"{total_pnl_eur:+,.2f} €")
        col2.metric("💵 Cash Available", f"€{cash:,.2f}")
        col3.metric("📈 Total Return", f"{total_pnl_pct:+.2f}%", f"{total_pnl_eur:+,.2f} €")
        col4.metric("📊 Live Invested Value", f"€{live_invested_val:,.2f}")
        
        st.caption(f"🌍 *Live USD to EUR Conversion Rate:* **{st.session_state.current_fx_rate:.4f}**")
        
        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #10b981;' 
                elif val < 0: return 'color: #ef4444;' 
            return ''
            
        styled_df = live_df.style.format({
            'cost': '€{:.2f}', 'Live Price (€)': '€{:.2f}', 'Current Value (€)': '€{:.2f}', 
            'PnL (€)': '€{:.2f}', 'PnL (%)': '{:.2f}%'
        }).map(color_pnl, subset=['PnL (€)', 'PnL (%)'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        summary = pm.get_equity_summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Equity (Cost Basis)", f"€{summary['total_equity']:,.2f}")
        col2.metric("💵 Cash Available", f"€{summary['cash']:,.2f}")
        col3.metric("📈 Invested Capital", f"€{summary['invested']:,.2f}")
        if not df_port.empty:
            st.dataframe(df_port, use_container_width=True, hide_index=True)
        else:
            st.info("Portfolio is empty.")

    st.markdown("---")
    st.subheader("🛠️ Management & Portfolio Injection")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        with st.expander("➕ Add Single Position / Deposit Cash"):
            with st.form("buy_form"):
                b_ticker = st.text_input("Ticker", "AAPL").upper()
                b_price = st.number_input("Entry Price", min_value=0.0, value=150.0)
                b_qty = st.number_input("Quantity", min_value=1, value=10)
                if st.form_submit_button("Execute"):
                    if pm.execute_buy(b_ticker, b_price, b_qty, 0.0):
                        st.success("Success!"); st.rerun()

        with st.expander("📥 Bulk Inject Existing Portfolio"):
            with st.form("bulk_inject_form"):
                bulk_data = st.text_area("Ticker, Quantity, Avg_Price")
                if st.form_submit_button("Force Inject"):
                    try:
                        conn = sqlite3.connect("data/hedgefund.db")
                        cursor = conn.cursor()
                        for line in bulk_data.split('\n'):
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                cursor.execute("INSERT INTO portfolio (ticker, cost, quantity, target, status, date_acquired) VALUES (?, ?, ?, 0.0, 'OPEN', ?)", 
                                               (parts[0].upper(), float(parts[2]), float(parts[1]), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        conn.commit(); conn.close()
                        st.success("Injected!"); st.rerun()
                    except Exception as e: st.error(f"Failed: {e}")

    with m_col2:
        with st.expander("✂️ Sell / Trim Position"):
            with st.form("sell_form"):
                s_ticker = st.text_input("Ticker").upper()
                s_price = st.number_input("Exit Price", min_value=0.0)
                s_qty = st.number_input("Quantity", min_value=1)
                if st.form_submit_button("Execute Sell"):
                    if pm.execute_sell(s_ticker, s_price, s_qty):
                        st.success("Sold!"); st.rerun()

    if st.button("🛡️ Run AI Guardian Audit on Portfolio"):
        audit_df = st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port
        if audit_df.empty or len(audit_df[audit_df['ticker'] != 'EUR']) == 0:
            st.warning("No active stocks to audit.")
        else:
            with st.spinner("Analyzing..."):
                audit_results = []
                for _, row in audit_df[audit_df['ticker'] != 'EUR'].iterrows():
                    ticker = row['ticker']
                    verdict = agent.get_guardian_audit(ticker, row.to_dict(), data_client.get_news(ticker), data_client.get_earnings_date(ticker))
                    audit_results.append({
                        'Ticker': ticker, 'Action': verdict.get('action', 'N/A'),
                        'Earnings Risk': verdict.get('earnings_risk', 'Unknown'),
                        'AI Advice': verdict.get('reasoning', ''),
                        'Execution Plan': verdict.get('proposed_stop', '')
                    })
                st.session_state.guardian_audit_df = pd.DataFrame(audit_results)

    if st.session_state.guardian_audit_df is not None:
        for _, row in st.session_state.guardian_audit_df.iterrows():
            with st.container(border=True):
                action = str(row['Action']).upper()
                color = "red" if 'SELL' in action else "orange" if 'TRIM' in action else "green"
                st.markdown(f"### {row['Ticker']} | Action: :{color}[**{action}**]")
                st.write(f"**Advice:** {row['AI Advice']}")

        export_df = pd.merge(st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port, 
                             st.session_state.guardian_audit_df, left_on='ticker', right_on='Ticker', how='left')
        st.download_button("📥 Download Audit Report", export_df.to_csv(index=False).encode('utf-8'), "audit.csv", "text/csv")

# ==========================================
# TAB 2: RADAR SCAN
# ==========================================
with tab_radar:
    st.header("🎯 Two-Tier AI Radar Scan")
    
    # --- MARKET REGIME KILL SWITCH ---
    with st.expander("🌍 Broad Market Regime Check", expanded=True):
        with st.spinner("Checking market trend..."):
            # Ensure the method name matches what's in your data_client (e.g., get_regime)
            regime = data_client.get_regime('SPY')
            
        if regime:
            if not regime['bullish']:
                st.error(f"**🔴 RISK OFF**")
                st.warning(f"SPY is below its 200-day SMA. Use caution.")
                override_killswitch = st.checkbox("⚠️ Override Kill Switch")
            else:
                st.success(f"**🟢 RISK ON**")
                override_killswitch = True
        else:
            st.warning("Regime check failed.")
            override_killswitch = True
    
    tickers_to_scan = get_sp500_tickers() # Assuming this helper exists or is cached above

    if not override_killswitch:
        st.info("🛑 Radar Scan disabled by Kill Switch.")
    else:
        # ALL SCAN LOGIC MUST BE INDENTED HERE
        if st.button("🚀 Launch Two-Tier Scan", type="primary"):
            st.subheader("⚙️ Phase 1: Quantitative Filter")
            quant_results = []
            progress_bar = st.progress(0)
            
            def fetch_quant(t):
                sector_etf = data_client.get_sector_for_ticker(t)
                sector_regime = data_client.get_regime(sector_etf)
                metrics = data_client.get_smart_momentum(t)
                if not metrics: return None
                return {
                    'Ticker': t, 'Sector': sector_etf,
                    'Sector Health': sector_regime['status'] if sector_regime else "Unknown",
                    'Price': metrics['Current_Price'], 'Smooth_Score': metrics['Smooth_Score']
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_quant, t): t for t in tickers_to_scan}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    res = future.result()
                    if res: quant_results.append(res)
                    progress_bar.progress((i + 1) / len(tickers_to_scan))
            
            if quant_results:
                df_quant = pd.DataFrame(quant_results).sort_values(by="Smooth_Score", ascending=False)
                top_20 = df_quant.head(20)
                st.dataframe(top_20, use_container_width=True)

                st.subheader("🧠 Phase 2: AI Deep Dive")
                final_results = []
                for i, t in enumerate(top_20['Ticker']):
                    ai_res = agent.get_hunter_verdict(t, top_20.iloc[i].to_dict(), data_client.get_news(t), data_client.get_earnings_date(t))
                    final_results.append({
                        "Ticker": t, "Smooth Score": top_20.iloc[i]['Smooth_Score'],
                        "AI Score": ai_res.get('score', 0), "Verdict": ai_res.get('verdict', 'ERROR'),
                        "Reasoning": ai_res.get('reasoning', '')
                    })
                    time.sleep(1) # Rate limit protection

                df_final = pd.DataFrame(final_results).sort_values(by="AI Score", ascending=False)
                st.dataframe(df_final, use_container_width=True)
                st.download_button("📥 Download Scan Report", df_final.to_csv(index=False).encode('utf-8'), "scan.csv", "text/csv")

# ==========================================
# TAB 3 & 4 (Analysis & Journal - Keep existing logic)
# ==========================================
with tab_analyze:
    st.header("🔍 Deep Stock Analyzer")
    # ... existing logic ...

with tab_journal:
    st.header("📓 Trade Journal")
    # ... existing logic ...
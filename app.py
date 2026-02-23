import streamlit as st
import pandas as pd
from src.data_client import MarketDataClient
from src.database import get_portfolio_df, get_journal_df
from src.agent import AIAgent
from src.portfolio import PortfolioManager

# --- INIT ---
st.set_page_config(page_title="🦅 AI Hedge Fund Pro", layout="wide", page_icon="📈")

@st.cache_resource
def get_clients():
    return MarketDataClient(), AIAgent(), PortfolioManager()

data_client, agent, pm = get_clients()

# --- HEADER ---
st.title("🦅 AI Hedge Fund Manager (Pro V2)")
st.markdown("Powered by Tiingo, Gemini 2.5 Flash, and SQLite.")

# --- TABS ---
tab_port, tab_radar, tab_analyze, tab_journal = st.tabs([
    "📂 Portfolio", "🎯 Radar Scan", "🔍 Deep Analyzer", "📓 Trade Journal"
])

# ==========================================
# TAB 1: PORTFOLIO & GUARDIAN
# ==========================================
with tab_port:
    st.header("Portfolio Overview")
    
    # 1. Equity Summary Cards
    summary = pm.get_equity_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Equity", f"€{summary['total_equity']:,.2f}")
    col2.metric("💵 Cash Available", f"€{summary['cash']:,.2f}")
    col3.metric("📈 Invested Capital", f"€{summary['invested']:,.2f}")
    
    # 2. Holdings Table
    df_port = get_portfolio_df()
    if not df_port.empty:
        st.dataframe(df_port, use_container_width=True, hide_index=True)
    else:
        st.info("Portfolio is empty. Add cash or buy positions to get started.")

    # 3. Actions & AI Audit
    st.markdown("---")
    st.subheader("🛠️ Management & AI Guardian")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        with st.expander("➕ Add Position / Deposit Cash"):
            with st.form("buy_form"):
                b_ticker = st.text_input("Ticker (Use 'EUR' for Cash)", "AAPL").upper()
                b_price = st.number_input("Entry Price (1.0 for EUR)", min_value=0.0, value=150.0, format="%.2f")
                b_qty = st.number_input("Quantity", min_value=1, value=10)
                b_target = st.number_input("Target Price", min_value=0.0, value=0.0)
                b_submit = st.form_submit_button("Execute Buy / Deposit")
                if b_submit:
                    if pm.execute_buy(b_ticker, b_price, b_qty, b_target):
                        st.success(f"Successfully bought {b_qty} of {b_ticker}!")
                        st.rerun()
                    else:
                        st.error("Failed. Check cash balance or inputs.")
                        
    with m_col2:
        with st.expander("✂️ Sell / Trim Position"):
            with st.form("sell_form"):
                s_ticker = st.text_input("Ticker to Sell").upper()
                s_price = st.number_input("Exit Price", min_value=0.0, value=150.0, format="%.2f")
                s_qty = st.number_input("Quantity to Sell", min_value=1, value=10)
                s_submit = st.form_submit_button("Execute Sell / Trim")
                if s_submit:
                    if pm.execute_sell(s_ticker, s_price, s_qty):
                        st.success(f"Successfully sold {s_qty} of {s_ticker}!")
                        st.rerun()
                    else:
                        st.error("Failed. Ensure you own the stock.")

    # 4. AI Guardian
    if st.button("🛡️ Run AI Guardian Audit on Portfolio"):
        if df_port.empty or len(df_port[df_port['ticker'] != 'EUR']) == 0:
            st.warning("No active stocks to audit.")
        else:
            with st.spinner("Guardian is analyzing your holdings..."):
                for _, row in df_port[df_port['ticker'] != 'EUR'].iterrows():
                    ticker = row['ticker']
                    pos_data = row.to_dict()
                    news = data_client.get_news(ticker)
                    earnings = data_client.get_earnings_date(ticker)
                    
                    verdict = agent.get_guardian_audit(ticker, pos_data, news, earnings)
                    
                    # Display as a clean card
                    with st.container(border=True):
                        st.markdown(f"### {ticker}  |  **Action:** `{verdict.get('action', 'N/A')}`")
                        st.write(f"**Earnings Risk:** {verdict.get('earnings_risk', 'Unknown')}")
                        st.write(f"**Advice:** {verdict.get('reasoning', '')}")
                        st.write(f"**Plan:** {verdict.get('proposed_stop', '')}")

# ==========================================
# TAB 2: RADAR SCAN
# ==========================================
with tab_radar:
    st.header("🎯 AI Radar Scan")
    st.write("Scan a list of tickers to find new opportunities.")
    
    scan_tickers = st.text_input("Enter tickers (comma separated)", "AAPL, MSFT, NVDA, TSLA")
    
    if st.button("Launch Scan", type="primary"):
        tickers = [t.strip().upper() for t in scan_tickers.split(",")]
        results = []
        
        progress_bar = st.progress(0)
        for i, t in enumerate(tickers):
            with st.spinner(f"Analyzing {t}..."):
                tech = data_client.get_technicals(t)
                if tech:
                    news = data_client.get_news(t)
                    earn = data_client.get_earnings_date(t)
                    ai_res = agent.get_hunter_verdict(t, tech, news, earn)
                    
                    results.append({
                        "Ticker": t,
                        "Price": tech['Price'],
                        "RSI": tech['RSI'],
                        "Earnings": earn,
                        "AI Score": ai_res.get('score', 0),
                        "Verdict": ai_res.get('verdict', 'N/A'),
                        "Reasoning": ai_res.get('reasoning', '')
                    })
            progress_bar.progress((i + 1) / len(tickers))
            
        if results:
            df_res = pd.DataFrame(results).sort_values(by="AI Score", ascending=False)
            st.dataframe(df_res, use_container_width=True, hide_index=True)

# ==========================================
# TAB 3: DEEP ANALYZER
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

# ==========================================
# TAB 4: TRADE JOURNAL
# ==========================================
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
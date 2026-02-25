import streamlit as st
import pandas as pd
import concurrent.futures
import requests
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
# TAB 2: RADAR SCAN (TWO-TIER PIPELINE)
# ==========================================
with tab_radar:
    st.header("🎯 Two-Tier AI Radar Scan")
    st.write("Phase 1: Quantitative technical filter. Phase 2: AI fundamental deep dive on the Top 20.")
    
    # Helper: Fetch S&P 500 list from Wikipedia
    @st.cache_data(ttl=86400) # Cache for 24 hours so we don't spam Wikipedia
    def get_sp500_tickers():
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            # Replace dots with dashes for yfinance/Tiingo compatibility (e.g., BRK.B -> BRK-B)
            return df['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            st.error(f"Failed to fetch S&P 500: {e}. Using default tech list.")
            return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]

    # Choose Universe
    universe_choice = st.radio("Select Universe:", ["Custom List", "S&P 500 (Full Scan)"], horizontal=True)
    if universe_choice == "Custom List":
        scan_tickers_input = st.text_input("Enter tickers (comma separated)", "AAPL, MSFT, NVDA, TSLA, AMD, INTC")
        tickers_to_scan = [t.strip().upper() for t in scan_tickers_input.split(",")]
    else:
        tickers_to_scan = get_sp500_tickers()
        st.info(f"Loaded {len(tickers_to_scan)} tickers from S&P 500.")

    if st.button("🚀 Launch Two-Tier Scan", type="primary"):
        
        # --- PHASE 1: THE QUANT FILTER ---
        st.subheader("⚙️ Phase 1: Quantitative Filter")
        quant_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def fetch_quant(t):
            tech = data_client.get_technicals(t)
            if not tech: return None
            
            # Mechanical Quant Scoring Engine (Max 100)
            score = 0
            if tech['Price'] > tech['SMA_50']: score += 30      # Short-term trend up
            if tech['SMA_50'] > tech['SMA_200']: score += 30    # Long-term trend up
            if 40 <= tech['RSI'] <= 70: score += 40             # Healthy momentum (not overbought/oversold)
            elif tech['RSI'] < 40: score += 20                  # Value territory
            
            tech['Quant_Score'] = score
            return tech

        # Use Threading to fetch technicals at lightning speed
        status_text.text(f"Fetching technicals for {len(tickers_to_scan)} stocks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_quant, t): t for t in tickers_to_scan}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: quant_results.append(res)
                progress_bar.progress((i + 1) / len(tickers_to_scan))
                
        if not quant_results:
            st.error("Phase 1 failed. Check data connection.")
            st.stop()
            
        # Sort and take Top 20
        df_quant = pd.DataFrame(quant_results).sort_values(by="Quant_Score", ascending=False)
        top_20_df = df_quant.head(20)
        
        st.success(f"Phase 1 Complete! Filtered down to top {len(top_20_df)} candidates.")
        with st.expander("View Phase 1 Raw Data"):
            st.dataframe(top_20_df, use_container_width=True)

        # --- PHASE 2: THE AI HUNTER ---
        st.subheader("🧠 Phase 2: AI Deep Dive (Top 20)")
        final_results = []
        
        ai_progress = st.progress(0)
        ai_status = st.empty()
        
        top_tickers = top_20_df['Ticker'].tolist()
        
        for i, t in enumerate(top_tickers):
            ai_status.text(f"AI analyzing {t} ({i+1}/{len(top_tickers)})...")
            
            # Fetch Qualitative Data
            news = data_client.get_news(t)
            earn = data_client.get_earnings_date(t)
            
            # Retrieve the quant data we already pulled
            tech_data = top_20_df[top_20_df['Ticker'] == t].iloc[0].to_dict()
            
            # Ask Gemini
            ai_res = agent.get_hunter_verdict(t, tech_data, news, earn)
            
            final_results.append({
                "Ticker": t,
                "Price": tech_data['Price'],
                "Quant Score": tech_data['Quant_Score'],
                "AI Score": ai_res.get('score', 0),
                "Verdict": ai_res.get('verdict', 'ERROR'),
                "Earnings": earn,
                "Reasoning": ai_res.get('reasoning', '')
            })
            
            ai_progress.progress((i + 1) / len(top_tickers))
            
            # Pace the AI calls to respect Google's free tier (15 Requests Per Minute)
            time.sleep(4) 
            
        ai_status.text("Scan Complete!")
        
        # Display Final Results
        if final_results:
            df_final = pd.DataFrame(final_results).sort_values(by="AI Score", ascending=False)
            
            # Highlight the verdicts
            def highlight_verdict(val):
                if val == 'BUY': return 'background-color: #064e3b; color: white;' # Dark Green
                elif val == 'WATCH': return 'background-color: #78350f; color: white;' # Dark Orange
                elif val == 'AVOID': return 'background-color: #7f1d1d; color: white;' # Dark Red
                return ''
                
            styled_df = df_final.style.map(highlight_verdict, subset=['Verdict'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

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
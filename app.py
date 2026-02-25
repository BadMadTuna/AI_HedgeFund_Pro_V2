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
    
    # Initialize session state to hold live market data & audit so they don't disappear on refresh
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
            
            # 1. Fetch live USD to EUR exchange rate from Tiingo
            try:
                api_key = getattr(data_client, 'api_key', None) 
                if not api_key:
                    api_key = os.getenv("TIINGO_API_KEY")

                fx_url = "https://api.tiingo.com/tiingo/fx/top"
                fx_res = requests.get(fx_url, params={'tickers': 'eurusd', 'token': api_key})
                fx_res.raise_for_status()
                fx_data = fx_res.json()
                
                # eurusd gives us how many USD 1 EUR buys. To get USD to EUR multiplier, we invert it.
                eur_usd_rate = fx_data[0]['midPrice']
                usd_to_eur = 1.0 / eur_usd_rate
            except Exception as e:
                st.warning(f"Could not fetch live FX rate ({e}). Defaulting to 0.92.")
                usd_to_eur = 0.92

            # 2. Threading helper to fetch prices and INSTANTLY convert to Euros
            def fetch_live_price(row):
                ticker = row['ticker']
                if ticker == 'EUR':
                    return {'ticker': 'EUR', 'live_price': 1.0}
                try:
                    tech = data_client.get_technicals(ticker)
                    if tech and 'Price' in tech:
                        # Convert USD price to EUR
                        eur_price = tech['Price'] * usd_to_eur 
                        return {'ticker': ticker, 'live_price': eur_price}
                    return {'ticker': ticker, 'live_price': row['cost']}
                except:
                    return {'ticker': ticker, 'live_price': row['cost']}
            
            # Execute concurrent fetches
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_live_price, row): row for _, row in df_port.iterrows()}
                live_prices = {fut.result()['ticker']: fut.result()['live_price'] for fut in concurrent.futures.as_completed(futures)}
            
            # Build the Enriched DataFrame
            live_df = df_port.copy()
            live_df['Live Price (€)'] = live_df['ticker'].map(live_prices)
            live_df['Current Value (€)'] = live_df['Live Price (€)'] * live_df['quantity']
            
            # Calculate PnL (Ignoring Cash)
            is_stock = live_df['ticker'] != 'EUR'
            live_df['PnL (€)'] = 0.0
            live_df['PnL (%)'] = 0.0
            
            live_df.loc[is_stock, 'PnL (€)'] = (live_df['Live Price (€)'] - live_df['cost']) * live_df['quantity']
            live_df.loc[is_stock, 'PnL (%)'] = ((live_df['Live Price (€)'] - live_df['cost']) / live_df['cost']) * 100
            
            # Save it to session state
            st.session_state.live_port_df = live_df
            st.session_state.current_fx_rate = usd_to_eur

    # --- DISPLAY METRICS & TABLE ---
    if st.session_state.live_port_df is not None:
        # 1. DISPLAY LIVE DATA
        live_df = st.session_state.live_port_df
        
        cash = live_df[live_df['ticker'] == 'EUR']['Current Value (€)'].sum() if 'EUR' in live_df['ticker'].values else 0.0
        invested_cost = live_df[live_df['ticker'] != 'EUR']['cost'].multiply(live_df[live_df['ticker'] != 'EUR']['quantity']).sum()
        live_invested_val = live_df[live_df['ticker'] != 'EUR']['Current Value (€)'].sum()
        
        total_pnl_eur = live_df['PnL (€)'].sum()
        live_total_equity = cash + live_invested_val
        total_pnl_pct = (total_pnl_eur / invested_cost * 100) if invested_cost > 0 else 0.0
        
        # Enriched Live Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Live Total Equity", f"€{live_total_equity:,.2f}", f"{total_pnl_eur:+,.2f} €")
        col2.metric("💵 Cash Available", f"€{cash:,.2f}")
        col3.metric("📈 Total Return", f"{total_pnl_pct:+.2f}%", f"{total_pnl_eur:+,.2f} €")
        col4.metric("📊 Live Invested Value", f"€{live_invested_val:,.2f}")
        
        st.caption(f"🌍 *Live USD to EUR Conversion Rate:* **{st.session_state.current_fx_rate:.4f}**")
        
        # Enriched Live Holdings Table with Color Coding
        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #10b981;' # Green
                elif val < 0: return 'color: #ef4444;' # Red
            return ''
            
        styled_df = live_df.style.format({
            'cost': '€{:.2f}', 'Live Price (€)': '€{:.2f}', 'Current Value (€)': '€{:.2f}', 
            'PnL (€)': '€{:.2f}', 'PnL (%)': '{:.2f}%'
        }).map(color_pnl, subset=['PnL (€)', 'PnL (%)'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    else:
        # 2. DISPLAY FALLBACK DB DATA (Before hitting Refresh)
        summary = pm.get_equity_summary()
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Total Equity (Cost Basis)", f"€{summary['total_equity']:,.2f}")
        col2.metric("💵 Cash Available", f"€{summary['cash']:,.2f}")
        col3.metric("📈 Invested Capital", f"€{summary['invested']:,.2f}")
        
        if not df_port.empty:
            st.dataframe(df_port, use_container_width=True, hide_index=True)
        else:
            st.info("Portfolio is empty. Add cash or bulk inject positions to get started.")

    # 3. Actions & Portfolio Injection
    st.markdown("---")
    st.subheader("🛠️ Management & Portfolio Injection")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        with st.expander("➕ Add Single Position / Deposit Cash"):
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
                        
        # Bulk Portfolio Injection UI (Direct Database Bypass)
        with st.expander("📥 Bulk Inject Existing Portfolio"):
            st.write("Easily onboard your existing stocks without deducting cash. **Format:** `TICKER, QUANTITY, AVG_PRICE`")
            st.write("*(Use 'EUR' to set your initial cash balance)*")
            with st.form("bulk_inject_form"):
                bulk_data = st.text_area(
                    "Paste your portfolio data here:", 
                    "EUR, 50000, 1.0\nAAPL, 15, 175.50\nMSFT, 10, 400.00"
                )
                if st.form_submit_button("Force Inject Portfolio"):
                    try:
                        conn = sqlite3.connect("data/hedgefund.db")
                        cursor = conn.cursor()
                        success_count = 0
                        
                        for line in bulk_data.split('\n'):
                            line = line.strip()
                            if line:
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
                                        cursor.execute("INSERT INTO portfolio (ticker, cost, quantity, target, status, date_acquired) VALUES (?, ?, ?, ?, 'OPEN', ?)", 
                                                       (ticker, price, qty, 0.0, date_str))
                                    success_count += 1
                                    
                        conn.commit()
                        conn.close()
                        st.success(f"Successfully force-injected {success_count} positions directly into the database!")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Injection failed: {e}")
                        
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
    # 4. AI Guardian
    if st.button("🛡️ Run AI Guardian Audit on Portfolio"):
        # NEW: Use the live data if it exists, otherwise fall back to the DB
        audit_df = st.session_state.live_port_df if st.session_state.live_port_df is not None else df_port
        
        if audit_df.empty or len(audit_df[audit_df['ticker'] != 'EUR']) == 0:
            st.warning("No active stocks to audit. (Tip: Click 'Refresh Live Prices' first!)")
        else:
            with st.spinner("Guardian is analyzing your live holdings..."):
                audit_results = []
                for _, row in audit_df[audit_df['ticker'] != 'EUR'].iterrows(): # <--- The Fix
                    ticker = row['ticker']
                    pos_data = row.to_dict()
                    news = data_client.get_news(ticker)
                    earnings = data_client.get_earnings_date(ticker)
                    
                    verdict = agent.get_guardian_audit(ticker, pos_data, news, earnings)
                    
                    # Store results for the export file
                    audit_results.append({
                        'Ticker': ticker,
                        'Action': verdict.get('action', 'N/A'),
                        'Earnings Risk': verdict.get('earnings_risk', 'Unknown'),
                        'AI Advice': verdict.get('reasoning', ''),
                        'Execution Plan': verdict.get('proposed_stop', '')
                    })
                
                # Save the audit to session state so the download button doesn't disappear
                st.session_state.guardian_audit_df = pd.DataFrame(audit_results)

    # If an audit has been run, display the UI cards AND the download button
    if st.session_state.guardian_audit_df is not None:
        st.success("Audit Complete!")
        
        # Display the visual cards
        # Display the visual cards with Color Coding
        for _, row in st.session_state.guardian_audit_df.iterrows():
            with st.container(border=True):
                action = str(row['Action']).upper()
                
                # Apply Streamlit's native markdown color tags
                if 'SELL' in action:
                    action_colored = f":red[**{action}**]"
                elif 'TRIM' in action:
                    action_colored = f":orange[**{action}**]"
                elif 'KEEP' in action:
                    action_colored = f":green[**{action}**]"
                else:
                    action_colored = f"**{action}**"

                st.markdown(f"### {row['Ticker']}  |  Action: {action_colored}")
                
                # If Earnings Risk is elevated, color that too!
                risk = str(row['Earnings Risk'])
                if 'Critical' in risk or 'Elevated' in risk:
                    st.write(f"**Earnings Risk:** :red[{risk}]")
                else:
                    st.write(f"**Earnings Risk:** {risk}")
                    
                st.write(f"**Advice:** {row['AI Advice']}")
                st.write(f"**Plan:** {row['Execution Plan']}")

        # Merge Audit with Live Portfolio Data for the ultimate export
        if st.session_state.live_port_df is not None:
            export_df = pd.merge(st.session_state.live_port_df, st.session_state.guardian_audit_df, left_on='ticker', right_on='Ticker', how='left')
        else:
            export_df = pd.merge(df_port, st.session_state.guardian_audit_df, left_on='ticker', right_on='Ticker', how='left')
            
        # Create the Download Button
        csv_port = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Portfolio & AI Audit Report (CSV)",
            data=csv_port,
            file_name=f"portfolio_audit_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            type="primary"
        )


# ==========================================
# TAB 2: RADAR SCAN (TWO-TIER PIPELINE)
# ==========================================
with tab_radar:
    st.header("🎯 Two-Tier AI Radar Scan")
    st.write("Phase 1: Quantitative technical filter. Phase 2: AI fundamental deep dive on the Top 20.")

    # --- MARKET REGIME KILL SWITCH ---
    with st.expander("🌍 Broad Market Regime Check", expanded=True):
        with st.spinner("Checking S&P 500 trend..."):
            regime = data_client.get_market_regime()
            
        if regime['Status'] != 'Unknown':
            if regime['Color'] == 'red':
                st.error(f"**{regime['Status']}**")
                st.warning(f"The S&P 500 (SPY) is currently trading at **${regime['SPY_Price']}**, which is BELOW its 200-day moving average of **${regime['SPY_200_SMA']}**.\n\n"
                           f"**Institutional Rule:** Do not deploy new capital into long positions during a structural downtrend. Capital preservation is priority #1.")
                
                # The Override Checkbox
                override_killswitch = st.checkbox("⚠️ I understand the risks. Override the Kill Switch and allow scanning.")
            else:
                st.success(f"**{regime['Status']}**")
                st.write(f"The S&P 500 (SPY) is trading at **${regime['SPY_Price']}**, ABOVE its 200-day moving average of **${regime['SPY_200_SMA']}**. Conditions are favorable for long positions.")
                override_killswitch = True # Safe to scan
        else:
            st.warning("Could not determine market regime.")
            override_killswitch = True
    
    # Helper: Fetch S&P 500 list from Wikipedia
    @st.cache_data(ttl=86400) # Cache for 24 hours so we don't spam Wikipedia
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
            return df['Symbol'].str.replace('.', '-').tolist()
        except Exception as e:
            st.error(f"Failed to fetch S&P 500: {e}. Using default tech list.")
            return ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]

    universe_choice = st.radio("Select Universe:", ["Custom List", "S&P 500 (Full Scan)"], index=1, horizontal=True)
    if universe_choice == "Custom List":
        scan_tickers_input = st.text_input("Enter tickers (comma separated)", "AAPL, MSFT, NVDA, TSLA, AMD, INTC")
        tickers_to_scan = [t.strip().upper() for t in scan_tickers_input.split(",")]
    else:
        tickers_to_scan = get_sp500_tickers()
        st.info(f"Loaded {len(tickers_to_scan)} tickers from S&P 500.")

    if not override_killswitch:
        st.info("🛑 The Radar Scan is currently disabled by the Market Regime Kill Switch.")
    else:
        if st.button("🚀 Launch Two-Tier Scan", type="primary"):
        
        # --- PHASE 1: THE QUANT FILTER ---
        st.subheader("⚙️ Phase 1: Quantitative Filter")
        quant_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def fetch_quant(t):
            # 1. Get Sector for this ticker
            sector_etf = data_client.get_sector_for_ticker(t)
            
            # 2. Check the Regime for that specific Sector
            # (Note: In a production app, you'd cache this so you don't call Tiingo 500 times)
            sector_regime = data_client.get_regime(sector_etf)
            
            # 3. Only proceed if Sector is RISK ON (Optional: you can just flag it instead)
            metrics = data_client.get_smart_momentum(t)
            if not metrics: return None
            
            # Add sector health to the results
            result = {
                'Ticker': t,
                'Sector': sector_etf,
                'Sector Health': sector_regime['status'] if sector_regime else "Unknown",
                'Price': metrics['Current_Price'],
                'Smooth_Score': metrics['Smooth_Score'],
                # ... existing technicals ...
            }
            return result

        status_text.text(f"Calculating Volatility-Adjusted Momentum for {len(tickers_to_scan)} stocks...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_quant, t): t for t in tickers_to_scan}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: quant_results.append(res)
                progress_bar.progress((i + 1) / len(tickers_to_scan))
                
        if not quant_results:
            st.error("Phase 1 failed. Check data connection.")
            st.stop()
            
        df_quant = pd.DataFrame(quant_results).sort_values(by="Smooth_Score", ascending=False)
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
            
            news = data_client.get_news(t)
            earn = data_client.get_earnings_date(t)
            tech_data = top_20_df[top_20_df['Ticker'] == t].iloc[0].to_dict()
            
            ai_res = agent.get_hunter_verdict(t, tech_data, news, earn)
            
            final_results.append({
                "Ticker": t,
                "Price": tech_data['Price'],
                "Smooth Score": tech_data['Smooth_Score'],
                "AI Score": ai_res.get('score', 0),
                "Verdict": ai_res.get('verdict', 'ERROR'),
                "Earnings": earn,
                "Reasoning": ai_res.get('reasoning', '')
            })
            
            ai_progress.progress((i + 1) / len(top_tickers))
            time.sleep(4) 
            
        ai_status.text("Scan Complete!")
        
        # Display Final Results
        if final_results:
            df_final = pd.DataFrame(final_results).sort_values(by="AI Score", ascending=False)
            
            def highlight_verdict(val):
                if val == 'BUY': return 'background-color: #064e3b; color: white;' 
                elif val == 'WATCH': return 'background-color: #78350f; color: white;' 
                elif val == 'AVOID': return 'background-color: #7f1d1d; color: white;' 
                return ''
                
            styled_df = df_final.style.map(highlight_verdict, subset=['Verdict'])
            
            st.dataframe(
                styled_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Reasoning": st.column_config.TextColumn("Reasoning", width="large")
                }
            )

            st.divider()
            st.subheader("📝 Detailed AI Reasoning & Trade Sizing")
            st.write("Expand to read the full fundamental analysis and exact trade execution specs.")
            
            # dynamically fetch real total equity from your DB
            equity_summary = pm.get_equity_summary()
            ACCOUNT_SIZE = equity_summary.get('total_equity', 100000)
            
            for _, row in df_final.iterrows():
                if row['Verdict'] in ['BUY', 'WATCH']: 
                    with st.expander(f"{row['Verdict']} | {row['Ticker']} (AI Score: {row['AI Score']})"):
                        st.write(f"**Quant Smoothness Score:** {row['Smooth Score']}")
                        
                        if row['Verdict'] == 'BUY':
                            sizing = data_client.get_atr_and_sizing(row['Ticker'], account_value=ACCOUNT_SIZE, risk_pct=0.01)
                            if sizing:
                                st.success(f"**Execution Plan (1% Risk on ${ACCOUNT_SIZE:,.2f} Total Equity):**\n"
                                           f"- Buy **{sizing['Shares']} shares** at approx **${sizing['Current_Price']}**\n"
                                           f"- Total Capital Deployed: **${sizing['Total_Investment']:,}**\n"
                                           f"- Hard Stop Loss: **${sizing['Stop_Loss']}** (2x ATR)\n"
                                           f"- Max Risk if stopped out: **${sizing['Max_Loss_Risk']}**")
                            
                        st.markdown(row['Reasoning'])

            # --- DOWNLOAD BUTTON FOR RADAR SCAN ---
            st.divider()
            csv_scan = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete AI Scan Report (CSV)",
                data=csv_scan,
                file_name=f"ai_radar_scan_{datetime.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

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
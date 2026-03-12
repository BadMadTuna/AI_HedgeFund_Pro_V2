import os
import json
import google.generativeai as genai

class AIAgent:
    def __init__(self):
        # Ensure you have GEMINI_API_KEY in your environment variables
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # Using Gemini 2.5 Flash for speed, or Pro for deep reasoning. 
        # (Change to gemini-2.5-pro if you prefer deeper analysis over speed)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def _clean_json(self, text):
        """Helper to strip markdown formatting from Gemini JSON responses"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def get_hunter_verdict(self, ticker, tech_data, news, earnings, current_regime):
        """
        PHASE 2: The Universal AI Analyst (Layer 4).
        Dynamically analyzes payloads from either the Momentum Engine or Mean Reversion Engine.
        """
        system_prompt = f"""
        You are the Head Quantitative Analyst at a top-tier hedge fund. 
        Your job is to analyze this stock based on the data payload provided by our algorithmic engines.
        
        CURRENT MACRO REGIME: {current_regime}
        
        TICKER: {ticker}
        ENGINE PAYLOAD: {tech_data}
        RECENT NEWS: {news}
        EARNINGS DATE: {earnings}
        
        HOW TO SCORE THE DATA (BASED ON ENGINE PAYLOAD):
        - IF PAYLOAD CONTAINS 'Smooth_Score' (Engine A - Momentum): You are looking for strong uptrends. Score highly if Momentum is backed by positive fundamental news. Penalize if RSI is > 75.
        - IF PAYLOAD CONTAINS 'Upside_to_Mean' (Engine B - Mean Reversion): You are looking for panic selling in high-quality companies. Score highly if the stock is near or below the 'Lower_BB'. 
        - IF PAYLOAD CONTAINS 'Value_Score' (Engine C - Deep Value): We are in a structural bear market. You are hunting for survival and yield. Score highly if 'Dividend_Yield' is > 3% and 'Debt_to_Equity' is low. Penalize any mention of dividend cuts in the news.
        
        DYNAMIC REGIME RULES:
        1. IF REGIME IS 'QUIET_BULL': Green light for Risk-On. Favor growth and breakouts.
        2. IF REGIME IS 'VOLATILE_BULL': Market is choppy. Demand high free cash flow ('FCF_Yield') to survive the chop.
        3. IF REGIME IS 'QUIET_BEAR': Prioritize Deep Value. Issue an 'AVOID' for high-multiple tech.
        4. IF REGIME IS 'VOLATILE_BEAR': Capital preservation is paramount. Only issue a 'BUY' or 'WATCH' if it is a mathematically extreme mean-reversion setup with bulletproof fundamentals.
        
        UNIVERSAL RULES:
        5. If earnings are within 14 days, downgrade to 'WATCH' (we do not gamble on earnings).
        6. Provide a crisp, 2-3 sentence institutional thesis explaining your score.
        
        You MUST respond ONLY in this exact JSON format. Do not include any other text:
        {{
            "score": [Integer 0-100 indicating conviction],
            "verdict": "BUY" or "WATCH" or "AVOID",
            "reasoning": "[Your 2-3 sentence institutional thesis]"
        }}
        """
        
        try:
            response = self.model.generate_content(system_prompt)
            clean_text = self._clean_json(response.text)
            return json.loads(clean_text)
        except Exception as e:
            return {"score": 0, "verdict": "ERROR", "reasoning": f"AI Parsing Error: {e}"}

    def get_guardian_audit(self, ticker, pos_data, news, earnings, funds, current_regime):
        """
        TAB 1: The Guardian (Regime & Value Aware). 
        """
        cost = pos_data.get('cost', 1)
        live_price = pos_data.get('Live Price (€)', cost)
        pnl_pct = ((live_price - cost) / cost) * 100 if cost > 0 else 0

        # Safely extract yield to pass to the prompt
        div_yield = funds.get('Dividend_Yield', 0) if isinstance(funds, dict) else 0

        system_prompt = f"""
        You are the Chief Risk Officer for a quantitative hedge fund. 
        Your job is to audit an open position in the portfolio and enforce strict risk management rules.
        
        CURRENT MACRO REGIME: {current_regime}
        
        TICKER: {ticker}
        ENTRY COST: {cost}
        LIVE PRICE: {live_price}
        CURRENT PNL %: {pnl_pct:.2f}%
        DIVIDEND YIELD: {div_yield}
        RECENT NEWS: {news}
        EARNINGS DATE: {earnings}
        
        YOUR DYNAMIC RISK RULES (DICTATED BY REGIME & FUNDAMENTALS):
        1. IF REGIME IS 'QUIET_BULL': 
           - Give the trade room to breathe. Hard stop at -8%. 
           - Do not TRIM until +20% or higher. Let winners ride.
        
        2. IF REGIME IS 'VOLATILE_BULL' OR 'VOLATILE_BEAR': 
           - The market is choppy. Rallies fail fast. 
           - Take profits aggressively: advise 'TRIM' if PNL is > +8%.
           - If PNL is negative, scan the news for fundamental deterioration. If bad, cut immediately.
           
        3. IF REGIME IS 'QUIET_BEAR':
           - IF the stock pays a Dividend Yield > 0.03 (3%): This is an Engine C income play. Give it a wider hard stop (-12%) because we are collecting yield. HOWEVER, if the news mentions a "dividend cut" or "credit downgrade", advise 'SELL' immediately.
           - IF the stock pays NO dividend: This is a legacy growth trap from a previous bull market. Tighten the hard stop to -4%. Advise 'SELL' if it breaches, so we can rotate the cash.

        UNIVERSAL RULES:
        4. EARNINGS ROULETTE: If earnings are within 7 days, set Earnings Risk to "Elevated" and advise tightening the stop loss.
        
        You MUST respond ONLY in this exact JSON format. Do not include any other text:
        {{
            "action": "KEEP" or "TRIM" or "SELL",
            "earnings_risk": "Safe" or "Elevated" or "Critical",
            "reasoning": "[2 sentences explaining the action based on the regime, yield, and news]",
            "proposed_stop": "[Exact dollar amount or logic, e.g., 'Trail stop at $X']"
        }}
        """
        
        try:
            response = self.model.generate_content(system_prompt)
            clean_text = self._clean_json(response.text)
            return json.loads(clean_text)
        except Exception as e:
            return {"action": "ERROR", "earnings_risk": "Unknown", "reasoning": f"AI parsing failed: {e}", "proposed_stop": "N/A"}
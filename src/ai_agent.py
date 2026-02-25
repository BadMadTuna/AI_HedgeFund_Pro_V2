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

    def get_hunter_verdict(self, ticker, tech_data, news, earnings):
        """
        PHASE 2: The Hunter. Scrutinizes the Quant Filter's top picks.
        """
        system_prompt = f"""
        You are the Head Quantitative Analyst at a top-tier hedge fund. 
        Your job is to analyze this stock which just passed our Phase 1 momentum filter.
        
        TICKER: {ticker}
        QUANT DATA: {tech_data}
        RECENT NEWS: {news}
        EARNINGS DATE: {earnings}
        
        YOUR RULES:
        1. Be highly skeptical. Only issue a 'BUY' if the technical momentum is backed by strong fundamental news catalysts.
        2. If earnings are within 14 days, downgrade to 'WATCH' (we do not gamble on earnings).
        3. If the RSI is over 75, consider it overextended and issue a 'WATCH' for a pullback.
        4. Provide a crisp, 2-3 sentence thesis for your reasoning.
        
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

    def get_guardian_audit(self, ticker, pos_data, news, earnings):
        """
        TAB 1: The Guardian. Ruthlessly manages your existing portfolio risk.
        """
        # Calculate current PnL safely if the live price is available
        cost = pos_data.get('cost', 1)
        live_price = pos_data.get('Live Price (€)', cost) # Uses the new live price from our previous upgrade!
        pnl_pct = ((live_price - cost) / cost) * 100 if cost > 0 else 0

        system_prompt = f"""
        You are the Chief Risk Officer for a quantitative hedge fund. 
        Your job is to audit an open position in the portfolio and enforce strict risk management rules.
        
        TICKER: {ticker}
        ENTRY COST: {cost}
        LIVE PRICE: {live_price}
        CURRENT PNL %: {pnl_pct:.2f}%
        RECENT NEWS: {news}
        EARNINGS DATE: {earnings}
        
        YOUR RUTHLESS RULES:
        1. CUT LOSERS: If CURRENT PNL % is worse than -8.0%, you MUST advise "SELL". Capital preservation is priority #1.
        2. TAKE PROFITS: If CURRENT PNL % is greater than +20.0%, you MUST advise "TRIM" to lock in partial gains.
        3. EARNINGS ROULETTE: If earnings are within 7 days, set Earnings Risk to "Elevated" and advise tightening the stop loss.
        4. LET WINNERS RIDE: If PNL is between 0% and +19%, advise "KEEP" but propose a trailing stop loss at the 20-day moving average or 8% below current price.
        
        You MUST respond ONLY in this exact JSON format. Do not include any other text:
        {{
            "action": "KEEP" or "TRIM" or "SELL",
            "earnings_risk": "Safe" or "Elevated" or "Critical",
            "reasoning": "[2 sentences explaining the action based on the rules and news]",
            "proposed_stop": "[Exact dollar amount or logic, e.g., 'Trail stop at $X']"
        }}
        """
        
        try:
            response = self.model.generate_content(system_prompt)
            clean_text = self._clean_json(response.text)
            return json.loads(clean_text)
        except Exception as e:
            return {"action": "ERROR", "earnings_risk": "Unknown", "reasoning": f"AI parsing failed: {e}", "proposed_stop": "N/A"}
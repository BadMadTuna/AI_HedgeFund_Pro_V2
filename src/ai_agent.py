import os
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ GOOGLE_API_KEY not found in .env file.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # We use gemini-2.5-flash because it is lightning fast, cost-effective, 
        # and natively supports strict JSON schema enforcement.
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def get_hunter_verdict(self, ticker: str, technicals: dict, news: str, earnings: str, market_regime: str = "Neutral") -> dict:
        """Analyzes a new stock idea and returns strictly structured JSON."""
        
        prompt = f"""
        You are a Senior Hedge Fund Manager. Evaluate {ticker} based on the provided data.
        
        --- DATA ---
        Technicals: {technicals}
        Earnings Risk: {earnings}
        Market Regime: {market_regime}
        Recent News:
        {news}
        
        --- RULES ---
        1. If earnings are within 7 days, cap the score and strongly consider WATCH or AVOID.
        2. If RSI > 70, penalize the score. If RSI < 35, boost the score (value buy).
        3. Score > 80 = BUY, 60-80 = WATCH, < 60 = AVOID.
        
        Analyze the data logically and provide your structured assessment.
        """
        
        # The Magic: Forcing Gemini to output a guaranteed schema
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,  # Keep it low for analytical consistency
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "INTEGER", "description": "Score from 0 to 100"},
                        "verdict": {"type": "STRING", "description": "Must be exactly one of: BUY, WATCH, AVOID"},
                        "reasoning": {"type": "STRING", "description": "Concise reasoning for the verdict (max 2 sentences)"}
                    },
                    "required": ["score", "verdict", "reasoning"]
                }
            )
        )
        
        try:
            return json.loads(response.text)
        except Exception as e:
            print(f"Agent Parsing Error: {e}")
            return {"score": 0, "verdict": "ERROR", "reasoning": "Failed to parse AI output."}

    def get_guardian_audit(self, ticker: str, position_data: dict, news: str, earnings: str) -> dict:
        """Evaluates a currently held stock and advises on risk management."""
        
        prompt = f"""
        You are a strict Portfolio Risk Manager. Review our current holding in {ticker}.
        
        --- POSITION DATA ---
        {position_data}
        
        --- MARKET DATA ---
        Earnings Risk: {earnings}
        Recent News:
        {news}
        
        --- RULES ---
        1. If 'Earnings Risk' says 'Safe', DO NOT hallucinate earnings risk from the news.
        2. If RSI > 80, or if Gain > 20%, recommend TRIM.
        3. If the position is already 'Trimmed' and momentum is fading, recommend TRAIL STOP.
        4. If PnL is deeply negative (-8% or worse), recommend SELL to cut losses.
        5. Otherwise, recommend KEEP.
        """
        
        response = self.model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "ticker": {"type": "STRING"},
                        "action": {"type": "STRING", "description": "Must be exactly one of: KEEP, TRIM, TRAIL STOP, SELL"},
                        "earnings_risk": {"type": "STRING", "description": "Summarize the earnings risk in 2-3 words"},
                        "reasoning": {"type": "STRING", "description": "Concise risk management advice based on the rules"},
                        "proposed_stop": {"type": "STRING", "description": "Suggested stop loss action or price"}
                    },
                    "required": ["ticker", "action", "earnings_risk", "reasoning", "proposed_stop"]
                }
            )
        )
        
        try:
            return json.loads(response.text)
        except Exception as e:
            return {"ticker": ticker, "action": "ERROR", "earnings_risk": "Unknown", "reasoning": str(e), "proposed_stop": "None"}

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing AI Agent Structured Output...")
    agent = AIAgent()
    
    # Mock Data
    mock_tech = {"RSI": 45, "SMA_50": 150, "Price": 145}
    mock_news = "- Apple releases new AI features.\n- Strong iPhone demand reported."
    mock_earnings = "Safe (Earnings in 66 days)"
    mock_pos = {"cost": 120.0, "quantity": 50, "gain_pct": 20.8, "status": "Open"}
    
    print("\n--- The Hunter (Evaluating a new idea) ---")
    hunter_res = agent.get_hunter_verdict("AAPL", mock_tech, mock_news, mock_earnings)
    print(json.dumps(hunter_res, indent=2))
    
    print("\n--- The Guardian (Evaluating our portfolio holding) ---")
    guardian_res = agent.get_guardian_audit("AAPL", mock_pos, mock_news, mock_earnings)
    print(json.dumps(guardian_res, indent=2))
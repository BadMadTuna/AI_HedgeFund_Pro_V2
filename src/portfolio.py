import pandas as pd
from datetime import datetime
from src.database import SessionLocal, Position, Trade

class PortfolioManager:
    def __init__(self):
        # We don't keep a persistent session open, we open/close per transaction to be thread-safe
        pass

    def get_equity_summary(self) -> dict:
        """Calculates total account equity, cash, and invested amounts."""
        db = SessionLocal()
        positions = db.query(Position).all()
        db.close()

        cash = 0.0
        invested = 0.0

        for p in positions:
            if p.ticker == "EUR" or p.ticker == "CASH":
                cash += p.quantity
            elif p.quantity > 0:
                # For a real-time summary, you'd multiply p.quantity by CURRENT price.
                # For simplicity here, we use the cost basis to show "invested capital".
                invested += (p.cost * p.quantity)

        total_equity = cash + invested
        return {
            "total_equity": round(total_equity, 2),
            "cash": round(cash, 2),
            "invested": round(invested, 2)
        }

    def calculate_smart_size(self, entry_price: float, stop_loss: float, risk_pct: float = 1.0) -> int:
        """Calculates how many shares to buy based on portfolio equity and risk tolerance."""
        if entry_price <= stop_loss or entry_price <= 0:
            return 0
            
        summary = self.get_equity_summary()
        account_size = summary["total_equity"]
        
        risk_budget = account_size * (risk_pct / 100.0)
        risk_per_share = entry_price - stop_loss
        
        shares = int(risk_budget / risk_per_share)
        
        # Guardrail: Never put more than 20% of total equity into a single trade
        max_capital_allowed = account_size * 0.20
        if (shares * entry_price) > max_capital_allowed:
            shares = int(max_capital_allowed / entry_price)
            
        return max(0, shares)

    def execute_buy(self, ticker: str, price: float, quantity: int, target: float = 0.0, reason: str = "Manual Buy") -> bool:
        """Executes a buy order, deducts cash, and logs the trade."""
        if quantity <= 0:
            print("Quantity must be > 0.")
            return False

        cost_of_trade = price * quantity
        db = SessionLocal()
        
        try:
            # 1. Check Cash
            cash_pos = db.query(Position).filter(Position.ticker.in_(["EUR", "CASH"])).first()
            if not cash_pos or cash_pos.quantity < cost_of_trade:
                print("⚠️ Insufficient funds!")
                db.close()
                return False
                
            # 2. Deduct Cash
            cash_pos.quantity -= cost_of_trade
            
            # 3. Add Position
            existing_pos = db.query(Position).filter(Position.ticker == ticker).first()
            if existing_pos:
                # Average up/down logic
                total_cost = (existing_pos.cost * existing_pos.quantity) + cost_of_trade
                new_qty = existing_pos.quantity + quantity
                existing_pos.cost = total_cost / new_qty
                existing_pos.quantity = new_qty
            else:
                new_pos = Position(ticker=ticker, cost=price, quantity=quantity, target=target, status="Open")
                db.add(new_pos)
                
            # 4. Log to Journal
            trade_log = Trade(
                ticker=ticker, action="BUY", quantity=quantity, 
                entry_price=price, exit_price=0.0, pnl_pct=0.0, pnl_abs=0.0, reason=reason
            )
            db.add(trade_log)
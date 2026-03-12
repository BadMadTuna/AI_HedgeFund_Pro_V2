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

    # In src/portfolio.py

    def calculate_smart_size(self, entry_price: float, stop_loss: float, regime: str = "QUIET_BULL", risk_pct: float = 1.0) -> int:
        """Calculates position size with Dynamic Regime Guardrails."""
        if entry_price <= stop_loss or entry_price <= 0:
            return 0
            
        summary = self.get_equity_summary()
        account_size = summary["total_equity"]
        
        # 1. Base Risk Calculation
        risk_budget = account_size * (risk_pct / 100.0)
        risk_per_share = entry_price - stop_loss
        shares = int(risk_budget / risk_per_share)
        
        # 2. THE REGIME GUARDRAILS (Dynamic Capital Allocation)
        # We cap the maximum % of the total portfolio allowed in a single trade based on the environment.
        if regime == "VOLATILE_BEAR":
            # Cash is king. Block all new standard long entries.
            return 0 
            
        regime_caps = {
            "QUIET_BULL": 0.20,    # Max 20% of account per trade (Aggressive)
            "VOLATILE_BULL": 0.08, # Max 8% of account per trade (Overnight gap risk is high)
            "QUIET_BEAR": 0.04     # Max 4% of account per trade (Highly defensive)
        }
        
        cap_pct = regime_caps.get(regime, 0.04) # Default to strict defense if unknown
        max_capital_allowed = account_size * cap_pct
        
        # 3. Apply the Cap
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
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Trade Error: {e}")
            return False
        finally:
            db.close()

    def execute_sell(self, ticker: str, price: float, quantity: float = None, reason: str = "Manual Sell") -> bool:
        """Executes a sell/trim order, adds to cash, and logs the trade."""
        db = SessionLocal()
        
        try:
            # 1. Find Position
            pos = db.query(Position).filter(Position.ticker == ticker).first()
            if not pos or pos.quantity <= 0:
                print(f"⚠️ No active position found for {ticker}.")
                db.close()
                return False
                
            qty_to_sell = quantity if quantity and quantity < pos.quantity else pos.quantity
            if qty_to_sell <= 0:
                db.close()
                return False

            # 2. Calculate PnL
            sale_proceeds = price * qty_to_sell
            cost_basis = pos.cost * qty_to_sell
            pnl_abs = sale_proceeds - cost_basis
            pnl_pct = ((price - pos.cost) / pos.cost) * 100 if pos.cost > 0 else 0.0
            
            # 3. Update Position
            pos.quantity -= qty_to_sell
            if pos.quantity == 0:
                db.delete(pos)
                action_type = "SELL (Full)"
            else:
                pos.status = "Trimmed"
                action_type = "TRIM"
                
            # 4. Add Cash
            cash_pos = db.query(Position).filter(Position.ticker.in_(["EUR", "CASH"])).first()
            if cash_pos:
                cash_pos.quantity += sale_proceeds
            else:
                new_cash = Position(ticker="EUR", cost=1.0, quantity=sale_proceeds, status="Liquid")
                db.add(new_cash)
                
            # 5. Log to Journal
            trade_log = Trade(
                ticker=ticker, action=action_type, quantity=qty_to_sell, 
                entry_price=pos.cost, exit_price=price, pnl_pct=round(pnl_pct, 2), 
                pnl_abs=round(pnl_abs, 2), reason=reason
            )
            db.add(trade_log)
            
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            print(f"Sell Error: {e}")
            return False
        finally:
            db.close()

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Testing Portfolio Manager & Trade Execution...")
    pm = PortfolioManager()
    
    print("\n1. Initial Equity Summary:")
    print(pm.get_equity_summary())
    
    print("\n2. Executing Buy (10 shares of AAPL at $150)...")
    success = pm.execute_buy("AAPL", 150.0, 10, target=200.0, reason="AI Test Buy")
    print(f"Buy Success: {success}")
    
    print("\n3. Equity After Buy:")
    print(pm.get_equity_summary())
    
    print("\n4. Executing Sell (5 shares of AAPL at $175 - Taking Profit)...")
    pm.execute_sell("AAPL", 175.0, quantity=5, reason="Taking 50% Profit")
    
    print("\n5. Reading the Trade Journal:")
    from src.database import get_journal_df
    print(get_journal_df().to_string(index=False))
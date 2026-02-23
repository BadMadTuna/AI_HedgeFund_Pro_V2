import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# --- PATH SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DB_PATH = os.path.join(DATA_DIR, 'hedgefund.db')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# --- DATABASE SETUP ---
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- TABLE SCHEMAS ---
class Position(Base):
    """Represents an active holding in the portfolio."""
    __tablename__ = 'portfolio'
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    cost = Column(Float, default=0.0)
    quantity = Column(Float, default=0.0)
    target = Column(Float, default=0.0)
    status = Column(String, default="Open")
    date_acquired = Column(DateTime, default=datetime.now)

class Trade(Base):
    """Represents a closed or trimmed trade in the journal."""
    __tablename__ = 'journal'
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.now)
    ticker = Column(String, nullable=False)
    action = Column(String, nullable=False)  # BUY, SELL, TRIM
    quantity = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    exit_price = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    pnl_abs = Column(Float, default=0.0)
    reason = Column(String, default="")

# --- CORE FUNCTIONS ---
def init_db():
    """Creates the tables if they don't exist and seeds initial cash."""
    Base.metadata.create_all(bind=engine)
    
    # Seed a default cash position if the portfolio is entirely empty
    db = SessionLocal()
    if db.query(Position).count() == 0:
        cash = Position(ticker="EUR", cost=1.0, quantity=10000.0, target=0.0, status="Liquid")
        db.add(cash)
        db.commit()
    db.close()

def get_portfolio_df() -> pd.DataFrame:
    """Returns the current portfolio as a Pandas DataFrame."""
    db = SessionLocal()
    positions = db.query(Position).filter(Position.quantity > 0).all()
    db.close()
    
    if not positions:
        return pd.DataFrame(columns=['id', 'ticker', 'cost', 'quantity', 'target', 'status', 'date_acquired'])
        
    data = [{
        "id": p.id, "ticker": p.ticker, "cost": p.cost, 
        "quantity": p.quantity, "target": p.target, 
        "status": p.status, "date_acquired": p.date_acquired
    } for p in positions]
    return pd.DataFrame(data)

def get_journal_df() -> pd.DataFrame:
    """Returns the trade journal as a Pandas DataFrame."""
    db = SessionLocal()
    trades = db.query(Trade).order_by(Trade.date.desc()).all()
    db.close()
    
    if not trades:
        return pd.DataFrame()
        
    data = [{
        "date": t.date.strftime("%Y-%m-%d %H:%M"), "ticker": t.ticker, 
        "action": t.action, "quantity": t.quantity, "entry": t.entry_price, 
        "exit": t.exit_price, "pnl_pct": t.pnl_pct, "pnl_abs": t.pnl_abs, "reason": t.reason
    } for t in trades]
    return pd.DataFrame(data)

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Initializing Database...")
    init_db()
    print(f"Database created at: {DB_PATH}")
    
    print("\n--- Current Portfolio ---")
    df = get_portfolio_df()
    print(df.to_string(index=False))
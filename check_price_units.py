from services.scanner import fetch_ssi_market_data
from vnstock import Vnstock
import pandas as pd
from datetime import datetime

async def check_units():
    print("--- Fetching SSI Data ---")
    ssi_df = fetch_ssi_market_data()
    if ssi_df.empty:
        print("SSI Fetch Failed")
        return
        
    hpg_ssi = ssi_df[ssi_df['symbol'] == 'HPG'].iloc[0]
    print(f"SSI HPG Close: {hpg_ssi['close']} (Raw)")
    print(f"SSI HPG Volume: {hpg_ssi['volume']}")

    print("\n--- Fetching Vnstock History ---")
    stock = Vnstock().stock(symbol='HPG', source='VCI')
    df_hist = stock.quote.history(start="2025-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    hpg_vnstock = df_hist.iloc[-1]
    
    print(f"Vnstock HPG Close: {hpg_vnstock['close']} (Raw)")
    print(f"Vnstock HPG Volume: {hpg_vnstock['volume']}")
    
    # Check ratio
    ratio = hpg_ssi['close'] / hpg_vnstock['close']
    print(f"\nRatio (SSI / Vnstock): {ratio:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(check_units())

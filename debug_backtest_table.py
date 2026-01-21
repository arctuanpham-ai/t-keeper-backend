import asyncio
import pandas as pd
from services.t5_audit_report import generate_t5_audit_report
from services.scanner import fetch_stock_data, process_single_stock

async def print_backtest_details(symbol="HPG"):
    print(f"\n--- FETCHING DATA FOR {symbol} ---")
    data_bundle = fetch_stock_data(symbol, days=90) # Get enough history
    
    if not data_bundle or 'df' not in data_bundle:
        print("Failed to fetch data.")
        return

    df = data_bundle['df']
    latest = data_bundle['latest']
    
    # Calculate simple MA20 if not present
    if 'ma20' not in df.columns:
        df['ma20'] = df['close'].rolling(20).mean()
        
    price = float(latest['close'])
    volume = int(latest['volume'])
    ma20 = float(df['ma20'].iloc[-1]) if not pd.isna(df['ma20'].iloc[-1]) else price
    
    print(f"Current State: Price={price}, MA20={ma20:.2f}, Trend={'UP' if price > ma20 else 'DOWN'}")
    
    # Generate Report
    report = generate_t5_audit_report(
        symbol=symbol,
        price=price,
        volume=volume,
        rsi=50, # Dummy, not used in simplified backtest
        ma20=ma20,
        historical_df=df
    )
    
    # Extract Backtest Details
    hist_section = report.get('backtest_stats', {})
    details = hist_section.get('match_details', [])
    
    # Check if we need to call analyze_pattern_matching manually? 
    # The new generate_t5_audit_report already returns match_details in backtest_stats.
    # So we don't need to import and call generic function anymore.
    
    print(f"\n--- DETAILED BACKTEST TABLE: {symbol} ---")
    print(f"{'DATE':<12} | {'T0 PRICE':<10} | {'T+5 PRICE':<10} | {'PROFIT %':<10}")
    print("-" * 50)
    
    for row in details:
        date_str = row['date'].split('T')[0] if 'T' in row['date'] else row['date']
        print(f"{date_str:<12} | {row['t0_price']:<10} | {row['t5_price']:<10} | {row['profit_pct']:>7.2f}%")
        
    print(f"\nTotal Matches Found: {hist_section.get('similar_days_found', 0)}")
    print(f"Win Rate (>3% profit): {hist_section.get('t5_win_rate', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(print_backtest_details("HPG"))

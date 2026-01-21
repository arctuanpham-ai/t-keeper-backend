import time
from services.scanner import fetch_stock_data

def test_fetch():
    symbol = "HPG"
    print(f"Fetching data for {symbol}...")
    start = time.time()
    data = fetch_stock_data(symbol, days=60)
    elapsed = time.time() - start
    
    if data:
        print(f"Success! Time: {elapsed:.2f}s")
        print(f"Rows: {len(data['df'])}")
    else:
        print(f"Failed! Time: {elapsed:.2f}s")

if __name__ == "__main__":
    test_fetch()

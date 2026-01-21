import requests
import json
from vnstock import Vnstock
import pandas as pd

def test_vnstock_tcbs():
    print("\n--- Testing VNStock (TCBS Source) ---")
    try:
        stock = Vnstock().stock(symbol='VNINDEX', source='TCBS')
        df = stock.quote.history(days=1)
        print("VNINDEX (TCBS):")
        print(df)
    except Exception as e:
        print(f"Error (TCBS): {e}")

def test_vndi_direct():
    print("\n--- Testing VNDirect API (Direct) ---")
    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
    params = {
        "sort": "date",
        "q": "code:VNINDEX,VN30,HNX",
        "size": "5",
        "group": "index"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://dstock.vndirect.com.vn/",
        "Origin": "https://dstock.vndirect.com.vn",
        "Accept": "application/json"
    }
    
    try:
        resp = requests.get(url, params=params, headers=headers, verify=False, timeout=10)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                print(f"Indices found: {len(data['data'])}")
                print(data['data'][:2])
            else:
                print("No data key in response")
        else:
            print(resp.text[:200])
    except Exception as e:
        print(f"Error (VNDI): {e}")

if __name__ == "__main__":
    test_vnstock_tcbs()
    test_vndi_direct()

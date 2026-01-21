import requests
import json
import pandas as pd
import urllib3
urllib3.disable_warnings()

SSI_API_URL = "https://iboard-query.ssi.com.vn/stock/"
SSI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

def check_ssi_raw():
    print("Fetching SSI Data...")
    try:
        resp = requests.get(SSI_API_URL, headers=SSI_HEADERS, verify=False, timeout=10)
        data = resp.json()
        
        if 'data' not in data:
            print("No data key found!")
            return

        items = data['data']
        print(f"Total items: {len(items)}")
        
        # Check specific symbols
        targets = ['HPG', 'SSI', 'VCB', 'VND']
        
        found_count = 0
        for item in items:
            if item.get('stockSymbol') == 'HPG':
                found_count += 1
                print(f"\n--- HPG ITEM #{found_count} ---")
                print(json.dumps(item, indent=2))
                if found_count >= 3: break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_ssi_raw()

import requests
import json
import time

def fetch_ssi_indices():
    url = "https://iboard-query.ssi.com.vn/exchange-index-stat/ls"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    print("Testing SSI Index Fetch...")
    try:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        print(f"Status: {resp.status_code}")
        if 'data' in data:
            print(f"Indices found: {len(data['data'])}")
            for idx in data['data']:
                if idx['indexCode'] in ['VNINDEX', 'VN30', 'HNXIndex']:
                    print(f"{idx['indexCode']}: {idx['indexValue']} ({idx['change']} / {idx['changePercent']}%) - Vol: {idx['totalMatchVolume']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch_ssi_indices()

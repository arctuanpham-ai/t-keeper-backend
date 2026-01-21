import requests
import json

def test_vndi_index():
    url = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
    params = {
        "sort": "date",
        "q": "code:VNINDEX,VN30,HNX",
        "size": "5",
        "group": "index"  # Important?
    }
    # Note: HNX-INDEX code might be 'HNXIndex' or 'HNX'
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    print("Testing VNDirect Index Fetch...")
    try:
        resp = requests.get(url, params=params, headers=headers, verify=False)
        data = resp.json()
        
        if 'data' in data:
            print(f"Indices found: {len(data['data'])}")
            for idx in data['data']:
                print(f"{idx['code']}: {idx['close']} (Change: {idx['change']} / {idx['pctChange']}%) - Vol: {idx['mnVolume']}")
        else:
            print("No data found")
            print(data)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_vndi_index()

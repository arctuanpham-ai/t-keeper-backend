import requests
import urllib3
urllib3.disable_warnings()

def test_endpoints():
    endpoints = [
        "https://iboard-query.ssi.com.vn/exchange-index-stat/ls",
        "https://iboard-query.ssi.com.vn/index/ls",
        "https://iboard-query.ssi.com.vn/v2/index/ls",
        "https://banggia.vndirect.com.vn/stock-api/status", # random guess
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    for url in endpoints:
        print(f"Testing {url}...")
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=5)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                print(resp.text[:200])
        except Exception as e:
            print(f"Error: {e}")
            
if __name__ == "__main__":
    test_endpoints()

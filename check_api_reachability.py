import requests
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def check_url(url, name):
    print(f"Checking {name}...", end=" ", flush=True)
    start = time.time()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Verify=False for SSL
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        duration = (time.time() - start) * 1000
        print(f"Status: {response.status_code} | Time: {duration:.0f}ms")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    # TCBS One Stock
    check_url("https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker=HPG&type=stock&resolution=D&count=5", "TCBS (HPG History)")
    
    # VNDirect (Loose SSL)
    check_url("https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=floor:HOSE&size=10", "VNDirect (HOSE) - NoSSL")

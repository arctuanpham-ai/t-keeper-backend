import requests
import pandas as pd
import urllib3
urllib3.disable_warnings()

def check_ssi_indices():
    url = "https://iboard-query.ssi.com.vn/stock/"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    print("Fetching SSI Raw Data...")
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=15)
        data = resp.json()
        
        if 'data' not in data:
            print("No data")
            return

        df = pd.DataFrame(data['data'])
        print(f"Total items: {len(df)}")
        
        # Search for index-like symbols
        indices = df[df['stockSymbol'].isin(['VNINDEX', 'VN30', 'HNXIndex', 'HNX30', 'VNXALL', 'UPCOM'])]
        
        if not indices.empty:
            print("Found Indices in Stock Endpoint:")
            print(indices[['stockSymbol', 'matchedPrice', 'priceChangePercent', 'totalVolume']])
        else:
            print("Indices NOT found in stock endpoint.")
            print("Sample symbols:", df['stockSymbol'].head(10).tolist())
            
            # Check distinct exchanges
            if 'exchange' in df.columns:
                print("Exchanges:", df['exchange'].unique())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_ssi_indices()

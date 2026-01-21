from vnstock import Vnstock

def test_vnstock_index():
    print("Testing VNStock Index Fetch...")
    try:
        stock = Vnstock()
        
        # Try fetching VNINDEX price
        # Note: Vnstock might treat index as stock in some functions
        print("Fetching VNINDEX history...")
        df = stock.stock(symbol='VNINDEX', source='VCI').quote.history(days=1)
        print("VNINDEX:")
        print(df)
        
    except Exception as e:
        print(f"Error fetching VNINDEX: {e}")

if __name__ == "__main__":
    test_vnstock_index()

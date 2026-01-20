import asyncio
import sys
import os

# Add parent directory to path to import services
# Add parent directory to path to import services
sys.path.append(os.path.abspath(os.curdir))

from services.scanner import scan_market

async def test_scanner():
    print("Testing High Performance Scanner (Vectorized)...")
    try:
        start = asyncio.get_event_loop().time()
        result = await scan_market(min_price=5000, min_volume=100000, top_n=5)
        end = asyncio.get_event_loop().time()
        
        print(f"Success! Found {len(result.top_stocks)} stocks in {result.processing_time_ms}ms")
        for stock in result.top_stocks:
             print(f"  {stock.symbol} | P: {stock.price} | Score: {stock.score}")
             
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_scanner())

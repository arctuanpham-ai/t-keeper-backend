import asyncio
from services.scanner import scan_market
import pandas as pd

async def test_scanner():
    print("Testing scan_market...")
    try:
        result = await scan_market(top_n=5)
        print("Scanner Result:", result)
    except Exception as e:
        print(f"Scanner Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_scanner())

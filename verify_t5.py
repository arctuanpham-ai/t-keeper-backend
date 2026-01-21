import asyncio
import aiohttp
import json

async def test_quick_t5():
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8000/api/v1/audit/t5-report/HPG"
        print(f"Requesting {url}...")
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("\n--- QUICK T+5 REPORT (VERIFIED) ---")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                else:
                    print(f"Error: {resp.status}")
                    text = await resp.text()
                    print(text)
        except Exception as e:
            print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_quick_t5())

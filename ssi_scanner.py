"""
SSI iBoard Market Scanner - FINAL SOLUTION
==========================================
Sử dụng endpoint /stock/ của SSI iBoard để lấy TOÀN BỘ thị trường
trong 1 request duy nhất!

Ưu điểm:
- Chỉ 1 API request (so với 3 request VNDirect hoặc 16 request POST)
- Trả về 11,000+ mã bao gồm cổ phiếu, ETF, phái sinh
- Real-time data từ SSI iBoard trading engine
- Không bị chặn rate limit
"""
import requests
import pandas as pd
import numpy as np
import time
import urllib3

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}


def fetch_ssi_market_data() -> pd.DataFrame:
    """
    Fetch TOÀN BỘ thị trường từ SSI iBoard trong 1 request.
    
    Returns:
        DataFrame với các columns đã chuẩn hóa:
        - symbol: Mã CK
        - price: Giá khớp (matchedPrice)
        - volume: Tổng KLGD (stockVol)
        - ref_price: Giá tham chiếu
        - open, high, low, close
        - pct_change: % thay đổi giá
        - exchange: Sàn giao dịch
    """
    print("🚀 [SSI Scanner] Fetching entire market data...")
    start = time.time()
    
    url = "https://iboard-query.ssi.com.vn/stock/"
    
    try:
        resp = requests.get(url, headers=HEADERS, verify=False, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('code') != 'SUCCESS' or 'data' not in data:
            print(f"❌ API returned error: {data.get('message')}")
            return pd.DataFrame()
        
        items = data['data']
        print(f"   📊 Raw items received: {len(items)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(items)
        
        # Filter chỉ lấy stocks (không lấy phái sinh, bonds)
        # Dựa vào stockType hoặc exchange
        valid_exchanges = ['hose', 'hnx', 'upcom']  # lowercase
        if 'exchange' in df.columns:
            df = df[df['exchange'].str.lower().isin(valid_exchanges)]
            print(f"   ✅ After filtering exchanges: {len(df)} stocks")
        
        # Standardize column names
        df = df.rename(columns={
            'stockSymbol': 'symbol',
            'matchedPrice': 'price',
            'stockVol': 'volume',  # Tổng KLGD trong ngày
            'refPrice': 'ref_price',
            'openPrice': 'open',
            'highest': 'high',
            'lowest': 'low',
            'priorClosePrice': 'close',  # Giá đóng cửa phiên trước
            'priceChangePercent': 'pct_change',
            'ceiling': 'ceiling',
            'floor': 'floor_price'
        })
        
        # Chọn các columns cần thiết
        cols = ['symbol', 'exchange', 'price', 'volume', 'ref_price', 
                'open', 'high', 'low', 'close', 'pct_change', 
                'ceiling', 'floor_price', 'matchedVolume', 'nmTotalTradedValue']
        
        # Chỉ lấy columns tồn tại
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy()
        
        # Ensure numeric types
        numeric_cols = ['price', 'volume', 'ref_price', 'open', 'high', 'low', 'close', 'pct_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        fetch_time = time.time() - start
        print(f"   ⏱️ Fetch completed in {fetch_time:.2f}s")
        
        return df
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout - SSI server không phản hồi")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching SSI data: {e}")
        return pd.DataFrame()


def calculate_t_plus_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính điểm T+ cho toàn bộ DataFrame (Vectorized operations)
    
    Score components:
    - Price momentum: +2 nếu tăng >= 2%
    - High volume: +1 nếu KLGD > 500,000
    - Very high volume: +2 nếu KLGD > 2,000,000
    - Near ceiling: +2 nếu giá >= 98% trần
    - Positive intraday: +1 nếu giá hiện tại > giá mở cửa
    """
    if df.empty:
        return df
    
    print("📈 [SSI Scanner] Calculating T+ scores...")
    start = time.time()
    
    df['score'] = 0
    
    # 1. Price momentum
    if 'pct_change' in df.columns:
        df.loc[df['pct_change'] >= 2, 'score'] += 2
        df.loc[df['pct_change'] >= 4, 'score'] += 1  # Bonus for strong momentum
    
    # 2. Volume scoring
    if 'volume' in df.columns:
        df.loc[df['volume'] >= 500000, 'score'] += 1
        df.loc[df['volume'] >= 2000000, 'score'] += 2
        df.loc[df['volume'] >= 5000000, 'score'] += 1  # Extra for very high volume
    
    # 3. Near ceiling (breakout potential)
    if 'price' in df.columns and 'ceiling' in df.columns:
        df.loc[(df['price'] > 0) & (df['price'] >= df['ceiling'] * 0.98), 'score'] += 2
    
    # 4. Positive intraday close
    if 'price' in df.columns and 'open' in df.columns:
        df.loc[(df['open'] > 0) & (df['price'] > df['open']), 'score'] += 1
    
    # 5. Liquidity bonus (có giao dịch)
    if 'matchedVolume' in df.columns:
        df.loc[df['matchedVolume'] > 100, 'score'] += 1
    
    score_time = (time.time() - start) * 1000
    print(f"   ⏱️ Scoring completed in {score_time:.1f}ms")
    
    return df


def apply_filters(df: pd.DataFrame, 
                  min_price: int = 5000, 
                  max_price: int = 100000,
                  min_volume: int = 100000) -> pd.DataFrame:
    """Apply user filters"""
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # Price filter
    if 'price' in df.columns:
        df = df[(df['price'] >= min_price) & (df['price'] <= max_price)]
    
    # Volume filter
    if 'volume' in df.columns:
        df = df[df['volume'] >= min_volume]
    
    print(f"   🔍 Filtered: {initial_count} -> {len(df)} stocks")
    
    return df


def scan_market_ssi(min_price: int = 5000, 
                    max_price: int = 100000, 
                    min_volume: int = 100000,
                    top_n: int = 10) -> dict:
    """
    Main Scanner Function - SSI iBoard Edition
    
    Pipeline:
    1. Fetch entire market (1 request, ~2s)
    2. Apply filters
    3. Calculate scores (vectorized, ~50ms)
    4. Sort and return top N
    
    Returns:
        dict with success, stocks, timing info
    """
    total_start = time.time()
    
    print(f"\n{'='*50}")
    print(f"🎯 SSI Market Scanner")
    print(f"   Price: {min_price:,} - {max_price:,}")
    print(f"   Min Volume: {min_volume:,}")
    print(f"   Top N: {top_n}")
    print(f"{'='*50}\n")
    
    # Step 1: Fetch data
    df = fetch_ssi_market_data()
    
    if df.empty:
        return {
            'success': False,
            'message': 'Không thể lấy dữ liệu từ SSI',
            'stocks': [],
            'total_scanned': 0,
            'processing_time_ms': 0
        }
    
    total_scanned = len(df)
    
    # Step 2: Apply filters
    df = apply_filters(df, min_price, max_price, min_volume)
    
    if df.empty:
        return {
            'success': True,
            'message': 'Không có mã nào thỏa điều kiện lọc',
            'stocks': [],
            'total_scanned': total_scanned,
            'processing_time_ms': int((time.time() - total_start) * 1000)
        }
    
    # Step 3: Calculate scores
    df = calculate_t_plus_scores(df)
    
    # Step 4: Sort and get top N
    df = df.sort_values('score', ascending=False)
    top_df = df.head(top_n)
    
    # Convert to list of dicts
    stocks = top_df.to_dict('records')
    
    processing_time = int((time.time() - total_start) * 1000)
    
    qualified_count = len(df[df['score'] > 0])
    
    print(f"\n{'='*50}")
    print(f"🏁 RESULTS")
    print(f"   Total scanned: {total_scanned:,}")
    print(f"   After filter: {len(df):,}")
    print(f"   Qualified (score > 0): {qualified_count:,}")
    print(f"   Processing time: {processing_time}ms")
    print(f"{'='*50}")
    
    return {
        'success': True,
        'message': f'Quét {total_scanned:,} mã trong {processing_time}ms',
        'stocks': stocks,
        'total_scanned': total_scanned,
        'qualified_count': qualified_count,
        'processing_time_ms': processing_time
    }


# ===== TEST =====
if __name__ == "__main__":
    result = scan_market_ssi(
        min_price=10000,    # 10,000 VND
        max_price=50000,    # 50,000 VND
        min_volume=500000,  # 500K shares
        top_n=10
    )
    
    if result['success'] and result['stocks']:
        print(f"\n🏆 TOP {len(result['stocks'])} STOCKS:")
        print("-" * 60)
        for i, stock in enumerate(result['stocks'], 1):
            symbol = stock.get('symbol', 'N/A')
            price = stock.get('price', 0)
            volume = stock.get('volume', 0)
            score = stock.get('score', 0)
            pct = stock.get('pct_change', 0)
            
            print(f"{i:2}. {symbol:6} | Price: {price:>10,.0f} | Vol: {volume:>12,.0f} | Score: {score:2} | Change: {pct:>+.2f}%")
    else:
        print(f"❌ {result.get('message', 'Unknown error')}")

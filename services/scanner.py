"""
The Scanner Engine v3.0 - SSI iBoard + VNDirect Fallback
=========================================================
Performance Target: < 5 seconds for entire market scan

Primary Strategy (SSI iBoard):
- Single API request to fetch entire market (7,000+ stocks)
- Fastest and most reliable

Fallback Strategy (VNDirect):
- 3 requests for HOSE, HNX, UPCOM
- Used when SSI is unavailable
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional
import time
import requests
import urllib3

from models import StockData, TechnicalIndicators, ScoreBreakdown, LatencyStatus, ScannerResponse
from config import SCANNER_MIN_PRICE, SCANNER_MIN_VOLUME, SCANNER_TOP_N

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===== SSI iBoard Configuration (PRIMARY) =====
SSI_API_URL = "https://iboard-query.ssi.com.vn/stock/"
SSI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# ===== VNDirect Configuration (FALLBACK) =====
VNDIRECT_API_BASE = "https://finfo-api.vndirect.com.vn/v4/stock_prices"
VNDIRECT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://banggia.vndirect.com.vn",
    "Referer": "https://banggia.vndirect.com.vn/"
}

REQUEST_TIMEOUT = 15  # seconds


# ==========================================
# SSI iBoard Data Fetching (PRIMARY SOURCE)
# ==========================================

def fetch_ssi_market_data() -> pd.DataFrame:
    """
    Fetch TOÀN BỘ thị trường từ SSI iBoard trong 1 request.
    
    Returns:
        DataFrame với các columns đã chuẩn hóa
    """
    print("[Scanner] SSI: Fetching entire market...")
    start = time.time()
    
    try:
        resp = requests.get(SSI_API_URL, headers=SSI_HEADERS, verify=False, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get('code') != 'SUCCESS' or 'data' not in data:
            print(f"[Scanner] SSI API error: {data.get('message')}")
            return pd.DataFrame()
        
        items = data['data']
        print(f"[Scanner] SSI: Raw items: {len(items)}")
        
        df = pd.DataFrame(items)
        
        # Filter 1: Chỉ lấy boardId = MAIN (Bỏ ODD_LOT, BUY_IN)
        if 'boardId' in df.columns:
            df = df[df['boardId'] == 'MAIN']

        # Filter 2: Chỉ lấy stocks từ sàn chính
        valid_exchanges = ['hose', 'hnx', 'upcom']
        if 'exchange' in df.columns:
            df = df[df['exchange'].str.lower().isin(valid_exchanges)]
            print(f"[Scanner] SSI: After filter (Main & Exchange): {len(df)} stocks")
        
        # Standardize columns
        df = df.rename(columns={
            'stockSymbol': 'symbol',
            'matchedPrice': 'close',  # Use matched price as close
            'stockVol': 'volume',
            'refPrice': 'ref_price',
            'openPrice': 'open',
            'highest': 'high',
            'lowest': 'low',
            'priceChangePercent': 'pct_change',
            'ceiling': 'ceiling',
            'floor': 'floor_price'
        })
        
        # Ensure numeric columns
        numeric_cols = ['close', 'volume', 'ref_price', 'open', 'high', 'low', 'pct_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        fetch_time = time.time() - start
        print(f"[Scanner] SSI: Completed in {fetch_time:.2f}s")
        
        return df
        
    except requests.exceptions.Timeout:
        print("[Scanner] SSI: Request timeout")
        return pd.DataFrame()
    except Exception as e:
        print(f"[Scanner] SSI Error: {e}")
        return pd.DataFrame()


# ==========================================
# VNDirect Data Fetching (FALLBACK)
# ==========================================

def fetch_exchange_batch(exchange: str) -> pd.DataFrame:
    """
    Fetch all stocks from one exchange using VNDirect Finfo API
    Returns DataFrame with price data
    """
    params = {
        "sort": "date",
        "q": f"floor:{exchange}",
        "size": "1000",
        "fields": "code,floor,open,high,low,close,average,volume,nmVolume,nmValue,change,pctChange"
    }
    
    try:
        resp = requests.get(
            VNDIRECT_API_BASE, 
            headers=VNDIRECT_HEADERS, 
            params=params, 
            verify=False,  # Thêm verify=False
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            df['exchange'] = exchange
            return df
        return pd.DataFrame()
        
    except Exception as e:
        print(f"[Scanner] VNDirect Error {exchange}: {e}")
        return pd.DataFrame()


def fetch_vndirect_market_data() -> pd.DataFrame:
    """
    Fallback: Batch fetch entire market using 3 API requests to VNDirect
    """
    exchanges = ['HOSE', 'HNX', 'UPCOM']
    all_data = []
    
    print("[Scanner] VNDirect: Fetching exchanges...", flush=True)
    start = time.time()
    
    for exchange in exchanges:
        df = fetch_exchange_batch(exchange)
        if not df.empty:
            all_data.append(df)
            print(f"  -> {exchange}: {len(df)} stocks")
        time.sleep(0.3)  # Polite delay
    
    fetch_time = time.time() - start
    print(f"[Scanner] VNDirect: Completed in {fetch_time:.2f}s")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Standardize columns
    column_map = {
        'code': 'symbol',
        'accumulatedVol': 'volume',
        'basicPrice': 'ref_price',
        'ceilingPrice': 'ceiling',
        'floorPrice': 'floor_price'
    }
    combined = combined.rename(columns=column_map)
    
    # Ensure numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'ref_price']
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0)
    
    return combined


# ===== Global Cache =====
MARKET_CACHE = {
    "data": pd.DataFrame(),
    "timestamp": 0
}
CACHE_TTL = 30 # seconds

def fetch_all_market_data() -> pd.DataFrame:
    """
    Strategy: SSI (Primary) -> VNDirect (Fallback)
    With Caching to prevent spamming APIs
    """
    global MARKET_CACHE
    
    # Check cache
    if not MARKET_CACHE["data"].empty and (time.time() - MARKET_CACHE["timestamp"] < CACHE_TTL):
        # print("[Scanner] Using cached market data")
        return MARKET_CACHE["data"]

    df = fetch_ssi_market_data()
    if df.empty:
        print("[Scanner] SSI failed, switching to VNDirect...")
        df = fetch_vndirect_market_data()
        
    if not df.empty:
        # Update Cache
        MARKET_CACHE["data"] = df
        MARKET_CACHE["timestamp"] = time.time()
        
    return df

def fetch_stock_data(symbol: str, days: int = 60) -> dict:
    """
    Fetch historical data + Realtime update
    """
    try:
        from vnstock import Vnstock
        client = Vnstock()
        stock = client.stock(symbol=symbol, source='VCI')
        
        # 1. Get History
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return None
            
        # 2. Get Realtime from Cache/Market Scan
        # This ensures we have the latest tick even if history endpoint is lagged
        market_df = fetch_all_market_data()
        
        latest_data = df.iloc[-1].to_dict() # Default to history
        
        if not market_df.empty:
            realtime_row = market_df[market_df['symbol'] == symbol]
            if not realtime_row.empty:
                rt = realtime_row.iloc[0]
                
                # Check if we need to append a new row or update the last one
                # If dataframe last date is today, update. If yesterday, append.
                last_date_str = str(df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['time'].iloc[-1])
                today_str = datetime.now().strftime('%Y-%m-%d')
                
                # SSI returns realtime price/vol
                # VNStock history is in 1000 VND, SSI is in VND.
                # Must normalize SSI to match history.
                rt_price = float(rt['close']) / 1000.0
                rt_vol = int(rt['volume'])
                
                # Update 'latest' dict for immediate use
                latest_data['close'] = rt_price
                latest_data['volume'] = rt_vol
                # Update other fields if needed (High, Low, etc.)
                
                # Merge into DF for backtest accuracy
                # If the last row in DF is NOT today, append new row for "Today"
                # Note: df['time'] usually holds date in YYYY-mm-dd
                
                # Simple logic: Just overwrite/append to ensure backtest sends "Current" correctly
                # actually generate_t5 uses 'price' and 'volume' arguments for T0.
                # So mostly we need to ensure 'latest' is correct.
                # But for 'historical_df' passed to backtest, it's better if it includes today?
                # The backtest logic excludes last 5 days anyway.
                # So appending Today is useful for "Trend Similarity" check of T0.
                
                # Convert index to datetime for comparison
                if not isinstance(df.index, pd.DatetimeIndex):
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                
                # Check if the last entry in history is today
                if df.index[-1].strftime('%Y-%m-%d') == today_str:
                    # Update today's row
                    df.loc[df.index[-1], 'close'] = rt_price
                    df.loc[df.index[-1], 'volume'] = rt_vol
                    # Optionally update high/low/open if available and relevant
                    if 'open' in rt and rt['open'] > 0: df.loc[df.index[-1], 'open'] = float(rt['open']) / 1000.0
                    if 'high' in rt and rt['high'] > 0: df.loc[df.index[-1], 'high'] = float(rt['high']) / 1000.0
                    if 'low' in rt and rt['low'] > 0: df.loc[df.index[-1], 'low'] = float(rt['low']) / 1000.0
                else:
                    # Append a new row for today
                    rt_open = float(rt.get('open', 0)) / 1000.0 if rt.get('open', 0) > 0 else rt_price
                    rt_high = float(rt.get('high', 0)) / 1000.0 if rt.get('high', 0) > 0 else rt_price
                    rt_low = float(rt.get('low', 0)) / 1000.0 if rt.get('low', 0) > 0 else rt_price
                    
                    new_row_data = {
                        'open': rt_open,
                        'high': rt_high,
                        'low': rt_low,
                        'close': rt_price,
                        'volume': rt_vol,
                        # Fill other columns with NaN or default values if needed
                    }
                    # Ensure all columns match the existing DataFrame
                    for col in df.columns:
                        if col not in new_row_data:
                            new_row_data[col] = np.nan
                    
                    new_row_df = pd.DataFrame([new_row_data], index=[pd.to_datetime(today_str)], columns=df.columns)
                    df = pd.concat([df, new_row_df])
                
        return {
            'df': df,
            'latest': latest_data,
            'symbol': symbol
        }
    except Exception as e:
        print(f"[Scanner] Error fetching {symbol}: {e}")
        return None
# ==========================================
# Scoring and Filtering
# ==========================================

def calculate_vectorized_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate T+ scores for all stocks using vectorized operations
    Performance: < 50ms for 7000+ stocks
    """
    if df.empty:
        return df
    
    start = time.time()
    
    df['score'] = 0
    
    # Ensure we have ref_price, fallback to open
    if 'ref_price' not in df.columns or df['ref_price'].sum() == 0:
        df['ref_price'] = df.get('open', 0)
    
    # Calculate % change if not exists
    if 'pct_change' not in df.columns or df['pct_change'].sum() == 0:
        df['pct_change'] = np.where(
            df['ref_price'] > 0,
            (df['close'] - df['ref_price']) / df['ref_price'] * 100,
            0
        )
    
    # Calculate trading value (billion VND)
    df['value_bil'] = (df['close'] * df['volume']) / 1_000_000_000
    
    # ===== SCORING RULES =====
    
    # 1. Price momentum: >= 2% = +2 points
    df.loc[df['pct_change'] >= 2, 'score'] += 2
    
    # 2. Strong momentum: >= 4% = +1 bonus
    df.loc[df['pct_change'] >= 4, 'score'] += 1
    
    # 3. High volume: > 500K shares = +1
    df.loc[df['volume'] >= 500000, 'score'] += 1
    
    # 4. Very high volume: > 2M = +2
    df.loc[df['volume'] >= 2000000, 'score'] += 2
    
    # 5. Extreme volume: > 5M = +1 bonus
    df.loc[df['volume'] >= 5000000, 'score'] += 1
    
    # 6. High value day: > 5 billion VND = +1
    df.loc[df['value_bil'] > 5, 'score'] += 1
    
    # 7. Very high value: > 20 billion = +2
    df.loc[df['value_bil'] > 20, 'score'] += 2
    
    # 8. Price near ceiling (breakout): +2
    if 'ceiling' in df.columns:
        df.loc[(df['close'] > 0) & (df['close'] >= df['ceiling'] * 0.98), 'score'] += 2
    
    # 9. Positive intraday: Close > Open = +1
    if 'open' in df.columns:
        df.loc[(df['open'] > 0) & (df['close'] > df['open']), 'score'] += 1
    
    score_time = (time.time() - start) * 1000
    print(f"[Scanner] Scoring completed in {score_time:.1f}ms")
    
    return df


def apply_filters(df: pd.DataFrame, min_price: int, max_price: int, min_volume: int) -> pd.DataFrame:
    """
    Apply user-defined filters using vectorized operations
    """
    if df.empty:
        return df
    
    # Price filter
    df = df[(df['close'] >= min_price) & (df['close'] <= max_price)]
    
    # Volume filter
    df = df[df['volume'] >= min_volume]
    
    # Symbol filter (Optional, handled by caller if needed, but useful here)
    if 'symbol' in df.columns and hasattr(df, 'symbol_filter') and df.symbol_filter:
         df = df[df['symbol'].str.contains(df.symbol_filter, case=False, na=False)]

    return df


def dataframe_to_stock_data(row: pd.Series) -> StockData:
    """
    Convert a DataFrame row to StockData model
    """
    indicators = TechnicalIndicators(
        rsi=None,
        macd=None,
        macd_signal=None,
        macd_histogram=None,
        ma10=None,
        ma20=None,
        ma50=None,
        volume_change_pct=row.get('pct_change', 0),
        macd_crossover=False,
        pct_change_3d=None
    )
    
    breakdown = ScoreBreakdown(
        volume_score=1 if row.get('volume', 0) >= 100000 else 0,
        momentum_score=int(row.get('pct_change', 0) > 0),
        technical_score=0
    )
    
    # Determine price source
    price_source = "SSI iBoard" if 'exchange' in row.index else "VNDirect"
    
    return StockData(
        symbol=row['symbol'],
        price=int(row['close']),
        price_source=price_source,
        tick_time=datetime.now(),
        received_time=datetime.now(),
        latency_ms=0,
        latency_status=LatencyStatus.OK,
        open=float(row.get('open', 0)),
        high=float(row.get('high', 0)),
        low=float(row.get('low', 0)),
        close=float(row['close']),
        volume=int(row['volume']),
        avg_volume_20=int(row['volume']),
        indicators=indicators,
        score=int(row.get('score', 0)),
        score_breakdown=breakdown
    )


# ==========================================
# Main Scanner Entry Point
# ==========================================

async def scan_market(
    min_price: int = SCANNER_MIN_PRICE,
    max_price: int = 1000000,
    min_volume: int = SCANNER_MIN_VOLUME,
    top_n: int = SCANNER_TOP_N
) -> ScannerResponse:
    """
    Main Scanner Entry Point - SSI iBoard Primary Strategy
    
    Pipeline:
    1. Fetch entire market (1 SSI request or 3 VNDirect requests)
    2. Apply filters
    3. Vectorized scoring (~20ms)
    4. Sort and return Top N
    
    Total target: < 5 seconds
    """
    start_time = time.time()
    
    print(f"[Scanner] Starting: Price {min_price}-{max_price}, Vol>{min_volume}, Top {top_n}")
    
    # Step 1: Fetch all market data
    df = fetch_all_market_data()
    
    if df.empty:
        print("[Scanner] No data fetched, returning mock response")
        return generate_mock_response(top_n)
    
    total_scanned = len(df)
    print(f"[Scanner] Total stocks fetched: {total_scanned}")
    
    # Step 2: Apply filters
    df = apply_filters(df, min_price, max_price, min_volume)
    filtered_count = len(df)
    print(f"[Scanner] After filtering: {filtered_count} stocks")
    
    if df.empty:
        print("[Scanner] No stocks passed filters, returning empty response")
        return ScannerResponse(
            success=True,
            timestamp=datetime.now(),
            processing_time_ms=0,
            total_stocks_scanned=total_scanned,
            qualified_stocks=0,
            top_stocks=[],
            message="Không có mã nào thỏa mãn điều kiện lọc."
        )
    
    # Step 3: Calculate scores
    df = calculate_vectorized_scores(df)
    
    # Step 4: Sort and get top N
    df = df.sort_values('score', ascending=False)
    top_df = df.head(top_n)
    
    # Step 5: Convert to StockData objects
    top_stocks = [dataframe_to_stock_data(row) for _, row in top_df.iterrows()]
    
    processing_time = int((time.time() - start_time) * 1000)
    
    # Count qualified (score > 0)
    qualified_count = len(df[df['score'] > 0])
    
    print(f"[Scanner] Completed in {processing_time}ms")
    
    return ScannerResponse(
        success=True,
        timestamp=datetime.now(),
        processing_time_ms=processing_time,
        total_stocks_scanned=total_scanned,
        qualified_stocks=qualified_count,
        top_stocks=top_stocks,
        message=f"Quét {total_scanned} mã từ SSI, lọc được {filtered_count}, xong trong {processing_time}ms"
    )


def generate_mock_response(top_n: int = 5) -> ScannerResponse:
    """Fallback mock data when both APIs fail"""
    mock_stocks = [
        StockData(
            symbol="HPG", price=28500, price_source="Mock Data",
            tick_time=datetime.now(), received_time=datetime.now(),
            latency_ms=0, latency_status=LatencyStatus.OK,
            open=28000, high=28600, low=27900, close=28500,
            volume=15000000, avg_volume_20=12000000,
            indicators=TechnicalIndicators(rsi=65, ma20=27500, macd=0.5, macd_crossover=True),
            score=9, score_breakdown=ScoreBreakdown(volume_score=5)
        ),
        StockData(
            symbol="SSI", price=34200, price_source="Mock Data",
            tick_time=datetime.now(), received_time=datetime.now(),
            latency_ms=0, latency_status=LatencyStatus.OK,
            open=33500, high=34500, low=33500, close=34200,
            volume=8000000, avg_volume_20=6000000,
            indicators=TechnicalIndicators(rsi=55, ma20=33000, macd=0.8, macd_crossover=True),
            score=8, score_breakdown=ScoreBreakdown(volume_score=4)
        ),
        StockData(
            symbol="VND", price=18500, price_source="Mock Data",
            tick_time=datetime.now(), received_time=datetime.now(),
            latency_ms=0, latency_status=LatencyStatus.OK,
            open=18200, high=18700, low=18100, close=18500,
            volume=12000000, avg_volume_20=10000000,
            indicators=TechnicalIndicators(rsi=60, ma20=18000, macd=0.3, macd_crossover=False),
            score=7, score_breakdown=ScoreBreakdown(volume_score=4)
        ),
        StockData(
            symbol="STB", price=32000, price_source="Mock Data",
            tick_time=datetime.now(), received_time=datetime.now(),
            latency_ms=0, latency_status=LatencyStatus.OK,
            open=31500, high=32200, low=31400, close=32000,
            volume=5000000, avg_volume_20=4000000,
            indicators=TechnicalIndicators(rsi=58, ma20=31000, macd=0.4, macd_crossover=True),
            score=6, score_breakdown=ScoreBreakdown(volume_score=3)
        ),
        StockData(
            symbol="MBB", price=25500, price_source="Mock Data",
            tick_time=datetime.now(), received_time=datetime.now(),
            latency_ms=0, latency_status=LatencyStatus.OK,
            open=25200, high=25700, low=25100, close=25500,
            volume=7000000, avg_volume_20=6000000,
            indicators=TechnicalIndicators(rsi=52, ma20=25000, macd=0.2, macd_crossover=False),
            score=5, score_breakdown=ScoreBreakdown(volume_score=3)
        ),
    ]
    
    return ScannerResponse(
        success=True,
        timestamp=datetime.now(),
        processing_time_ms=10,
        total_stocks_scanned=0,
        qualified_stocks=len(mock_stocks),
        top_stocks=mock_stocks[:top_n],
        message="Dữ liệu mẫu (API không khả dụng)"
    )


# ==========================================
# Single Stock Functions (For Deep Dive)
# ==========================================

def fetch_stock_data(symbol: str, days: int = 60) -> dict:
    """
    Fetch historical data for a single stock
    Used by deep-dive analysis
    """
    try:
        from vnstock import Vnstock
        client = Vnstock()
        stock = client.stock(symbol=symbol, source='VCI')
        
        # Get historical data
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        df = stock.quote.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            return None
        
        return {
            'df': df,
            'latest': df.iloc[-1].to_dict(),
            'symbol': symbol
        }
    except Exception as e:
        print(f"[Scanner] Error fetching {symbol}: {e}")
        return None


def process_single_stock(symbol: str) -> StockData:
    """
    Process a single stock and return StockData with full details
    """
    data = fetch_stock_data(symbol, days=30)
    if data is None:
        return None
    
    df = data['df']
    latest = data['latest']
    
    # Calculate indicators
    from services.indicators import calculate_all_indicators
    indicators_dict = calculate_all_indicators(df) or {}
    
    indicators = TechnicalIndicators(
        rsi=indicators_dict.get('rsi'),
        macd=indicators_dict.get('macd'),
        macd_signal=indicators_dict.get('macd_signal'),
        macd_histogram=indicators_dict.get('macd_histogram'),
        ma10=indicators_dict.get('ma10'),
        ma20=indicators_dict.get('ma20'),
        ma50=indicators_dict.get('ma50'),
        volume_change_pct=indicators_dict.get('volume_change_pct', 0),
        macd_crossover=indicators_dict.get('macd_crossover', False),
        pct_change_3d=indicators_dict.get('pct_change_3d')
    )
    
    # Calculate score
    score = 0
    if indicators.rsi and 50 <= indicators.rsi <= 70:
        score += 2
    if indicators.macd_crossover:
        score += 3
    if indicators.volume_change_pct and indicators.volume_change_pct > 50:
        score += 2
    
    return StockData(
        symbol=symbol,
        price=int(latest.get('close', 0)),
        price_source="VCI Historical",
        tick_time=datetime.now(),
        received_time=datetime.now(),
        latency_ms=0,
        latency_status=LatencyStatus.OK,
        open=float(latest.get('open', 0)),
        high=float(latest.get('high', 0)),
        low=float(latest.get('low', 0)),
        close=float(latest.get('close', 0)),
        volume=int(latest.get('volume', 0)),
        avg_volume_20=int(df['volume'].tail(20).mean()) if 'volume' in df.columns else 0,
        indicators=indicators,
        score=score,
        score_breakdown=ScoreBreakdown(volume_score=1)
    )


# For testing
if __name__ == "__main__":
    import sys
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    async def test():
        result = await scan_market(min_price=10000, max_price=50000, min_volume=500000, top_n=10)
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Message: {result.message}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Total scanned: {result.total_stocks_scanned}")
        print(f"Qualified: {result.qualified_stocks}")
        print(f"\nTop {len(result.top_stocks)} stocks:")
        for stock in result.top_stocks:
            print(f"  {stock.symbol}: Score={stock.score}, Price={stock.price:,}, Vol={stock.volume:,}")
    
    asyncio.run(test())

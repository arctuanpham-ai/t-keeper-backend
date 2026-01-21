"""
T+5 Deep-Dive Investment Report Module
======================================
Refactored to implement "Quick T+5 Report" based on Price Action & 60-session Backtest.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# --- HÀM BỔ TRỢ: TÍNH RSI (Dùng Pandas) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- HÀM CHÍNH: TẠO BÁO CÁO T+5 ---
def generate_t5_audit_report(
    symbol: str,
    price: float,
    volume: int,
    historical_df: Optional[pd.DataFrame] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Sinh báo cáo Quick T+5 theo logic người dùng yêu cầu.
    """
    
    # 0. CHUẨN BỊ DỮ LIỆU
    if historical_df is None or historical_df.empty:
        return {
            "symbol": symbol,
            "error": "Thiếu dữ liệu lịch sử để phân tích.",
            "final_decision": {"action": "KHÔNG CÓ DỮ LIỆU", "score": "0/10", "reason": "No Data"}
        }

    df = historical_df.copy()
    
    # Chuẩn hóa tên cột
    col_map = {
        'time': 'Date', 'date': 'Date',
        'close': 'Close', 'volume': 'Volume',
        'open': 'Open', 'high': 'High', 'low': 'Low'
    }
    df = df.rename(columns=lambda x: col_map.get(x.lower(), x))
    
    # 1. INDICATORS
    # Tính MA20
    df['MA20_Price'] = df['Close'].rolling(window=20).mean()
    df['MA20_Vol'] = df['Volume'].rolling(window=20).mean()
    
    # Tính RSI (14)
    df['RSI'] = calculate_rsi(df['Close'], 14)
    
    # Data points
    if len(df) < 2:
         return {"symbol": symbol, "error": "History < 2 rows"}
         
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Prefer arguments if passed (realtime)
    curr_price = price if price > 0 else current['Close']
    curr_vol = volume if volume > 0 else current['Volume']
    curr_rsi = current['RSI']
    curr_ma20 = current['MA20_Price']
    curr_ma20_vol = current['MA20_Vol']

    # --- LAYER 1: TECHNICAL CHECK (Bộ lọc 10/10) ---
    tech_checks = {
        "price_trend": "TĂNG" if curr_price > curr_ma20 else "GIẢM",
        "rsi_safe": True if (pd.notna(curr_rsi) and curr_rsi < 70) else False,
        "vol_pass": True if curr_vol >= 100000 else False
    }
    
    # --- LAYER 2: MARKET SENTIMENT (Price Action) ---
    # Logic: Price Change & Vol Ratio
    # Check division by zero
    prev_close = prev['Close']
    if prev_close == 0: prev_close = curr_price # Fail-safe

    price_change = (curr_price - prev_close) / prev_close
    
    ma_vol = curr_ma20_vol if pd.notna(curr_ma20_vol) else 1
    if ma_vol == 0: ma_vol = 1
    
    vol_ratio = curr_vol / ma_vol
    
    sentiment_status = "TRUNG LẬP"
    sentiment_score = 0
    sentiment_desc = ""

    if price_change > 0: # Giá Xanh
        if vol_ratio > 1.2:
            sentiment_status = "HƯNG PHẤN (DÒNG TIỀN MẠNH)"
            sentiment_score = 1
            sentiment_desc = "Giá tăng kèm Vol đột biến (>120% TB20). Cầu vào quyết liệt."
        elif vol_ratio < 0.8:
            sentiment_status = "NGHI NGỜ (CẦU YẾU)"
            sentiment_desc = "Giá tăng nhưng Vol thấp. Cẩn trọng Bull-trap."
            sentiment_score = 0
        else:
            sentiment_status = "TÍCH CỰC"
            sentiment_score = 0.5
            sentiment_desc = "Tăng giá ổn định với thanh khoản trung bình."
    else: # Giá Đỏ
        if vol_ratio > 1.2:
            sentiment_status = "SỢ HÃI (BÁN THÁO)"
            sentiment_score = -1
            sentiment_desc = "Giá giảm kèm Vol lớn. Áp lực xả hàng mạnh."
        elif vol_ratio < 0.8:
            sentiment_status = "TIẾT CUNG (TEST ĐÁY)"
            sentiment_score = 0.5 
            sentiment_desc = "Giá giảm nhẹ, Vol cạn kiệt. Không còn ai muốn bán."
        else:
            sentiment_status = "TIÊU CỰC"
            sentiment_score = -0.5
            sentiment_desc = "Giảm giá thông thường."

    # --- LAYER 3: BACKTEST 60 PHIÊN ---
    is_above_ma20 = curr_price > curr_ma20
    
    similar_days = 0
    wins = 0
    match_details = []
    
    # Limit to 60 recent sessions
    limit_scan = 60
    # Safe checks for start index
    start_idx = max(20, len(df) - limit_scan - 5)
    end_idx = len(df) - 5
    
    closes = df['Close'].values
    ma20s = df['MA20_Price'].values
    dates_arr = df['Date'].values
    
    if end_idx > start_idx:
        for i in range(start_idx, end_idx):
            if i >= len(closes): break
            
            c_i = closes[i]
            ma_i = ma20s[i]
            
            if pd.isna(c_i) or pd.isna(ma_i): continue
            
            hist_trend = c_i > ma_i
            
            # Trend Similarity Logic
            if hist_trend == is_above_ma20:
                similar_days += 1
                # T+5 Check
                if i + 5 < len(closes):
                    c_future = closes[i+5]
                    # Win if profit > 3%
                    if c_future > c_i * 1.03:
                        wins += 1
                    
                    # Store details for UI (Last 10 matches or specific relevant ones)
                    dt_str = str(dates_arr[i])
                    if 'T' in dt_str: dt_str = dt_str.split('T')[0]
                    pct = ((c_future - c_i) / c_i) * 100
                    
                    match_details.append({
                        "date": dt_str,
                        "t0_price": float(f"{c_i:.2f}"),
                        "t5_price": float(f"{c_future:.2f}"),
                        "profit_pct": float(f"{pct:.2f}")
                    })

    win_rate = (wins / similar_days * 100) if similar_days > 0 else 0
    match_details.reverse() # Newest first

    # --- TỔNG HỢP & RA QUYẾT ĐỊNH ---
    final_score = 5
    if tech_checks['rsi_safe']: final_score += 1
    else: final_score -= 2
    if tech_checks['vol_pass']: final_score += 1
    if tech_checks['price_trend'] == "TĂNG": final_score += 1
    
    final_score += sentiment_score
    
    if win_rate >= 60: final_score += 2
    elif win_rate < 30: final_score -= 2
    
    # Clamp
    final_score = max(0, min(10, final_score))
    
    action = "QUAN SÁT"
    decision_color = "yellow"
    if final_score >= 8: 
        action = "MUA MẠNH"
        decision_color = "green"
    elif final_score >= 6: 
        action = "MUA THĂM DÒ"
        decision_color = "blue"
    elif final_score <= 4: 
        action = "BÁN / CƠ CẤU"
        decision_color = "red"
    
    # --- OUTPUT JSON DICT ---
    # Matching User's Request Keys + Extra for safety
    
    report = {
        "symbol": symbol,
        "sentiment_analysis": {
            "status": sentiment_status,
            "description": sentiment_desc,
            "score_contribution": f"{sentiment_score:+}",
            "vol_ratio": f"{vol_ratio:.2f}x"
        },
        "backtest_60_sessions": {
            "trend_similarity": f"Đã tìm thấy {similar_days} phiên có xu hướng {'TĂNG' if is_above_ma20 else 'GIẢM'} tương tự.",
            "win_rate_t5": f"{win_rate:.1f}%",
            "avg_profit_t5": "Win Condition: >3% Profit",  # User template text
            "conclusion": "Dữ liệu quá khứ ủng hộ kịch bản T+5." if win_rate > 50 else "Dữ liệu quá khứ không ủng hộ.",
            "match_details": match_details[:10], # Keep detailed list for table
            "similar_days_found": similar_days,  # For verification scripts
            "t5_win_rate_raw": win_rate
        },
        "technical_check": {
            "price": f"{curr_price}",
            "rsi": f"{curr_rsi:.1f}",
            "vol": f"{curr_vol/1000000:.2f}M",
            "trend": tech_checks['price_trend']
        },
        "final_decision": {
            "score": f"{final_score:.1f}/10",
            "action": action,
            "color": decision_color,
            "reason": f"Winrate quá khứ {win_rate:.0f}% + Tâm lý {sentiment_status}."
        },
        # Legacy mappings for generic consumers
        "overall_score": float(final_score),
        "win_probability": f"{win_rate:.1f}%",
        "decision": action,
        "decision_color": decision_color
    }
    
    return report

if __name__ == "__main__":
    # Mock
    dates = pd.date_range(end=datetime.now(), periods=80)
    df_mock = pd.DataFrame({
        'Date': dates,
        'Close': np.linspace(20, 25, 80),
        'Volume': np.random.randint(100000, 2000000, 80)
    })
    print(generate_t5_audit_report("TEST", 25.5, 1500000, df_mock))

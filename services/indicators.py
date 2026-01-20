"""
Technical Indicators Calculator
Calculates RSI, MACD, MA for T+ scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    rsi_indicator = RSIIndicator(close=prices, window=period)
    return rsi_indicator.rsi()


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD, Signal line, and Histogram"""
    macd_indicator = MACD(close=prices, window_fast=fast, window_slow=slow, window_sign=signal)
    return {
        "macd": macd_indicator.macd(),
        "signal": macd_indicator.macd_signal(),
        "histogram": macd_indicator.macd_diff()
    }


def calculate_ma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    sma_indicator = SMAIndicator(close=prices, window=period)
    return sma_indicator.sma_indicator()


def calculate_volume_change(volumes: pd.Series, period: int = 20) -> pd.Series:
    """Calculate volume change percentage vs average"""
    avg_volume = volumes.rolling(window=period).mean()
    return ((volumes - avg_volume) / avg_volume * 100).fillna(0)


def detect_macd_crossover(macd: float, signal: float, prev_macd: float, prev_signal: float) -> bool:
    """Detect if MACD crossed above signal line"""
    if prev_macd is None or prev_signal is None:
        return False
    # Bullish crossover: MACD was below signal, now above
    return prev_macd <= prev_signal and macd > signal


def calculate_all_indicators(df: pd.DataFrame) -> Dict:
    """
    Calculate all technical indicators for a stock
    
    Args:
        df: DataFrame with columns [open, high, low, close, volume]
    
    Returns:
        Dictionary with all indicator values for the latest bar
    """
    if df.empty or len(df) < 50:
        return None
    
    try:
        # Ensure we have required columns
        required_cols = ['close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        close = df['close']
        volume = df['volume']
        
        # Calculate indicators
        rsi = calculate_rsi(close)
        macd_data = calculate_macd(close)
        ma10 = calculate_ma(close, 10)
        ma20 = calculate_ma(close, 20)
        ma50 = calculate_ma(close, 50)
        vol_change = calculate_volume_change(volume)
        
        # Get latest values
        latest_idx = len(df) - 1
        prev_idx = latest_idx - 1 if latest_idx > 0 else 0
        
        # Detect MACD crossover
        macd_cross = detect_macd_crossover(
            macd_data['macd'].iloc[latest_idx],
            macd_data['signal'].iloc[latest_idx],
            macd_data['macd'].iloc[prev_idx] if prev_idx >= 0 else None,
            macd_data['signal'].iloc[prev_idx] if prev_idx >= 0 else None
        )
        
        return {
            "rsi": round(rsi.iloc[latest_idx], 2) if not pd.isna(rsi.iloc[latest_idx]) else None,
            "macd": round(macd_data['macd'].iloc[latest_idx], 2) if not pd.isna(macd_data['macd'].iloc[latest_idx]) else None,
            "macd_signal": round(macd_data['signal'].iloc[latest_idx], 2) if not pd.isna(macd_data['signal'].iloc[latest_idx]) else None,
            "macd_histogram": round(macd_data['histogram'].iloc[latest_idx], 2) if not pd.isna(macd_data['histogram'].iloc[latest_idx]) else None,
            "macd_crossover": macd_cross,
            "ma10": round(ma10.iloc[latest_idx], 2) if not pd.isna(ma10.iloc[latest_idx]) else None,
            "ma20": round(ma20.iloc[latest_idx], 2) if not pd.isna(ma20.iloc[latest_idx]) else None,
            "ma50": round(ma50.iloc[latest_idx], 2) if not pd.isna(ma50.iloc[latest_idx]) else None,
            "volume_change_pct": round(vol_change.iloc[latest_idx], 2) if not pd.isna(vol_change.iloc[latest_idx]) else None,
            "avg_volume_20": int(volume.rolling(20).mean().iloc[latest_idx]) if not pd.isna(volume.rolling(20).mean().iloc[latest_idx]) else None,
            "close": float(close.iloc[latest_idx]),
        }
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return None


def calculate_t_plus_score(indicators: Dict, close_price: float) -> Dict:
    """
    Calculate T+ score based on technical indicators
    
    Scoring criteria:
    - RSI < 30 (Oversold): +2
    - 30 <= RSI <= 70 (Safe zone): +1
    - Price > MA20: +3
    - Price > MA50: +2
    - Volume > 150% avg: +5
    - MACD crossover: +3
    
    Returns:
        Dictionary with total score and breakdown
    """
    score = 0
    breakdown = {
        "rsi_score": 0,
        "ma20_score": 0,
        "ma50_score": 0,
        "volume_score": 0,
        "macd_score": 0
    }
    
    if indicators is None:
        return {"score": 0, "breakdown": breakdown}
    
    # RSI scoring
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            breakdown["rsi_score"] = 2
            score += 2
        elif 30 <= rsi <= 70:
            breakdown["rsi_score"] = 1
            score += 1
    
    # MA20 scoring
    ma20 = indicators.get("ma20")
    if ma20 is not None and close_price > ma20:
        breakdown["ma20_score"] = 3
        score += 3
    
    # MA50 scoring
    ma50 = indicators.get("ma50")
    if ma50 is not None and close_price > ma50:
        breakdown["ma50_score"] = 2
        score += 2
    
    # Volume scoring (> 150% of average = > 50% change)
    vol_change = indicators.get("volume_change_pct")
    if vol_change is not None and vol_change > 50:
        breakdown["volume_score"] = 5
        score += 5
    
    # MACD crossover scoring
    if indicators.get("macd_crossover"):
        breakdown["macd_score"] = 3
        score += 3
    
    return {
        "score": score,
        "breakdown": breakdown
    }

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# Gemini AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Firebase
FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "")

# Scanner Configuration
# Scanner Configuration
SCANNER_MIN_PRICE = 5000  # Minimum price filter (5,000 VND)
SCANNER_MAX_PRICE = 35000 # Strict Price Cap
SCANNER_MIN_VOLUME = 1500000  # Strict Volume: 1.5M
SCANNER_TOP_N = 5  # Number of top stocks to return
SCANNER_LATENCY_WARNING_MS = 30000  # 30 seconds latency warning threshold

# T+ Scoring Weights
SCORING_CONFIG = {
    "rsi_oversold": {"condition": "rsi < 30", "score": 2},
    "rsi_safe": {"condition": "30 <= rsi <= 70", "score": 1},
    "price_above_ma20": {"condition": "close > ma20", "score": 3},
    "price_above_ma50": {"condition": "close > ma50", "score": 2},
    "volume_breakout": {"condition": "volume > 1.5 * avg_volume", "score": 5},
    "macd_cross": {"condition": "macd_cross_signal", "score": 3},
}
MINIMUM_SCORE = 8  # Minimum score to qualify

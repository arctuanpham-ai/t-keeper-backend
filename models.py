from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class LatencyStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    STALE = "stale"


class ScoreBreakdown(BaseModel):
    """Chi tiết điểm T+ cho từng tiêu chí"""
    rsi_score: int = 0
    ma20_score: int = 0
    ma50_score: int = 0
    volume_score: int = 0
    macd_score: int = 0


class TechnicalIndicators(BaseModel):
    """Các chỉ số kỹ thuật"""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    pct_change_3d: Optional[float] = None
    macd_histogram: Optional[float] = None
    ma10: Optional[float] = None
    ma20: Optional[float] = None
    ma50: Optional[float] = None
    volume_change_pct: Optional[float] = None


class StockData(BaseModel):
    """Dữ liệu cổ phiếu với data integrity fields"""
    symbol: str
    company_name: Optional[str] = None
    price: float = Field(..., description="Giá khớp lệnh - KHÔNG được làm tròn sai lệch")
    price_source: str = Field(default="vnstock", description="Nguồn dữ liệu")
    tick_time: datetime = Field(..., description="Thời gian khớp lệnh từ gói tin")
    received_time: datetime = Field(default_factory=datetime.now, description="Thời gian nhận dữ liệu")
    latency_ms: int = Field(..., description="Độ trễ tính bằng milliseconds")
    latency_status: LatencyStatus = LatencyStatus.OK
    
    # Technical data
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    avg_volume_20: Optional[int] = None
    
    # Indicators
    indicators: Optional[TechnicalIndicators] = None
    
    # T+ Score
    score: int = 0
    score_breakdown: Optional[ScoreBreakdown] = None
    

class ScannerRequest(BaseModel):
    """Request params for scanner"""
    min_price: Optional[int] = 5000
    max_price: Optional[int] = 1000000
    min_volume: Optional[int] = 100000
    top_n: Optional[int] = 5


class ScannerResponse(BaseModel):
    """Response from scanner"""
    success: bool
    timestamp: datetime
    processing_time_ms: int
    total_stocks_scanned: int
    qualified_stocks: int
    top_stocks: List[StockData]
    message: Optional[str] = None


class TradingPlan(BaseModel):
    """Kế hoạch giao dịch từ AI"""
    symbol: str
    entry_zone: str = Field(..., description="Vùng mua an toàn")
    stoploss: str = Field(..., description="Điểm cắt lỗ")
    target_1: str = Field(..., description="Mục tiêu chốt lời 1")
    target_2: str = Field(..., description="Mục tiêu chốt lời 2")
    risk_reward_ratio: Optional[str] = None
    reasoning: str = Field(..., description="Giải thích lý do")
    confidence: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.now)


class DeepDiveRequest(BaseModel):
    """Request for AI deep dive analysis"""
    symbol: str
    historical_data: Optional[List[Dict]] = None  # 30 sessions price data
    include_fundamentals: bool = True


class VisionAdvisorRequest(BaseModel):
    """Request for vision analysis"""
    image_base64: str
    context: Optional[str] = None  # Additional context about the chart


class VisionAdvisorResponse(BaseModel):
    """Response from vision advisor"""
    success: bool
    patterns_detected: List[str]
    trendlines_analysis: Optional[str] = None
    action_recommendation: str = Field(..., description="Mua ngay / Chờ chỉnh / Bán")
    confidence: str
    reasoning: str
    generated_at: datetime = Field(default_factory=datetime.now)


class PortfolioAuditRequest(BaseModel):
    """Request for Portfolio Advice"""
    symbol: str
    entry_price: float
    entry_date: datetime # ISO format
    volume: int = 0
    image_base64: Optional[str] = None # Optional chart image


class PortfolioAuditResponse(BaseModel):
    """Advice from Portfolio Guardian"""
    symbol: str
    decision: str  # BÁN / GIỮ / CHỐT LỜI
    advice_content: str  # Markdown content
    timestamp: datetime = Field(default_factory=datetime.now)


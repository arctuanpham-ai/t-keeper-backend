"""
AI Advisor Service
Uses Google Gemini for Deep Dive Analysis and Vision Analysis
"""
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Optional
import json
import base64

from config import GEMINI_API_KEY
from models import TradingPlan, VisionAdvisorResponse


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Models
model_pro = genai.GenerativeModel('gemini-1.5-pro')
model_vision = genai.GenerativeModel('gemini-1.5-flash')


async def generate_trading_plan(
    symbol: str,
    historical_data: List[Dict],
    current_price: float,
    indicators: Dict
) -> TradingPlan:
    """
    Generate AI-powered trading plan using Gemini Pro
    
    Args:
        symbol: Stock symbol (e.g., "HPG")
        historical_data: List of OHLCV data for last 30 sessions
        current_price: Current stock price
        indicators: Technical indicators (RSI, MACD, MA, etc.)
    """
    
    # Prepare context data
    price_summary = ""
    if historical_data:
        prices = [d.get('close', 0) for d in historical_data[-10:]]
        high_10d = max(prices) if prices else current_price
        low_10d = min(prices) if prices else current_price
        price_summary = f"""
        - Giá hiện tại: {current_price:,.0f} VND
        - Cao nhất 10 phiên: {high_10d:,.0f} VND
        - Thấp nhất 10 phiên: {low_10d:,.0f} VND
        """
    
    def fmt_price(val):
        if val is None or val == 'N/A': return 'N/A'
        try: return f"{float(val):,.0f}"
        except: return str(val)

    indicators_summary = f"""
    Chỉ số kỹ thuật:
    - RSI(14): {indicators.get('rsi', 'N/A')}
    - MACD: {indicators.get('macd', 'N/A')}
    - MACD Signal: {indicators.get('macd_signal', 'N/A')}
    - MA10: {fmt_price(indicators.get('ma10'))} VND
    - MA20: {fmt_price(indicators.get('ma20'))} VND
    - MA50: {fmt_price(indicators.get('ma50'))} VND
    - Thay đổi khối lượng: {indicators.get('volume_change_pct', 'N/A')}%
    """
    
    prompt = f"""
    Đóng vai chuyên gia đầu tư ngắn hạn T+ chứng khoán Việt Nam.
    
    Mã cổ phiếu: {symbol}
    
    {price_summary}
    
    {indicators_summary}
    
    NHIỆM VỤ: Lập kế hoạch giao dịch T+ cho mã {symbol} với các thông tin sau:
    
    1. **Vùng mua an toàn (Entry)**: Vùng giá khuyến nghị mua vào (ví dụ: 28,000 - 28,500)
    2. **Điểm cắt lỗ (Stoploss)**: Mức giá cắt lỗ nếu giá đi ngược (ví dụ: 27,000)
    3. **Mục tiêu chốt lời 1 (Target 1)**: Mức chốt lời đầu tiên (ví dụ: 30,000)
    4. **Mục tiêu chốt lời 2 (Target 2)**: Mức chốt lời thứ hai nếu tiếp tục tăng (ví dụ: 32,000)
    5. **Tỷ lệ Risk/Reward**: Tính toán tỷ lệ rủi ro/lợi nhuận
    6. **Giải thích ngắn gọn**: Lý do cho các mức giá đề xuất
    
    Trả lời theo format JSON sau (chỉ trả JSON, không thêm text khác):
    {{
        "entry_zone": "28,000 - 28,500",
        "stoploss": "27,000",
        "target_1": "30,000",
        "target_2": "32,000",
        "risk_reward_ratio": "1:2",
        "reasoning": "Giải thích ngắn gọn...",
        "confidence": "Cao/Trung bình/Thấp"
    }}
    """
    
    try:
        response = model_pro.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            return TradingPlan(
                symbol=symbol,
                entry_zone=data.get('entry_zone', 'N/A'),
                stoploss=data.get('stoploss', 'N/A'),
                target_1=data.get('target_1', 'N/A'),
                target_2=data.get('target_2', 'N/A'),
                risk_reward_ratio=data.get('risk_reward_ratio'),
                reasoning=data.get('reasoning', 'Không có giải thích'),
                confidence=data.get('confidence'),
                generated_at=datetime.now()
            )
    except Exception as e:
        print(f"Error generating trading plan: {e}")
        
    # Return fallback plan
    return TradingPlan(
        symbol=symbol,
        entry_zone="Không xác định",
        stoploss="Không xác định",
        target_1="Không xác định",
        target_2="Không xác định",
        reasoning=f"Lỗi khi tạo kế hoạch: AI không phản hồi",
        generated_at=datetime.now()
    )


async def analyze_chart_image(
    image_base64: str,
    context: Optional[str] = None
) -> VisionAdvisorResponse:
    """
    Analyze chart image using Gemini Vision
    
    Args:
        image_base64: Base64 encoded image data
        context: Optional additional context about the chart
    """
    
    prompt = f"""
    Bạn là chuyên gia phân tích kỹ thuật (Technical Analyst) chứng khoán.
    
    Hãy phân tích biểu đồ nến (candlestick chart) trong hình ảnh này:
    
    {f"Ngữ cảnh: {context}" if context else ""}
    
    NHIỆM VỤ:
    1. Nhận diện các mẫu hình nến (Candlestick Patterns): Hammer, Engulfing, Doji, Morning Star, v.v.
    2. Phân tích đường xu hướng (Trendline): Uptrend, Downtrend, Sideway
    3. Nhận diện các vùng hỗ trợ/kháng cự nếu thấy
    4. Đưa ra khuyến nghị hành động: MUA NGAY / CHỜ ĐIỀU CHỈNH / BÁN
    
    Trả lời theo format JSON sau (chỉ trả JSON, không thêm text khác):
    {{
        "patterns_detected": ["Hammer", "Bullish Engulfing"],
        "trendlines_analysis": "Cổ phiếu đang trong uptrend ngắn hạn...",
        "support_resistance": "Hỗ trợ: 25,000 | Kháng cự: 28,000",
        "action_recommendation": "MUA NGAY",
        "confidence": "Cao/Trung bình/Thấp",
        "reasoning": "Giải thích lý do khuyến nghị..."
    }}
    """
    
    try:
        # Prepare image data
        image_data = {
            'mime_type': 'image/png',
            'data': image_base64
        }
        
        response = model_vision.generate_content([prompt, image_data])
        response_text = response.text
        
        # Extract JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            return VisionAdvisorResponse(
                success=True,
                patterns_detected=data.get('patterns_detected', []),
                trendlines_analysis=data.get('trendlines_analysis'),
                action_recommendation=data.get('action_recommendation', 'CHỜ ĐIỀU CHỈNH'),
                confidence=data.get('confidence', 'Trung bình'),
                reasoning=data.get('reasoning', 'Không có giải thích'),
                generated_at=datetime.now()
            )
    except Exception as e:
        print(f"Error analyzing chart: {e}")
    
    # Return fallback response
    return VisionAdvisorResponse(
        success=False,
        patterns_detected=[],
        action_recommendation="KHÔNG XÁC ĐỊNH",
        confidence="Thấp",
        reasoning=f"Lỗi khi phân tích ảnh: Không thể xử lý",
        generated_at=datetime.now()
    )

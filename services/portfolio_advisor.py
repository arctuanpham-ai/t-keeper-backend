import google.generativeai as genai
from datetime import datetime, timedelta
import os
from models import PortfolioAuditRequest, PortfolioAuditResponse

# Try to get API Key from environment or hardcoded (for demo)
# Ideally this should be in .env
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

MODEL_NAME = 'gemini-pro'
VISION_MODEL_NAME = 'gemini-pro-vision'

async def audit_portfolio_item(request: PortfolioAuditRequest, current_price: float, current_rsi: float = None) -> PortfolioAuditResponse:
    """
    Core logic for Portfolio Guardian.
    Combines "Iron Rules" (Rule-based) with LLM generation.
    """
    
    # 1. Calculate Core Metrics
    now = datetime.now()
    if request.entry_date.tzinfo is None:
        # Assume entry_date is naive, make it aware (or just use naive arithmetic)
        pass # simplified

    # Calculate T+
    # Logic: T+0 (Same day), T+1, T+2.5 (Afternoon of T+2)
    # Simple approximation: Days diff
    delta_days = (now - request.entry_date).total_seconds() / (24 * 3600)
    
    t_status = "T+?"
    can_sell = False
    
    if delta_days < 0.5: t_status = "T+0"
    elif delta_days < 1.5: t_status = "T+1"
    elif delta_days < 2.5: t_status = "T+2"
    else: 
        t_status = f"T+{int(delta_days)}"
        can_sell = True
        
    if delta_days >= 2.5:
        # Check if it's afternoon of T+2 (>= 13:00)
        # Simplified: If > 2.5 days, we assume sellable
        can_sell = True
        holding_status = "Hàng Đã về"
    else:
        holding_status = "Hàng Chưa về"

    # Calculate P/L
    if request.entry_price > 0:
        profit_pct = ((current_price - request.entry_price) / request.entry_price) * 100
    else:
        profit_pct = 0

    # 2. Rule-based Logic (Iron Rules)
    decision = "QUAN SÁT"
    reason = "Chưa có tín hiệu rõ ràng."
    
    # Stop Loss / Take Profit Rules
    if profit_pct < -4:
        decision = "BÁN HẾT (STOPLOSS)"
        reason = "Vi phạm kỷ luật cắt lỗ (-4%)."
        if not can_sell:
             decision = "CẢNH BÁO (KẸP T+)"
             reason = "Lỗ nặng nhưng chưa thể bán. Chuẩn bị bán ngay khi hàng về."
    
    elif -3 <= profit_pct <= -1:
        decision = "CẢNH BÁO"
        reason = "Đang lỗ nhẹ (-1% đến -3%). Cần theo dõi sát."
        
    elif 5 <= profit_pct <= 7:
        decision = "CHỐT LỜI 1/2"
        reason = "Đạt mục tiêu T+ (5-7%). Hiện thực hóa lợi nhuận."
        
    elif profit_pct > 7:
        decision = "CHỐT LỜI / GỒNG LÃI"
        reason = "Lãi tốt (>7%). Dời điểm chặn lãi lên."
        
    elif -1 < profit_pct < 1 and delta_days > 3:
        decision = "BÁN CƠ CẤU"
        reason = "T+3 về không lãi (đi ngang). Bán để đảo sang mã khác mạnh hơn."

    # 3. Generate Advice with LLM
    # We construct a prompt with the calculated metrics
    
    system_prompt = f"""### ROLE
Bạn là "PORTFOLIO GUARDIAN" - Trợ lý quản trị rủi ro.

### INPUT DATA
- Mã: {request.symbol}
- Vị thế: {t_status} ({holding_status})
- Giá vốn: {request.entry_price:,.0f}
- Hiện tại: {current_price:,.0f}
- P/L: {profit_pct:.2f}%
- RSI: {current_rsi if current_rsi else 'N/A'}

### RULE (KỶ LUẬT THÉP)
1. Lỗ > -4% (hoặc -3% nếu xấu): Khuyên BÁN NGAY (nếu hàng về).
2. L lãi 5-7%: Khuyên CHỐT 1/2.
3. T+3 đi ngang: Khuyên BÁN CƠ CẤU.
4. Hàng chưa về: Chỉ cảnh báo, KHÔNG khuyên bán.

DỰA VÀO DỮ LIỆU TRÊN, HÃY VIẾT BÁO CÁO NGẮN GỌN (Markdown).
"""

    prompt = """
OUTPUT FORMAT:
🔔 **CẬP NHẬT DANH MỤC: {SYMBOL}**
* **Vị thế:** {T_STATUS} ({HOLDING_STATUS})
* **Hiệu suất:** {PROFIT_PCT}% (Giá vốn: {ENTRY} -> Hiện tại: {CURRENT})

📉 **PHÂN TÍCH TÌNH HUỐNG:**
* **Kỹ thuật:** [Nhận định ngắn gọn về xu hướng giá và RSI]
* **Rủi ro:** [Rủi ro lớn nhất lúc này]

🛡 **KHUYẾN NGHỊ HÀNH ĐỘNG:**
* **Lệnh:** {DECISION}
* **Lý do:** {REASON}
* **Kế hoạch tiếp theo:** [Gợi ý mốc chặn lãi/cắt lỗ tiếp theo]

Viết với giọng văn dứt khoát, chuyên nghiệp, không an ủi.
""".replace("{SYMBOL}", request.symbol)\
   .replace("{T_STATUS}", t_status)\
   .replace("{HOLDING_STATUS}", holding_status)\
   .replace("{PROFIT_PCT}", f"{profit_pct:+.2f}")\
   .replace("{ENTRY}", f"{request.entry_price:,.0f}")\
   .replace("{CURRENT}", f"{current_price:,.0f}")\
   .replace("{DECISION}", decision)\
   .replace("{REASON}", reason)

    # Call Gemini
    try:
        if not API_KEY:
            raise Exception("No API Key")
            
        model = genai.GenerativeModel(MODEL_NAME)
        response = await model.generate_content_async(system_prompt + prompt)
        advice_content = response.text
    except Exception as e:
        # Fallback if AI fails
        print(f"AI Generation failed: {e}")
        advice_content = f"""
🔔 **CẬP NHẬT DANH MỤC: {request.symbol}**
* **Vị thế:** {t_status} ({holding_status})
* **Hiệu suất:** {profit_pct:+.2f}% ({request.entry_price:,.0f} -> {current_price:,.0f})

📉 **PHÂN TÍCH TÌNH HUỐNG:**
* **Kỹ thuật:** RSI {current_rsi if current_rsi else 'N/A'}. 
* **Tự động:** Hệ thống phát hiện vi phạm điều kiện {decision}.

🛡 **KHUYẾN NGHỊ HÀNH ĐỘNG:**
* **Lệnh:** {decision}
* **Lý do:** {reason} (AI Offline Mode)
"""

    return PortfolioAuditResponse(
        symbol=request.symbol,
        decision=decision,
        advice_content=advice_content
    )

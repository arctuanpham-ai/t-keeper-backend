"""
T+ Keeper Backend API
FastAPI application with scanner, AI analysis, and data integrity endpoints
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from models import (
    ScannerRequest, ScannerResponse,
    DeepDiveRequest, TradingPlan,
    VisionAdvisorRequest, VisionAdvisorResponse
)
from services.scanner import scan_market, fetch_stock_data, process_single_stock
from services.ai_advisor import generate_trading_plan, analyze_chart_image
from services.t5_audit_report import generate_t5_audit_report
from config import API_HOST, API_PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 T+ Keeper API Starting...")
    yield
    print("👋 T+ Keeper API Shutting down...")


app = FastAPI(
    title="T+ Keeper API",
    description="API hỗ trợ đầu tư chứng khoán ngắn hạn (T+) với AI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "T+ Keeper API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "api": "ok",
            "scanner": "ok",
            "ai_advisor": "ok"
        }
    }


# ==================== SCANNER ENDPOINTS ====================

@app.post("/api/scanner/scan", response_model=ScannerResponse)
async def scan_stocks(request: ScannerRequest = None):
    """
    Legacy Endpoint (Frontend Compat) - Forward to New Engine
    """
    try:
        if request is None:
            request = ScannerRequest()
        
        # Call High-Performance Engine
        result = await scan_market(
            min_price=request.min_price,
            max_price=request.max_price,
            min_volume=request.min_volume,
            top_n=request.top_n
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan/top-opportunities", response_model=ScannerResponse)
async def get_top_opportunities(min_price: int = 5000, min_volume: int = 100000, top_n: int = 5):
    """
    High Performance Vectorized Scanner
    """
    try:
        result = await scan_market(
            min_price=min_price,
            min_volume=min_volume,
            top_n=top_n
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scanner/stock/{symbol}")
async def get_stock_detail(symbol: str):
    """
    Lấy chi tiết một mã cổ phiếu với data integrity fields
    """
    try:
        stock = process_single_stock(symbol.upper())
        if stock is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy mã {symbol}")
        return stock
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== T+5 AUDIT REPORT ENDPOINT ====================

@app.get("/api/v1/audit/t5-report/{symbol}")
async def get_t5_audit_report(symbol: str):
    """
    Sinh Báo Cáo Thẩm Định Đầu Tư T+5 đa chiều
    """
    try:
        # 1. Thu thập dữ liệu real-time và lịch sử
        # fetch_stock_data returns dict: {'df': df, 'latest': dict, 'symbol': str}
        data_bundle = fetch_stock_data(symbol.upper(), days=60)
        
        if data_bundle is None or 'df' not in data_bundle:
            # Fallback to simple processing if direct fetch fails
            stock_info = process_single_stock(symbol.upper())
            if not stock_info:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã {symbol}")
            
            price = stock_info.price
            volume = stock_info.volume or 0
            rsi = stock_info.indicators.rsi if stock_info.indicators else None
            ma20 = stock_info.indicators.ma20 if stock_info.indicators else None
            historical_df = None
        else:
            # Use data from bundle
            df = data_bundle['df']
            latest = data_bundle['latest']
            price = float(latest.get('close', 0))
            volume = int(latest.get('volume', 0))
            historical_df = df
            
            # Use calculated indicators if possible
            rsi = None
            ma20 = None
            if len(df) >= 20:
                ma20 = float(df['close'].tail(20).mean())
            
            # Simple RSI logic if we wanted to calculate it here, 
            # but generate_t5_audit_report handles None
        
        # 2. Xác định xem có phải Bank không
        banking_codes = ["VCB", "BID", "CTG", "TCB", "MBB", "ACB", "VPB", "HDB", "STB", "LPB", "SHB", "VIB", "TPB", "MSB", "OCB"]
        is_banking = symbol.upper() in banking_codes
        
        # 3. Gọi module phân tích 4 lớp
        report = generate_t5_audit_report(
            symbol=symbol.upper(),
            price=price,
            volume=volume,
            rsi=rsi,
            ma20=ma20,
            is_banking=is_banking,
            historical_df=historical_df
        )
        
        return report
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AI ADVISOR ENDPOINTS ====================

@app.post("/api/advisor/deep-dive", response_model=TradingPlan)
async def deep_dive_analysis(request: DeepDiveRequest):
    """
    Deep Dive Analysis - Lập kế hoạch đầu tư chi tiết với AI
    
    Gửi dữ liệu lịch sử 30 phiên cho Gemini Pro để phân tích
    """
    try:
        # Get stock data if not provided
        stock_data = fetch_stock_data(request.symbol.upper(), days=60)
        if stock_data is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu cho mã {request.symbol}")
        
        # Convert DataFrame to list of dicts for historical data
        df = stock_data['df']
        historical_data = df.tail(30).to_dict('records')
        
        # Get current indicators
        from services.indicators import calculate_all_indicators
        indicators = calculate_all_indicators(df) or {}
        
        result = await generate_trading_plan(
            symbol=request.symbol.upper(),
            historical_data=historical_data,
            current_price=stock_data['latest']['close'],
            indicators=indicators
        )
        if result:
             # Save to Firebase (Mock user_id for now)
             from services.firebase_service import save_trading_plan
             await save_trading_plan("demo_user", result.dict())
             
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/advisor/vision", response_model=VisionAdvisorResponse)
async def vision_advisor(request: VisionAdvisorRequest):
    """
    Vision Advisor - Phân tích biểu đồ qua ảnh
    
    User upload ảnh chụp màn hình biểu đồ từ TradingView/AmiBroker
    """
    try:
        result = await analyze_chart_image(
            image_base64=request.image_base64,
            context=request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RUN SERVER ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

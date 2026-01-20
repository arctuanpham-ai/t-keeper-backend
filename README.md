# T+ Keeper Backend

API hỗ trợ đầu tư chứng khoán ngắn hạn (T+) với AI

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Run server
python main.py
```

## API Endpoints

### Scanner
- `POST /api/scanner/scan` - Quét toàn bộ thị trường, trả về Top 5 mã T+
- `GET /api/scanner/stock/{symbol}` - Lấy chi tiết một mã cổ phiếu

### AI Advisor
- `POST /api/advisor/deep-dive` - Phân tích sâu và lập kế hoạch giao dịch
- `POST /api/advisor/vision` - Phân tích biểu đồ từ ảnh

## Data Integrity

Mỗi response đều có:
- `price_source`: Nguồn dữ liệu
- `tick_time`: Thời gian khớp lệnh
- `latency_ms`: Độ trễ (ms)
- `latency_status`: ok / warning / stale

# 🚀 Deploy T+ Keeper Backend to Render

## Bước 1: Push code lên GitHub

### Option A: Dùng Git command line
```bash
# Trong thư mục backend
cd C:\Users\TUAN\.gemini\antigravity\scratch\t_keeper_backend

# Thêm remote GitHub (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/t-keeper-api.git

# Push code
git push -u origin master
```

### Option B: Dùng GitHub Desktop
1. Mở GitHub Desktop
2. Add local repository: `C:\Users\TUAN\.gemini\antigravity\scratch\t_keeper_backend`
3. Publish to GitHub

---

## Bước 2: Deploy lên Render

1. Truy cập **https://render.com** và đăng nhập (hoặc Sign Up miễn phí)

2. Click **"New +"** → **"Web Service"**

3. Connect GitHub repo vừa push

4. Cấu hình:
   - **Name:** `t-keeper-api`
   - **Region:** Singapore (gần nhất)
   - **Branch:** `master`
   - **Runtime:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

5. Environment Variables (Add):
   - `GEMINI_API_KEY` = (API key của anh)
   - `PYTHON_VERSION` = `3.10.0`

6. Click **"Create Web Service"**

7. Đợi 3-5 phút để build và deploy

---

## Bước 3: Cập nhật Frontend

Sau khi Render deploy xong, sẽ có URL như: `https://t-keeper-api.onrender.com`

Cập nhật file `src/api.ts` trong frontend:
```typescript
const API_BASE_URL = 'https://t-keeper-api.onrender.com';
```

Rồi rebuild và deploy lại frontend:
```bash
cd t_keeper_app
npm run build
npx firebase deploy --only hosting
```

---

## ⚠️ Lưu ý Render Free Tier
- Spin down sau 15 phút không hoạt động
- First request có thể mất 30-60 giây để "cold start"
- Upgrade lên $7/month để always-on

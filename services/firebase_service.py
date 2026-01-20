import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_CREDENTIALS_PATH
import os

# Initialize Firebase
db = None

def initialize_firebase():
    global db
    try:
        if not FIREBASE_CREDENTIALS_PATH or not os.path.exists(FIREBASE_CREDENTIALS_PATH):
            print("⚠️ access local Firestore is disabled (Missing credentials)")
            return

        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("🔥 Firebase initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing Firebase: {e}")

# Initialize on module load
initialize_firebase()

async def save_trading_plan(user_id: str, plan: dict):
    """Save generated trading plan to Firestore"""
    if not db: return None
    try:
        # Save to users/{user_id}/history/{symbol}
        doc_ref = db.collection('users').document(user_id).collection('history').document(plan['symbol'])
        doc_ref.set(plan)
        return True
    except Exception as e:
        print(f"Error saving to Firebase: {e}")
        return False

async def get_user_history(user_id: str, limit: int = 20):
    """Get user's trading plan history"""
    if not db: return []
    try:
        docs = db.collection('users').document(user_id).collection('history')\
            .order_by('generated_at', direction=firestore.Query.DESCENDING)\
            .limit(limit).stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"Error reading from Firebase: {e}")
        return []

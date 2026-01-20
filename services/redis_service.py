import redis
import json
from config import REDIS_URL
import os

# Initialize Redis
redis_client = None

def initialize_redis():
    global redis_client
    try:
        # Check if Redis is available (simple connection check optional)
        pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
        redis_client = redis.Redis(connection_pool=pool)
        redis_client.ping() # Test connection
        print("⚡ Redis initialized successfully")
    except Exception as e:
        print(f"⚠️ Redis connection failed (Caching disabled): {e}")
        redis_client = None

initialize_redis()

def cache_set(key: str, value: any, ttl_seconds: int = 300):
    """Set cache with TTL"""
    if not redis_client: return
    try:
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        redis_client.setex(key, ttl_seconds, value)
    except Exception as e:
        print(f"Redis Set Error: {e}")

def cache_get(key: str):
    """Get cache"""
    if not redis_client: return None
    try:
        val = redis_client.get(key)
        if val:
            try:
                return json.loads(val)
            except:
                return val
        return None
    except Exception as e:
        print(f"Redis Get Error: {e}")
        return None

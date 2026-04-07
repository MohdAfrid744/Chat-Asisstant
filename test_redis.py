import redis

try:
    r = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )

    r.set("test_key", "Redis Working!")

    value = r.get("test_key")

    print("Redis Test:", value)

except Exception as e:
    print("Redis Error:", e)
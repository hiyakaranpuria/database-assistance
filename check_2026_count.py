from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")
db = client.ai_test_db

count_2026 = db.orders.count_documents({"orderDate": {"$gte": datetime(2026, 1, 1)}})
count_total = db.orders.count_documents({})

print(f"Total Orders: {count_total}")
print(f"Orders in 2026: {count_2026}")

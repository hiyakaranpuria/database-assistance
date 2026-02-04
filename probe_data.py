from pymongo import MongoClient
import json
from bson import json_util

client = MongoClient("mongodb://localhost:27017")
db = client.ai_test_db

print("--- ORDER SAMPLE ---")
order = db.orders.find_one()
print(json.dumps(order, default=json_util.default, indent=2))

print("\n--- PRODUCT SAMPLE ---")
product = db.products.find_one()
print(json.dumps(product, default=json_util.default, indent=2))

from pymongo import MongoClient
import json
from bson import json_util

client = MongoClient("mongodb://localhost:27017")
db = client.ai_test_db

print("Starting Pipeline Debug...")

# Step 1: Count Orders
print(f"Total Orders: {db.orders.count_documents({})}")

# Step 2: Group
pipeline_group = [
    {"$group": {"_id": "$productId", "count": {"$sum": 1}}}
]
grouped = list(db.orders.aggregate(pipeline_group))
print(f"Grouped Results (Unique Products in Orders): {len(grouped)}")

if len(grouped) > 0:
    sample_id = grouped[0]['_id']
    print(f"Sample Grouped ID: {sample_id} (Type: {type(sample_id)})")
    
    # Check if this ID exists in products
    prod = db.products.find_one({"_id": sample_id})
    print(f"Does this ID exist in products? {'YES' if prod else 'NO'}")

# Step 3: Lookup
pipeline_lookup = pipeline_group + [
    {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}}
]
looked_up = list(db.orders.aggregate(pipeline_lookup))
print(f"Lookup Results: {len(looked_up)}")

matches = sum(1 for x in looked_up if x['product'])
print(f"Lookup Matches (Non-empty product array): {matches}")

# Step 4: Unwind
pipeline_unwind = pipeline_lookup + [{"$unwind": "$product"}]
unwound = list(db.orders.aggregate(pipeline_unwind))
print(f"Unwind Results: {len(unwound)}")

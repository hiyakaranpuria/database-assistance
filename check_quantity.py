from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']
collection = db['orders']

print(f"Total Orders: {collection.count_documents({})}")

# Check data type of 'quantity'
sample = collection.find_one()
if sample:
    q_val = sample.get('quantity')
    print(f"Sample Quantity: {q_val} (Type: {type(q_val)})")

# Check if ANY order has quantity > 5 (numeric)
count_numeric = collection.count_documents({"quantity": {"$gt": 5}})
print(f"Orders with quantity > 5 (Numeric): {count_numeric}")

# Check if ANY order has quantity > "5" (String comparison)
count_string = collection.count_documents({"quantity": {"$gt": "5"}})
print(f"Orders with quantity > '5' (String): {count_string}")

# Show top 5 quantities to see what's actually there
pipeline = [
    {"$group": {"_id": "$quantity", "count": {"$sum": 1}}},
    {"$sort": {"_id": -1}},
    {"$limit": 5}
]
print("\nTop 5 Quantities in DB:")
for doc in collection.aggregate(pipeline):
    print(f"  Qty: {doc['_id']} (Count: {doc['count']})")

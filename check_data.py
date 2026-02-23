from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

print("--- DATA INSPECTION ---")
doc = db.orders.find_one({}, {"orderDate": 1, "status": 1, "amount": 1})
if doc:
    date_val = doc.get('orderDate')
    print(f"Sample OrderDate: {date_val}")
    print(f"Type of OrderDate: {type(date_val)}")
    print(f"Status: {doc.get('status')}")
    print(f"Amount: {doc.get('amount')}")
else:
    print("No orders found.")

print("\n--- DATE RANGE ---")
pipeline = [
    {
        "$group": {
            "_id": None,
            "minDate": {"$min": "$orderDate"},
            "maxDate": {"$max": "$orderDate"},
            "count": {"$sum": 1}
        }
    }
]
results = list(db.orders.aggregate(pipeline))
if results:
    print(f"Total Orders: {results[0]['count']}")
    print(f"Min Date: {results[0]['minDate']}")
    print(f"Max Date: {results[0]['maxDate']}")
else:
    print("Aggregate failed or no data.")

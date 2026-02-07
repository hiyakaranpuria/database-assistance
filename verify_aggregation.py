from pymongo import MongoClient
import json

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

print("--- TESTING AGGREGATION LOOKUP ---")

# Pipeline that SHOULD work
pipeline = [
    { 
        "$lookup": { 
            "from": "customers", 
            "localField": "customerId", 
            "foreignField": "_id", 
            "as": "customerInfo" 
        } 
    },
    { "$unwind": "$customerInfo" },
    { 
        "$match": { 
            "customerInfo.city": { "$in": ["Mumbai", "Udaipur", "mumbai", "udaipur"] } 
        } 
    },
    { "$limit": 5 },
    {
        "$project": {
            "_id": 0,
            "orderId": "$_id",
            "amount": 1,
            "city": "$customerInfo.city",
            "customer": "$customerInfo.name"
        }
    }
]

print(f"Executing pipeline:\n{json.dumps(pipeline, indent=2)}")

try:
    results = list(db.orders.aggregate(pipeline))
    print(f"\nFound {len(results)} matches:")
    for doc in results:
        print(doc)
except Exception as e:
    print(f"\n❌ ERROR: {e}")

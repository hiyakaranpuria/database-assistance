from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

collections_to_check = ['users', 'customers']

print("--- City Analysis ---")
for coll_name in collections_to_check:
    if coll_name in db.list_collection_names():
        coll = db[coll_name]
        print(f"\nCollection: {coll_name}")
        
        # Check Mumbai
        mumbai_count = coll.count_documents({"city": "Mumbai"})
        print(f"  Mumbai: {mumbai_count}")
        
        # Check Udaipur
        udaipur_count = coll.count_documents({"city": "Udaipur"})
        print(f"  Udaipur: {udaipur_count}")
        
        # Check unique cities (top 5)
        pipeline = [
            {"$group": {"_id": "$city", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        top_cities = list(coll.aggregate(pipeline))
        print(f"  Top Cities: {', '.join([f'{c['_id']} ({c['count']})' for c in top_cities])}")

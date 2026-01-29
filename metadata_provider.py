from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime

# Move connection to a config or env later
client = MongoClient("mongodb://localhost:27017")
db = client["ai_test_db"]

def infer_type(value):
    if isinstance(value, str): return "string"
    if isinstance(value, (int, float)): return "number"
    if isinstance(value, bool): return "boolean"
    if isinstance(value, datetime): return "date"
    if isinstance(value, list): return "array"
    if isinstance(value, dict): return "object"
    return "unknown"

def extract_metadata(sample_size=10): # Smaller sample is usually enough
    metadata = {}
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        field_info = {}

        # Sample documents to find fields
        for doc in collection.find().limit(sample_size):
            for field, value in doc.items():
                if field == "_id": continue
                
                ftype = infer_type(value)
                # If it's a dict, maybe just label it 'object' for now
                field_info[field] = ftype

        metadata[collection_name] = {
            "fields": field_info
        }
    return metadata
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

order_sample = db['orders'].find_one()
customer_sample = db['customers'].find_one()

print(f"--- Join Analysis ---")
print(f"Order Sample (customerId): {order_sample.get('customerId')} (Type: {type(order_sample.get('customerId'))})")
print(f"Customer Sample (_id): {customer_sample.get('_id')} (Type: {type(customer_sample.get('_id'))})")

if type(order_sample.get('customerId')) != type(customer_sample.get('_id')):
    print("\n🚨 TYPE MISMATCH DETECTED!")
    print("The 'customerId' in Orders is NOT the same type as '_id' in Customers.")
    print("Standard $lookup will FAIL. You need to convert types in the query.")
else:
    print("\n✅ Types match. Join should work.")

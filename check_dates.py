from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client.ai_test_db
min_date = list(db.orders.find().sort("orderDate", 1).limit(1))
max_date = list(db.orders.find().sort("orderDate", -1).limit(1))
print("Min Date:", min_date[0]['orderDate'] if min_date else "None")
print("Max Date:", max_date[0]['orderDate'] if max_date else "None")

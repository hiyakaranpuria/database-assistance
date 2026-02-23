from pymongo import MongoClient
from datetime import datetime

client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

print("--- 2024 MONTHLY SALES ANALYSIS ---")

pipeline = [
    {
        # Filter for the year 2024 only
        "$match": {
            "orderDate": {
                "$gte": datetime(2024, 1, 1),
                "$lt": datetime(2025, 1, 1)
            }
        }
    },
    {
        # Group by month number
        "$group": {
            "_id": {"$month": "$orderDate"},
            "totalSales": {"$sum": "$amount"},
            "orderCount": {"$sum": 1}
        }
    },
    {
        # Sort by highest sales
        "$sort": {"totalSales": -1}
    },
    {
        # Get the top month
        "$limit": 1
    },
    {
        # Format for readability
        "$project": {
            "_id": 0,
            "Month": "$_id",
            "TotalRevenue": "$totalSales",
            "TransactionCount": "$orderCount"
        }
    }
]

results = list(db.orders.aggregate(pipeline))

if results:
    res = results[0]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_name = months[res['Month'] - 1]
    print(f"🏆 Top Month: {month_name} 2024")
    print(f"💰 Total Sales: ₹{res['TotalRevenue']:,.2f}")
    print(f"📦 Orders: {res['TransactionCount']}")
else:
    print("❌ No matching data found for 2024.")

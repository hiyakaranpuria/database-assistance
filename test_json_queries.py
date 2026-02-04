#!/usr/bin/env python3
"""
Test script to validate JSON format of sample queries
"""

import json

# Test both sample queries
sample_queries = {
    "Daily Sales Last 30 Days": '''[
    {
        "$match": {
            "status": "completed"
        }
    },
    {
        "$group": {
            "_id": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$orderDate"
                }
            },
            "dailySales": {"$sum": "$amount"},
            "orderCount": {"$sum": 1}
        }
    },
    {"$sort": {"_id": 1}}
]''',
    "Product Category Analysis": '''[
    {
        "$lookup": {
            "from": "products",
            "localField": "productId",
            "foreignField": "_id",
            "as": "product"
        }
    },
    {"$unwind": "$product"},
    {
        "$lookup": {
            "from": "categories",
            "localField": "product.categoryId",
            "foreignField": "_id",
            "as": "category"
        }
    },
    {"$unwind": "$category"},
    {
        "$group": {
            "_id": "$category.name",
            "totalSales": {"$sum": "$amount"},
            "productCount": {"$addToSet": "$productId"}
        }
    },
    {
        "$project": {
            "categoryName": "$_id",
            "totalSales": 1,
            "productCount": {"$size": "$productCount"}
        }
    }
]'''
}

print("🧪 Testing Sample Queries JSON Format\n")

for name, query in sample_queries.items():
    print(f"Testing: {name}")
    try:
        parsed = json.loads(query)
        print(f"✅ Valid JSON with {len(parsed)} stages")
    except json.JSONDecodeError as e:
        print(f"❌ JSON Error: {e}")
    print("-" * 50)
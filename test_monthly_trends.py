#!/usr/bin/env python3
"""
Test script for monthly trends functionality
"""

from intelligent_query_engine import generate_intelligent_query
import json

def test_monthly_trends():
    """Test various monthly trends queries"""
    
    test_queries = [
        "show monthly trends",
        "monthly sales trends",
        "sales by month",
        "monthly patterns",
        "seasonal analysis",
        "month by month sales"
    ]
    
    print("🧪 Testing Monthly Trends Functionality\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        try:
            result = generate_intelligent_query(query)
            parsed = json.loads(result)
            print(f"✅ Generated pipeline with {len(parsed)} stages")
            print(f"First stage: {list(parsed[0].keys())[0] if parsed else 'Empty'}")
        except Exception as e:
            print(f"❌ Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_monthly_trends()
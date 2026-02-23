#!/usr/bin/env python3
"""
Test Query Coverage - Shows what works and what doesn't
"""

from simple_chat_flow import SimpleChatFlow

def test_query_types():
    """Test different types of queries"""
    
    chat = SimpleChatFlow()
    
    test_cases = [
        # ✓ SHOULD WORK - Simple queries
        {
            "category": "Simple Find",
            "questions": [
                "Show me all customers",
                "List all products",
                "Get all orders",
            ]
        },
        
        # ✓ SHOULD WORK - Filtering
        {
            "category": "Filtering",
            "questions": [
                "Show customers from New York",
                "Products with price greater than 100",
                "Orders with status completed",
            ]
        },
        
        # ✓ SHOULD WORK - Aggregations
        {
            "category": "Aggregations",
            "questions": [
                "Total revenue from all orders",
                "Count total customers",
                "Average product price",
            ]
        },
        
        # ⚠ MIGHT WORK - Date queries
        {
            "category": "Date Queries",
            "questions": [
                "Orders from 2024",
                "Customers created this year",
                "Products added last month",
            ]
        },
        
        # ⚠ MIGHT WORK - Grouping
        {
            "category": "Grouping",
            "questions": [
                "Total sales by month",
                "Count orders by status",
                "Revenue by customer",
            ]
        },
        
        # ⚠ CHALLENGING - Joins
        {
            "category": "Joins (Lookup)",
            "questions": [
                "Show orders with customer names",
                "Products with their category names",
                "Orders with payment details",
            ]
        },
        
        # ❌ LIKELY TO FAIL - Complex queries
        {
            "category": "Complex Queries",
            "questions": [
                "Top 5 customers who spent the most in the last 3 months",
                "Products that have never been ordered",
                "Monthly revenue trend with percentage change",
            ]
        },
        
        # ❌ WILL FAIL - Ambiguous queries
        {
            "category": "Ambiguous Queries",
            "questions": [
                "Show me everything",
                "What's the data?",
                "Give me some information",
            ]
        },
    ]
    
    results = {
        'passed': 0,
        'failed': 0,
        'details': []
    }
    
    print("="*70)
    print("QUERY COVERAGE TEST")
    print("="*70)
    
    for test_case in test_cases:
        category = test_case['category']
        questions = test_case['questions']
        
        print(f"\n{'='*70}")
        print(f"Category: {category}")
        print(f"{'='*70}")
        
        category_results = []
        
        for question in questions:
            print(f"\n→ Testing: {question}")
            
            try:
                response = chat.ask(question, max_retries=2)
                
                if response['success']:
                    print(f"  ✓ SUCCESS - Found {len(response['results'])} results")
                    results['passed'] += 1
                    category_results.append({
                        'question': question,
                        'status': 'PASS',
                        'query': response['query'][:100]
                    })
                else:
                    print(f"  ✗ FAILED - {response['error'][:100]}")
                    results['failed'] += 1
                    category_results.append({
                        'question': question,
                        'status': 'FAIL',
                        'error': response['error'][:100]
                    })
                    
            except Exception as e:
                print(f"  ✗ ERROR - {str(e)[:100]}")
                results['failed'] += 1
                category_results.append({
                    'question': question,
                    'status': 'ERROR',
                    'error': str(e)[:100]
                })
        
        results['details'].append({
            'category': category,
            'results': category_results
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total Passed: {results['passed']}")
    print(f"Total Failed: {results['failed']}")
    print(f"Success Rate: {results['passed'] / (results['passed'] + results['failed']) * 100:.1f}%")
    
    # Detailed breakdown
    print(f"\n{'='*70}")
    print("BREAKDOWN BY CATEGORY")
    print(f"{'='*70}")
    
    for detail in results['details']:
        category = detail['category']
        category_results = detail['results']
        
        passed = sum(1 for r in category_results if r['status'] == 'PASS')
        total = len(category_results)
        
        print(f"\n{category}: {passed}/{total} passed")
        
        for result in category_results:
            status_icon = "✓" if result['status'] == 'PASS' else "✗"
            print(f"  {status_icon} {result['question']}")
    
    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    
    chat = SimpleChatFlow()
    
    print("\n" + "="*70)
    print("EDGE CASES TEST")
    print("="*70)
    
    edge_cases = [
        ("Empty query", ""),
        ("Non-existent collection", "Show me all unicorns"),
        ("Non-existent field", "Show customers with salary greater than 1000"),
        ("SQL syntax", "SELECT * FROM customers"),
        ("Invalid operation", "DELETE all orders"),
        ("Very long question", "Show me all the customers who have placed orders in the last 6 months and have spent more than $1000 and live in cities starting with 'New' and have email addresses ending with '.com' and were created after January 2023"),
    ]
    
    for test_name, question in edge_cases:
        print(f"\n→ Testing: {test_name}")
        print(f"  Question: {question[:100]}")
        
        try:
            response = chat.ask(question, max_retries=1)
            
            if response['success']:
                print(f"  ✓ Handled gracefully")
            else:
                print(f"  ✓ Failed gracefully: {response['error'][:80]}")
                
        except Exception as e:
            print(f"  ✗ Crashed: {str(e)[:80]}")


if __name__ == "__main__":
    print("Starting comprehensive query coverage test...")
    print("This will test various query types to see what works and what doesn't.\n")
    
    # Test different query types
    results = test_query_types()
    
    # Test edge cases
    test_edge_cases()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("✓ Simple queries (find, filter) work well")
    print("✓ Basic aggregations (sum, count, avg) work well")
    print("⚠ Date queries need proper format")
    print("⚠ Joins ($lookup) are challenging")
    print("❌ Very complex multi-stage pipelines may fail")
    print("❌ Ambiguous questions will fail")
    print("\nRecommendation: Use clear, specific questions with explicit field names")

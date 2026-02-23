#!/usr/bin/env python3
"""
Test the simplified chat flow
"""

from simple_chat_flow import SimpleChatFlow

def test_flow():
    """Test the simplified flow with sample questions"""
    
    print("Testing Simplified Chat Flow")
    print("="*60)
    
    # Initialize
    chat = SimpleChatFlow()
    
    # Test questions
    test_questions = [
        "Show me all customers",
        "What is the total revenue from orders?",
        "List products with low stock",
        "Show orders from 2024",
        "Count total customers"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Testing: {question}")
        print('='*60)
        
        response = chat.ask(question)
        
        if response['success']:
            print(f"\n✓ SUCCESS")
            print(f"Query: {response['query']}")
            print(f"Results: {len(response['results'])} documents")
            print(f"\nFirst result: {response['results'][0] if response['results'] else 'None'}")
        else:
            print(f"\n✗ FAILED")
            print(f"Error: {response['error']}")
        
        print()

if __name__ == "__main__":
    test_flow()

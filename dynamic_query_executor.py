import json
from bson import json_util
from pymongo import MongoClient
from metadata_provider import extract_metadata

class DynamicQueryExecutor:
    def __init__(self, connection_string="mongodb://localhost:27017", db_name="ai_test_db"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.metadata = extract_metadata()
        
    def execute_query(self, query_str, user_question=""):
        """Execute query with automatic collection detection"""
        try:
            # Safety check
            forbidden = ["drop", "delete", "remove", "$out", "$merge"]
            if any(word in query_str.lower() for word in forbidden):
                return "Error: Destructive operations are not allowed."

            # Parse query
            query_data = json.loads(query_str, object_hook=json_util.object_hook)
            
            # Determine collection
            collection_name = self._determine_collection(user_question, query_str)
            
            # Execute query
            collection = self.db[collection_name]
            results = list(collection.aggregate(query_data))
            
            return {
                'results': results if results else [],
                'collection': collection_name,
                'query': query_data
            }
            
        except Exception as e:
            return f"Execution Error: {str(e)}"
    
    def _determine_collection(self, question, query_str):
        """Intelligently determine which collection to query"""
        question_lower = question.lower()
        query_lower = query_str.lower()
        
        # Sales/revenue queries should use orders collection
        if any(word in question_lower for word in ['sales', 'selling', 'revenue', 'total sales', 'income', 'orders', 'spending']):
            if 'orders' in self.metadata:
                return 'orders'
        
        # Direct collection name mentions
        for collection in self.metadata.keys():
            if collection.lower() in question_lower:
                return collection
        
        # Field-based detection - prioritize orders for amount fields
        if 'amount' in query_lower or '$sum' in query_lower:
            if 'orders' in self.metadata:
                return 'orders'
        
        # Check other field matches
        for collection, info in self.metadata.items():
            fields = info.get('fields', {})
            
            # Check if query or question mentions fields from this collection
            for field in fields.keys():
                if field.lower() in question_lower or field.lower() in query_lower:
                    return collection
        
        # Intent-based fallback - prefer orders for financial queries
        if any(word in question_lower for word in ['sales', 'revenue', 'order', 'purchase', 'total', 'amount']):
            if 'orders' in self.metadata:
                return 'orders'
            # Look for collections with amount/price fields
            for collection, info in self.metadata.items():
                fields = info.get('fields', {})
                if any(field in ['amount', 'price', 'total', 'cost'] for field in fields.keys()):
                    return collection
        
        # Default to orders if available, otherwise first collection
        if 'orders' in self.metadata:
            return 'orders'
        return list(self.metadata.keys())[0] if self.metadata else 'orders'

# Global instance
executor = DynamicQueryExecutor()

def execute_mongo_query(query_str, user_question=""):
    """Execute query with dynamic collection detection"""
    result = executor.execute_query(query_str, user_question)
    
    # Return just results for backward compatibility
    if isinstance(result, dict) and 'results' in result:
        return result['results']
    return result
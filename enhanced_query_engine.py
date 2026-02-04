#!/usr/bin/env python3
"""
Enhanced Query Engine - Improved natural language processing with Local LLM support
"""

import re
import json
from datetime import datetime, timedelta
from dynamic_query_generator import DynamicQueryGenerator

# Try to import local LLM integration
try:
    from llm_integration import generate_llm_query, LLM_AVAILABLE
    print("✅ Local LLM integration loaded")
except ImportError:
    print("⚠️  Local LLM integration not available")
    LLM_AVAILABLE = False

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"$date": obj.isoformat() + "Z"}
        return super().default(obj)

class EnhancedQueryEngine(DynamicQueryGenerator):
    def __init__(self):
        super().__init__()
        self.business_synonyms = {
            'revenue': ['sales', 'income', 'earnings', 'turnover'],
            'customers': ['clients', 'buyers', 'users', 'consumers'],
            'products': ['items', 'goods', 'merchandise', 'inventory'],
            'orders': ['purchases', 'transactions', 'bookings']
        }
        
        self.time_patterns = {
            'this year': self._get_current_year_range(),
            'last year': self._get_last_year_range(),
            'this month': self._get_current_month_range(),
            'last month': self._get_last_month_range(),
            'recently': self._get_recent_range()
        }
    
    def _get_current_year_range(self):
        now = datetime.now()
        return {
            'start': datetime(now.year, 1, 1),
            'end': now
        }
    
    def _get_last_year_range(self):
        now = datetime.now()
        return {
            'start': datetime(now.year - 1, 1, 1),
            'end': datetime(now.year, 1, 1)
        }
    
    def _get_current_month_range(self):
        now = datetime.now()
        return {
            'start': datetime(now.year, now.month, 1),
            'end': now
        }
    
    def _get_last_month_range(self):
        now = datetime.now()
        if now.month == 1:
            start = datetime(now.year - 1, 12, 1)
            end = datetime(now.year, 1, 1)
        else:
            start = datetime(now.year, now.month - 1, 1)
            end = datetime(now.year, now.month, 1)
        return {'start': start, 'end': end}
    
    def _get_recent_range(self):
        now = datetime.now()
        return {
            'start': now - timedelta(days=90),
            'end': now
        }
    
    def enhance_question(self, question):
        """Enhance question with business intelligence"""
        question = question.lower().strip()
        
        # Expand synonyms
        for canonical, synonyms in self.business_synonyms.items():
            for synonym in synonyms:
                if synonym in question and canonical not in question:
                    question = question.replace(synonym, canonical)
        
        # Add smart defaults
        if any(word in question for word in ['sales', 'revenue']) and not self._has_time_reference(question):
            question += ' this year'
        
        # Handle churn queries
        if any(phrase in question for phrase in ['haven\'t ordered', 'inactive', 'lost customers']):
            question = 'customers who haven\'t ordered recently'
        
        return question
    
    def _has_time_reference(self, question):
        time_words = ['year', 'month', 'week', 'day', 'recently', 'last', 'this', '2023', '2024', '2025']
        return any(word in question for word in time_words)
    
    def generate_enhanced_query(self, question):
        """Generate query with enhanced understanding and LLM support"""
        
        # Try local LLM first if available
        if LLM_AVAILABLE:
            try:
                llm_query = generate_llm_query(question, use_multilingual=True)
                print(f"🤖 Using Local LLM for query generation")
                return llm_query, "orders" # Return tuple to match expected signature
            except Exception as e:
                print(f"⚠️  LLM failed, using fallback: {e}")
        
        # Fallback to enhanced rule-based system
        enhanced_question = self.enhance_question(question)
        
        # Handle special business cases
        if 'haven\'t ordered recently' in enhanced_question:
            return self._generate_churn_query()
        
        elif 'seasonal' in enhanced_question or 'monthly pattern' in enhanced_question:
            return self._generate_seasonal_query()
        
        elif any(word in enhanced_question for word in ['least', 'worst']) and 'product' in enhanced_question:
            return self._generate_worst_products_query()
        
        else:
            # Use the parent class method with enhancements
            return super().generate_query(enhanced_question)
    
    def _generate_churn_query(self):
        """Generate churn analysis query"""
        cutoff_date = datetime.now() - timedelta(days=30)  # Changed to 30 days for demo data
        
        pipeline = [
            {
                "$match": {
                    "status": "completed"
                }
            },
            {
                "$group": {
                    "_id": "$customerId",
                    "lastOrderDate": {"$max": "$orderDate"},
                    "totalOrders": {"$sum": 1},
                    "totalSpent": {"$sum": "$amount"}
                }
            },
            {
                "$match": {
                    "lastOrderDate": {"$lt": cutoff_date}
                }
            },
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "customer"
                }
            },
            {"$unwind": "$customer"},
            {
                "$project": {
                    "customerName": "$customer.name",
                    "customerEmail": "$customer.email",
                    "lastOrderDate": 1,
                    "totalOrders": 1,
                    "totalSpent": 1
                }
            },
            {"$sort": {"lastOrderDate": 1}},
            {"$limit": 10}
        ]
        
        return json.dumps(pipeline, cls=DateTimeEncoder), "orders"
    
    def _generate_seasonal_query(self):
        """Generate seasonal analysis query"""
        pipeline = [
            {
                "$match": {
                    "status": "completed",
                    "orderDate": {
                        "$gte": datetime.now() - timedelta(days=365)
                    }
                }
            },
            {
                "$group": {
                    "_id": {"$month": "$orderDate"},
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1}
                }
            },
            {
                "$project": {
                    "month": "$_id",
                    "totalSales": 1,
                    "orderCount": 1,
                    "monthName": {
                        "$switch": {
                            "branches": [
                                {"case": {"$eq": ["$_id", 1]}, "then": "January"},
                                {"case": {"$eq": ["$_id", 2]}, "then": "February"},
                                {"case": {"$eq": ["$_id", 3]}, "then": "March"},
                                {"case": {"$eq": ["$_id", 4]}, "then": "April"},
                                {"case": {"$eq": ["$_id", 5]}, "then": "May"},
                                {"case": {"$eq": ["$_id", 6]}, "then": "June"},
                                {"case": {"$eq": ["$_id", 7]}, "then": "July"},
                                {"case": {"$eq": ["$_id", 8]}, "then": "August"},
                                {"case": {"$eq": ["$_id", 9]}, "then": "September"},
                                {"case": {"$eq": ["$_id", 10]}, "then": "October"},
                                {"case": {"$eq": ["$_id", 11]}, "then": "November"},
                                {"case": {"$eq": ["$_id", 12]}, "then": "December"}
                            ],
                            "default": "Unknown"
                        }
                    }
                }
            },
            {"$sort": {"month": 1}}
        ]
        
        return json.dumps(pipeline, cls=DateTimeEncoder), "orders"

# Global instance
enhanced_engine = EnhancedQueryEngine()

def generate_enhanced_query(question):
    """Main function to generate enhanced queries"""
    try:
        query, collection = enhanced_engine.generate_enhanced_query(question)
        return query
    except Exception as e:
        # Fallback to simple query
        return json.dumps([{"$limit": 10}], cls=DateTimeEncoder)  
        
    def generate_narrative(self, question, data):
        """Generate natural language narrative using LLM"""
        try:
            from llm_integration import local_llm
            return local_llm.synthesize_answer(question, data)
        except Exception:
            return None
  
    def _generate_worst_products_query(self):
        """Generate worst performing products query"""
        pipeline = [
            {
                "$match": {
                    "status": "completed"
                }
            },
            {
                "$group": {
                    "_id": "$productId",
                    "totalSales": {"$sum": "$amount"},
                    "quantitySold": {"$sum": "$quantity"},
                    "orderCount": {"$sum": 1}
                }
            },
            {
                "$lookup": {
                    "from": "products",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "product"
                }
            },
            {"$unwind": "$product"},
            {
                "$project": {
                    "productName": "$product.name",
                    "totalSales": 1,
                    "quantitySold": 1,
                    "orderCount": 1
                }
            },
            {"$sort": {"totalSales": 1}},  # Ascending for worst
            {"$limit": 10}
        ]
        
        return json.dumps(pipeline, cls=DateTimeEncoder), "orders"
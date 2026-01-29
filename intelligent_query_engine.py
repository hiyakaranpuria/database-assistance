#!/usr/bin/env python3
"""
Intelligent Query Engine - Enhanced Natural Language Processing for Database Queries
Handles complex business questions with context awareness and smart suggestions
"""

import re
import json
from datetime import datetime, timedelta
from collections import defaultdict
from metadata_provider import extract_metadata
from pymongo import MongoClient

class BusinessLogicEngine:
    def __init__(self):
        self.business_synonyms = {
            'revenue': ['sales', 'income', 'earnings', 'turnover', 'money'],
            'customers': ['clients', 'buyers', 'users', 'consumers', 'people'],
            'products': ['items', 'goods', 'merchandise', 'inventory', 'stuff'],
            'profit': ['margin', 'earnings', 'net income', 'gains'],
            'orders': ['purchases', 'transactions', 'sales', 'bookings'],
            'performance': ['results', 'metrics', 'stats', 'numbers']
        }
        
        self.business_metrics = {
            'conversion_rate': {'formula': 'orders / customers * 100', 'unit': '%'},
            'average_order_value': {'formula': 'total_revenue / total_orders', 'unit': '$'},
            'churn_rate': {'formula': 'inactive_customers / total_customers * 100', 'unit': '%'},
            'retention_rate': {'formula': 'returning_customers / total_customers * 100', 'unit': '%'}
        }
        
        self.date_patterns = {
            'today': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            'yesterday': datetime.now() - timedelta(days=1),
            'last week': datetime.now() - timedelta(days=7),
            'last month': datetime.now() - timedelta(days=30),
            'last quarter': datetime.now() - timedelta(days=90),
            'last year': datetime.now() - timedelta(days=365),
            'this week': self._get_week_start(),
            'this month': self._get_month_start(),
            'this quarter': self._get_quarter_start(),
            'this year': self._get_year_start(),
            'year to date': self._get_year_start(),
            'ytd': self._get_year_start()
        }
    
    def _get_week_start(self):
        today = datetime.now()
        return today - timedelta(days=today.weekday())
    
    def _get_month_start(self):
        today = datetime.now()
        return today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _get_quarter_start(self):
        today = datetime.now()
        quarter_month = ((today.month - 1) // 3) * 3 + 1
        return today.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _get_year_start(self):
        today = datetime.now()
        return today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

class ConversationContext:
    def __init__(self):
        self.context_stack = []
        self.current_filters = {}
        self.last_collection = None
        self.last_timeframe = None
    
    def add_context(self, question, query, results):
        self.context_stack.append({
            'question': question.lower(),
            'query': query,
            'results': results,
            'timestamp': datetime.now()
        })
        
        # Keep only last 5 interactions
        if len(self.context_stack) > 5:
            self.context_stack.pop(0)
    
    def get_context_clues(self, question):
        """Extract context from previous questions"""
        clues = {}
        
        if not self.context_stack:
            return clues
        
        last_interaction = self.context_stack[-1]
        
        # Handle reference words
        if any(word in question.lower() for word in ['that', 'those', 'them', 'it']):
            clues['reference'] = last_interaction
        
        # Handle comparative words
        if any(word in question.lower() for word in ['compare', 'vs', 'versus', 'against']):
            clues['comparison'] = last_interaction
        
        # Handle time references
        if any(word in question.lower() for word in ['previous', 'before', 'earlier']):
            clues['time_shift'] = 'previous'
        
        return clues

class IntelligentQueryEngine:
    def __init__(self):
        self.metadata = extract_metadata()
        self.business_logic = BusinessLogicEngine()
        self.context = ConversationContext()
        self.client = MongoClient("mongodb://localhost:27017")
        self.db = self.client["ai_test_db"]
    
    def normalize_question(self, question):
        """Normalize and enhance the question with business intelligence"""
        question = question.lower().strip()
        
        # Expand business synonyms
        for canonical, synonyms in self.business_logic.business_synonyms.items():
            for synonym in synonyms:
                if synonym in question and canonical not in question:
                    question = question.replace(synonym, canonical)
        
        # Apply smart defaults
        question = self._apply_smart_defaults(question)
        
        # Resolve context references
        question = self._resolve_context_references(question)
        
        return question
    
    def _apply_smart_defaults(self, question):
        """Apply intelligent defaults based on business logic"""
        
        # Default time periods for different query types
        if any(word in question for word in ['sales', 'revenue', 'orders']) and not self._has_time_reference(question):
            question += ' this year'
        
        # Default to completed orders for sales queries
        if any(word in question for word in ['sales', 'revenue']) and 'status' not in question:
            question += ' completed orders'
        
        # Add performance context for comparison queries
        if any(word in question for word in ['best', 'worst', 'top', 'bottom']) and 'product' in question:
            if 'by' not in question:
                question += ' by sales'
        
        return question
    
    def _has_time_reference(self, question):
        """Check if question already has time reference"""
        time_words = ['year', 'month', 'week', 'day', 'quarter', 'today', 'yesterday', 'last', 'this', 'ytd', '2023', '2024', '2025']
        return any(word in question for word in time_words)
    
    def _resolve_context_references(self, question):
        """Resolve references to previous questions"""
        context_clues = self.context.get_context_clues(question)
        
        if 'reference' in context_clues:
            # Replace "that" with specific reference
            last_q = context_clues['reference']['question']
            if 'that' in question:
                # Extract the subject from last question
                subject = self._extract_subject(last_q)
                question = question.replace('that', subject)
        
        return question
    
    def _extract_subject(self, question):
        """Extract the main subject from a question"""
        if 'product' in question:
            return 'products'
        elif 'customer' in question:
            return 'customers'
        elif 'sales' in question or 'revenue' in question:
            return 'sales'
        elif 'order' in question:
            return 'orders'
        else:
            return 'data'
    
    def analyze_intent(self, question):
        """Advanced intent analysis with business logic"""
        question = question.lower()
        
        # Complex business intents
        if any(phrase in question for phrase in ['churn', 'inactive customers', 'lost customers']):
            return 'churn_analysis'
        
        if any(phrase in question for phrase in ['lifetime value', 'ltv', 'customer value']):
            return 'ltv_analysis'
        
        if any(phrase in question for phrase in ['trending', 'growing', 'declining', 'trend', 'trends']):
            return 'trend_analysis'
        
        if any(phrase in question for phrase in ['seasonal', 'monthly pattern', 'by month', 'monthly trends', 'monthly', 'month by month']):
            return 'seasonal_analysis'
        
        if any(phrase in question for phrase in ['compare', 'vs', 'versus', 'against']):
            return 'comparison_analysis'
        
        # Performance analysis
        if any(word in question for word in ['best', 'top', 'highest', 'most']):
            return 'top_performance'
        
        if any(word in question for word in ['worst', 'bottom', 'lowest', 'least']):
            return 'bottom_performance'
        
        # Aggregation intents
        if any(word in question for word in ['total', 'sum', 'revenue', 'sales']):
            return 'aggregate_sum'
        
        if any(word in question for word in ['average', 'mean', 'avg']):
            return 'aggregate_avg'
        
        if any(word in question for word in ['count', 'how many', 'number of']):
            return 'count'
        
        # Default to list
        return 'list'
    
    def generate_advanced_query(self, question):
        """Generate MongoDB query with advanced business logic"""
        normalized_question = self.normalize_question(question)
        intent = self.analyze_intent(normalized_question)
        
        # Extract entities and filters
        entities = self._extract_entities(normalized_question)
        time_filter = self._extract_time_filter(normalized_question)
        
        # Generate query based on intent
        if intent == 'churn_analysis':
            return self._generate_churn_query(time_filter)
        
        elif intent == 'ltv_analysis':
            return self._generate_ltv_query(entities, time_filter)
        
        elif intent == 'trend_analysis':
            return self._generate_trend_query(entities, time_filter)
        
        elif intent == 'seasonal_analysis':
            return self._generate_seasonal_query(entities)
        
        elif intent == 'comparison_analysis':
            return self._generate_comparison_query(normalized_question, entities, time_filter)
        
        elif intent in ['top_performance', 'bottom_performance']:
            return self._generate_performance_query(intent, entities, time_filter)
        
        elif intent == 'aggregate_sum':
            return self._generate_sum_query(entities, time_filter)
        
        elif intent == 'aggregate_avg':
            return self._generate_avg_query(entities, time_filter)
        
        elif intent == 'count':
            return self._generate_count_query(entities, time_filter)
        
        else:
            return self._generate_list_query(entities, time_filter)
    
    def _extract_entities(self, question):
        """Extract business entities from question"""
        entities = {
            'target': None,
            'groupby': None,
            'filter_field': None,
            'filter_value': None
        }
        
        # Target entity
        if 'product' in question:
            entities['target'] = 'products'
        elif 'customer' in question:
            entities['target'] = 'customers'
        elif 'order' in question:
            entities['target'] = 'orders'
        elif 'sales' in question or 'revenue' in question:
            entities['target'] = 'sales'
        
        # Group by entity
        if 'by city' in question or 'by location' in question:
            entities['groupby'] = 'city'
        elif 'by category' in question:
            entities['groupby'] = 'category'
        elif 'by month' in question:
            entities['groupby'] = 'month'
        elif 'by product' in question:
            entities['groupby'] = 'product'
        
        # Filters
        cities = ['mumbai', 'delhi', 'jaipur', 'udaipur', 'bangalore']
        for city in cities:
            if city in question:
                entities['filter_field'] = 'city'
                entities['filter_value'] = city.title()
        
        return entities
    
    def _extract_time_filter(self, question):
        """Extract time filters with intelligent parsing"""
        time_filter = {}
        
        # Check for specific date patterns
        for pattern, date_obj in self.business_logic.date_patterns.items():
            if pattern in question:
                if pattern in ['this year', 'year to date', 'ytd']:
                    time_filter['start'] = date_obj
                    time_filter['end'] = datetime.now()
                elif pattern in ['last year']:
                    time_filter['start'] = date_obj
                    time_filter['end'] = self.business_logic._get_year_start()
                elif pattern in ['this month']:
                    time_filter['start'] = date_obj
                    time_filter['end'] = datetime.now()
                elif pattern in ['last month']:
                    time_filter['start'] = date_obj
                    time_filter['end'] = self.business_logic._get_month_start()
                else:
                    time_filter['start'] = date_obj
                    time_filter['end'] = datetime.now()
                break
        
        # Check for specific years
        year_match = re.search(r'20\d{2}', question)
        if year_match:
            year = int(year_match.group())
            time_filter['start'] = datetime(year, 1, 1)
            time_filter['end'] = datetime(year + 1, 1, 1)
        
        return time_filter
    
    def _generate_churn_query(self, time_filter):
        """Generate query for churn analysis"""
        # Customers who haven't ordered in last 90 days
        cutoff_date = datetime.now() - timedelta(days=90)
        
        pipeline = [
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
                    "totalSpent": 1,
                    "daysSinceLastOrder": {
                        "$divide": [
                            {"$subtract": [datetime.now(), "$lastOrderDate"]},
                            86400000  # milliseconds in a day
                        ]
                    }
                }
            },
            {"$sort": {"daysSinceLastOrder": -1}},
            {"$limit": 20}
        ]
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_performance_query(self, intent, entities, time_filter):
        """Generate performance analysis queries"""
        sort_order = 1 if intent == 'bottom_performance' else -1
        
        if entities['target'] == 'products':
            pipeline = [
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
                        "orderCount": 1,
                        "avgOrderValue": {"$divide": ["$totalSales", "$orderCount"]}
                    }
                },
                {"$sort": {"totalSales": sort_order}},
                {"$limit": 10}
            ]
        
        elif entities['target'] == 'customers':
            pipeline = [
                {
                    "$group": {
                        "_id": "$customerId",
                        "totalSpent": {"$sum": "$amount"},
                        "orderCount": {"$sum": 1},
                        "avgOrderValue": {"$avg": "$amount"}
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
                        "customerCity": "$customer.city",
                        "totalSpent": 1,
                        "orderCount": 1,
                        "avgOrderValue": 1
                    }
                },
                {"$sort": {"totalSpent": sort_order}},
                {"$limit": 10}
            ]
        
        else:
            # Default to sales by city
            pipeline = [
                {
                    "$lookup": {
                        "from": "customers",
                        "localField": "customerId",
                        "foreignField": "_id",
                        "as": "customer"
                    }
                },
                {"$unwind": "$customer"},
                {
                    "$group": {
                        "_id": "$customer.city",
                        "totalSales": {"$sum": "$amount"},
                        "orderCount": {"$sum": 1},
                        "customerCount": {"$addToSet": "$customerId"}
                    }
                },
                {
                    "$project": {
                        "city": "$_id",
                        "totalSales": 1,
                        "orderCount": 1,
                        "customerCount": {"$size": "$customerCount"}
                    }
                },
                {"$sort": {"totalSales": sort_order}},
                {"$limit": 10}
            ]
        
        # Add time filter if specified
        if time_filter:
            match_stage = {"$match": {}}
            if 'start' in time_filter:
                match_stage["$match"]["orderDate"] = {
                    "$gte": time_filter['start'],
                    "$lt": time_filter.get('end', datetime.now())
                }
            pipeline.insert(0, match_stage)
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_seasonal_query(self, entities):
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
                    "_id": {
                        "month": {"$month": "$orderDate"},
                        "year": {"$year": "$orderDate"}
                    },
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "avgOrderValue": {"$avg": "$amount"}
                }
            },
            {
                "$project": {
                    "month": "$_id.month",
                    "year": "$_id.year",
                    "totalSales": 1,
                    "orderCount": 1,
                    "avgOrderValue": 1,
                    "monthName": {
                        "$switch": {
                            "branches": [
                                {"case": {"$eq": ["$_id.month", 1]}, "then": "January"},
                                {"case": {"$eq": ["$_id.month", 2]}, "then": "February"},
                                {"case": {"$eq": ["$_id.month", 3]}, "then": "March"},
                                {"case": {"$eq": ["$_id.month", 4]}, "then": "April"},
                                {"case": {"$eq": ["$_id.month", 5]}, "then": "May"},
                                {"case": {"$eq": ["$_id.month", 6]}, "then": "June"},
                                {"case": {"$eq": ["$_id.month", 7]}, "then": "July"},
                                {"case": {"$eq": ["$_id.month", 8]}, "then": "August"},
                                {"case": {"$eq": ["$_id.month", 9]}, "then": "September"},
                                {"case": {"$eq": ["$_id.month", 10]}, "then": "October"},
                                {"case": {"$eq": ["$_id.month", 11]}, "then": "November"},
                                {"case": {"$eq": ["$_id.month", 12]}, "then": "December"}
                            ],
                            "default": "Unknown"
                        }
                    }
                }
            },
            {"$sort": {"year": 1, "month": 1}}
        ]
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_ltv_query(self, entities, time_filter):
        """Generate customer lifetime value query"""
        pipeline = [
            {
                "$group": {
                    "_id": "$customerId",
                    "totalSpent": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "firstOrder": {"$min": "$orderDate"},
                    "lastOrder": {"$max": "$orderDate"}
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
                    "customerCity": "$customer.city",
                    "totalSpent": 1,
                    "orderCount": 1,
                    "avgOrderValue": {"$divide": ["$totalSpent", "$orderCount"]},
                    "customerLifespanDays": {
                        "$divide": [
                            {"$subtract": ["$lastOrder", "$firstOrder"]},
                            86400000
                        ]
                    }
                }
            },
            {"$sort": {"totalSpent": -1}},
            {"$limit": 20}
        ]
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_trend_query(self, entities, time_filter):
        """Generate trend analysis query"""
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
                    "_id": {
                        "year": {"$year": "$orderDate"},
                        "month": {"$month": "$orderDate"}
                    },
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "avgOrderValue": {"$avg": "$amount"}
                }
            },
            {
                "$project": {
                    "year": "$_id.year",
                    "month": "$_id.month",
                    "totalSales": 1,
                    "orderCount": 1,
                    "avgOrderValue": 1,
                    "period": {
                        "$concat": [
                            {"$toString": "$_id.year"},
                            "-",
                            {"$toString": "$_id.month"}
                        ]
                    }
                }
            },
            {"$sort": {"year": 1, "month": 1}}
        ]
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_comparison_query(self, question, entities, time_filter):
        """Generate comparison analysis query"""
        # Default comparison by city
        pipeline = [
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "customerId",
                    "foreignField": "_id",
                    "as": "customer"
                }
            },
            {"$unwind": "$customer"},
            {
                "$group": {
                    "_id": "$customer.city",
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "avgOrderValue": {"$avg": "$amount"},
                    "customerCount": {"$addToSet": "$customerId"}
                }
            },
            {
                "$project": {
                    "city": "$_id",
                    "totalSales": 1,
                    "orderCount": 1,
                    "avgOrderValue": 1,
                    "customerCount": {"$size": "$customerCount"}
                }
            },
            {"$sort": {"totalSales": -1}}
        ]
        
        if time_filter:
            match_stage = {"$match": {"status": "completed"}}
            if 'start' in time_filter:
                match_stage["$match"]["orderDate"] = {
                    "$gte": time_filter['start'],
                    "$lt": time_filter.get('end', datetime.now())
                }
            pipeline.insert(0, match_stage)
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_avg_query(self, entities, time_filter):
        """Generate average aggregation query"""
        pipeline = [{"$match": {"status": "completed"}}]
        
        # Add time filter
        if time_filter:
            pipeline[0]["$match"]["orderDate"] = {
                "$gte": time_filter['start'],
                "$lt": time_filter.get('end', datetime.now())
            }
        
        # Add grouping
        if entities['groupby'] == 'city':
            pipeline.extend([
                {
                    "$lookup": {
                        "from": "customers",
                        "localField": "customerId",
                        "foreignField": "_id",
                        "as": "customer"
                    }
                },
                {"$unwind": "$customer"},
                {
                    "$group": {
                        "_id": "$customer.city",
                        "avgOrderValue": {"$avg": "$amount"},
                        "orderCount": {"$sum": 1}
                    }
                },
                {"$sort": {"avgOrderValue": -1}}
            ])
        else:
            pipeline.append({
                "$group": {
                    "_id": None,
                    "avgOrderValue": {"$avg": "$amount"},
                    "orderCount": {"$sum": 1}
                }
            })
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_sum_query(self, entities, time_filter):
        """Generate sum aggregation query"""
        pipeline = [{"$match": {"status": "completed"}}]
        
        # Add time filter
        if time_filter:
            pipeline[0]["$match"]["orderDate"] = {
                "$gte": time_filter['start'],
                "$lt": time_filter.get('end', datetime.now())
            }
        
        # Add grouping
        if entities['groupby'] == 'city':
            pipeline.extend([
                {
                    "$lookup": {
                        "from": "customers",
                        "localField": "customerId",
                        "foreignField": "_id",
                        "as": "customer"
                    }
                },
                {"$unwind": "$customer"},
                {
                    "$group": {
                        "_id": "$customer.city",
                        "totalSales": {"$sum": "$amount"},
                        "orderCount": {"$sum": 1}
                    }
                },
                {"$sort": {"totalSales": -1}}
            ])
        else:
            pipeline.append({
                "$group": {
                    "_id": None,
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1}
                }
            })
        
        return json.dumps(pipeline, default=str), "orders"
    
    def _generate_count_query(self, entities, time_filter):
        """Generate count query"""
        if entities['target'] == 'customers':
            pipeline = [{"$count": "total"}]
            return json.dumps(pipeline, default=str), "customers"
        
        elif entities['target'] == 'products':
            pipeline = [{"$count": "total"}]
            return json.dumps(pipeline, default=str), "products"
        
        else:
            pipeline = [{"$match": {}}]
            
            if time_filter:
                pipeline[0]["$match"]["orderDate"] = {
                    "$gte": time_filter['start'],
                    "$lt": time_filter.get('end', datetime.now())
                }
            
            pipeline.append({"$count": "total"})
            return json.dumps(pipeline, default=str), "orders"
    
    def _generate_list_query(self, entities, time_filter):
        """Generate list query with smart defaults"""
        if entities['target'] == 'customers':
            pipeline = [
                {"$sort": {"createdAt": -1}},
                {"$limit": 10},
                {"$project": {"name": 1, "email": 1, "city": 1}}
            ]
            return json.dumps(pipeline, default=str), "customers"
        
        elif entities['target'] == 'products':
            pipeline = [
                {"$sort": {"createdAt": -1}},
                {"$limit": 10},
                {"$project": {"name": 1, "price": 1, "stock": 1}}
            ]
            return json.dumps(pipeline, default=str), "products"
        
        else:
            pipeline = [
                {"$match": {"status": "completed"}},
                {"$sort": {"orderDate": -1}},
                {"$limit": 10}
            ]
            
            if time_filter:
                pipeline[0]["$match"]["orderDate"] = {
                    "$gte": time_filter['start'],
                    "$lt": time_filter.get('end', datetime.now())
                }
            
            return json.dumps(pipeline, default=str), "orders"

# Global instance
intelligent_engine = IntelligentQueryEngine()

def generate_intelligent_query(question):
    """Main function to generate intelligent queries"""
    try:
        query, collection = intelligent_engine.generate_advanced_query(question)
        return query
    except Exception as e:
        # Fallback to simple query
        return json.dumps([{"$limit": 10}], default=str)
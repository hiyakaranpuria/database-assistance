import os
import json
import pickle
import numpy as np
import requests  # ✅ FIX 1: Added missing import
import ast  # ✅ FIX 2: Added missing import
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pymongo import MongoClient
from bson import ObjectId
import subprocess
import copy

# ============================================================================
# MONGODB CONNECTION & SCHEMA EXTRACTION
# ============================================================================

class MongoDBConnector:
    """Connect to MongoDB and extract schema"""
    
    def __init__(self, connection_string='mongodb://localhost:27017'):
        self.connection_string = connection_string
        self.client = None
        self.db = None
    
    def connect(self, database_name='ai_test_db'):  # ✅ FIX 3: Changed from 'ai-test-db'
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            # print(f"✓ Connected to MongoDB: {database_name}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def extract_schema(self) -> Dict:
        """Extract schema from all collections"""
        schema = {}
        for collection_name in self.db.list_collection_names():
            # Skip system collections
            if collection_name.startswith('system.'):
                continue
            
            try:
                collection = self.db[collection_name]
                doc_count = collection.count_documents({})
                
                # Simple schema inference from one document
                sample_doc = collection.find_one()
                fields_desc = {}
                
                if sample_doc:
                    for key, value in sample_doc.items():
                        fields_desc[key] = f"{type(value).__name__}"
                
                # Get indexes
                try:
                    indexes = [idx['name'] for idx in collection.list_indexes()]
                except:
                    indexes = []
                
                schema[collection_name] = {
                    "description": f"Collection with {len(fields_desc)} fields and {doc_count} documents",
                    "fields": fields_desc,
                    "doc_count": doc_count,
                    "indexed_fields": indexes
                }
                
                # ENHANCEMENT: Boost descriptions for known collections to improve matching
                if 'order' in collection_name.lower():
                    schema[collection_name]['description'] += ". Contains transaction records, purchases, sales, status (completed/pending), amounts, and order details."
                elif 'product' in collection_name.lower():
                    schema[collection_name]['description'] += ". Contains items for sale, inventory, stock, prices, and catalog information."
                elif 'customer' in collection_name.lower() or 'user' in collection_name.lower():
                    schema[collection_name]['description'] += ". Contains user profiles, client details, contact info, email, and demographics."
            except Exception as e:
                print(f"Warning: Could not extract {collection_name}: {e}")
        
        return schema

    def get_collection_sample_fields(self, collection_name: str) -> Dict:
        """Get actual field names and sample values from collection"""
        try:
            collection = self.db[collection_name]
            sample_doc = collection.find_one()
            
            if sample_doc:
                actual_fields = {}
                for field_name, value in sample_doc.items():
                    field_type = type(value).__name__
                    # Show ACTUAL field name and sample value
                    actual_fields[field_name] = {
                        'type': field_type,
                        'sample': str(value)[:50]  # First 50 chars of sample
                    }
                return actual_fields
        except Exception as e:
            print(f"Error getting sample fields: {e}")
        
        return {}
    
    def execute_query(self, collection_name: str, query: Dict) -> List[Dict]:
        """Execute MongoDB query and return results"""
        try:
            collection = self.db[collection_name]
            
            # Safety check - prevent data modification
            if any(op in str(query).lower() for op in ['$set', '$unset', '$push', '$pull', 'deleteOne', 'deleteMany', 'updateOne', 'updateMany', 'drop']):
                return {"error": "Data modification queries are not allowed"}
            
            results = list(collection.find(query).limit(10))
            
            # Convert ObjectId to string for JSON serialization
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return results
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# LLM INTEGRATION (Qwen2.5 3B Local via Ollama API)
# ============================================================================

class LocalLLMInterface:
    """Interface with local db-assistant model via Ollama API"""
    
    def __init__(self, model_name='db-assistant', api_url='http://localhost:11434'):
        self.model_name = model_name
        self.api_url = api_url
    
    def generate(self, user_prompt: str, temperature: float = 0) -> str:
        """Generate response using the Modelfile's SYSTEM prompt automatically"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1024
                },
                "keep_alive": "5m"
            }

            response = requests.post(
                f"{self.api_url}/api/generate", 
                json=payload, 
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                # LOGGING: Print the raw output from LLM to terminal
                print(f"🔹 LLM Generated Query:\n{result}\n")
                return result
            else:
                return f"ERROR: Ollama API returned {response.status_code}"
        
        except Exception as e:
            return f"ERROR: Failed to connect to Ollama API: {str(e)}"


# ============================================================================
# RESPONSE FORMATTER
# ============================================================================

class ResponseFormatter:
    """Format MongoDB query results into human-readable form"""
    
    @staticmethod
    def format_results(results: List[Dict], user_question: str) -> str:
        """Format MongoDB results into readable response"""
        
        if isinstance(results, dict) and "error" in results:
            return f"Error executing query: {results['error']}"
        
        if not results:
            return "❌ No result found in the database matching your search."
        
        count = len(results)
        response = f"Found {count} result{'s' if count != 1 else ''}:\n\n"

        def format_value(k, v):
            """Helper to format values based on key name and type"""
            if isinstance(v, (int, float)):
                # If it's a currency-like field
                if any(x in k.lower() for x in ['spent', 'amount', 'price', 'revenue', 'sales', 'money']):
                    return f"${v:,.2f}"
                # If it's a large number
                return f"{v:,}"
            return str(v)

        # Format each result
        for i, doc in enumerate(results, 1):
            if not isinstance(doc, dict):
                response += f"{i}. {str(doc)[:200]}\n"
                continue

            # Special treatment for aggregation results where _id is a month or category
            # We want to label the _id field helpfully
            parts = []
            for k, v in doc.items():
                if k == '_id':
                    # If _id is the only field or it's a primitive, give it a generic label
                    if len(doc) == 1 or not isinstance(v, (dict, list)):
                        label = "Result"
                        if "month" in user_question.lower(): label = "Month"
                        if "year" in user_question.lower(): label = "Year"
                        if "category" in user_question.lower(): label = "Category"
                        parts.append(f"{label}: {format_value(k, v)}")
                    continue
                
                parts.append(f"{k.replace('_', ' ').title()}: {format_value(k, v)}")
            
            response += f"{i}. {', '.join(parts)}\n"
        
        return response


# ============================================================================
# MAIN CHAT AGENT
# ============================================================================

class MongoDBChatAgent:
    """Main chat agent - handles user queries end-to-end"""
    
    def __init__(self, mongodb_uri='mongodb://localhost:27017', db_name='ai_test_db'):
        """Initialize the chat agent"""
        
        # Initialize components
        self.db_connector = MongoDBConnector(mongodb_uri)
        self.llm = LocalLLMInterface()
        self.formatter = ResponseFormatter()
        
        # Connect to MongoDB
        if not self.db_connector.connect(db_name):
            print("Warning: Failed to connect to MongoDB")
        

    def process_query(self, user_question: str) -> str:
        """DIRECT FLOW: User -> LLM (Memory-based) -> DB Execution"""
        # No system prompt needed - Ollama uses Modelfile
        user_prompt = f"Question: {user_question}\nQuery:"

        print(f"\n🚀 Step 1: LLM Processing user question...")
        query_code = self.llm.generate(user_prompt, temperature=0)

        if "ERROR:" in query_code:
            return f"⚠ {query_code}"

        # 3. Clean & Extract the Code
        query_code = query_code.strip()
        if "```" in query_code:
            blocks = re.findall(r'```(?:mongodb|json|python|javascript)?\s*(.*?)\s*```', query_code, re.DOTALL)
            if blocks: query_code = blocks[0].strip()

        # 4. Parse Collection and Operation
        # Try to find the pattern db.collection_name.operation(...)
        match = re.search(r'(?:db\.)?([a-zA-Z0-9_\-]+)\.(find|aggregate|countDocuments|count)\s*\(', query_code)
        
        collection_name = None
        operation = None
        json_text = ""

        if match:
            collection_name = match.group(1)
            operation = match.group(2)
            # Extract content inside parentheses
            start_idx = query_code.find('(', match.start()) + 1
            stack = 1
            end_idx = -1
            for i in range(start_idx, len(query_code)):
                if query_code[i] == '(': stack += 1
                elif query_code[i] == ')': stack -= 1
                if stack == 0:
                    end_idx = i
                    break
            if end_idx != -1:
                json_text = query_code[start_idx:end_idx].strip()
        else:
            # FALLBACK: Handle "Naked" output (just the JSON array or object)
            query_code_clean = query_code.strip()
            if query_code_clean.startswith('['):
                operation = 'aggregate'
                json_text = query_code_clean
            elif query_code_clean.startswith('{'):
                operation = 'find'
                json_text = query_code_clean
            
            if operation:
                # GUESS COLLECTION based on keywords in user question
                q = user_question.lower()
                if any(word in q for word in ['sale', 'revenue', 'order', 'month', 'transaction']):
                    collection_name = "orders"
                elif any(word in q for word in ['customer', 'user', 'client', 'buyer']):
                    collection_name = "customers"
                elif any(word in q for word in ['product', 'item', 'stock', 'inventory']):
                    collection_name = "products"
                elif 'payment' in q:
                    collection_name = "payments"
                else:
                    collection_name = "orders" # Default fallback

        if not collection_name or not operation:
            return f"❌ AI returned invalid format. Please try again. Code received: {query_code[:100]}..."

        print(f"✅ Step 2: Query Extracted -> Operation: {operation}")


        # 5. Execute Directly
        print(f"📊 Step 3: Executing Query on Collection: {collection_name}")
        try:
            # Quick syntax fix: Replace single quotes with double for JSON
            # and unquoted keys (like name: ) with quoted keys ("name": )
            def fix_json(s):
                # 0. Multilingual & Character Fix: Translate Chinese leakage and stuttering
                translations = {
                    "从": "from", "作为": "as", "本地变量": "localField", 
                    "外部变量": "foreignField", "匹配": "match", "分组": "group",
                    "排序": "sort", "限制": "limit", "项目": "project"
                }
                for cn, en in translations.items(): s = s.replace(cn, en)
                
                # 0.5 HEAL STUTTERING (Remove vertical/horizontal spaces between letters in keywords)
                # Catch: " m \n a \n x " or " m a x " -> "max"
                def collapse_stutter(match):
                    return '"' + match.group(0).replace(' ', '').replace('\n', '').replace('\r', '').replace('"', '') + '"'
                
                # Specifically targeting quoted words with spaces/newlines inside
                s = re.sub(r'\"([a-zA-Z]\s+[a-zA-Z](?:\s+[a-zA-Z])*)\"', collapse_stutter, s)
                s = re.sub(r'\"([a-zA-Z](?:\n+\s*[a-zA-Z])+)\"', collapse_stutter, s)

                # 1. Strip artifacts like "mongodb" language labels
                s = re.sub(r'^(mongodb|javascript|json|mongo|script)\s+', '', s, flags=re.IGNORECASE)

                # 2. Strip JavaScript/MongoShell constructors
                s = re.sub(r'new Date\((.*?)\)|ISODate\((.*?)\)|ObjectId\((.*?)\)', r'\1\2\3', s)
                s = s.replace('new Date()', f'"{datetime.now().isoformat()}"')

                # 3. Syntax Repair: Fix common LLM "double nesting" or broken roots
                s = re.sub(r'\"\$\{\s*\"\$(?:replaceRoot|set|project)\".*?\"newRoot\":\s*\"(.*?)\".*?\}\}', r'"\1"', s)

                # 4. Standard JSON cleaning
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s).strip()
                s = s.replace("'", '"')
                
                # 5. Quote unquoted keys and string values
                s = re.sub(r'([{,]\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', s)
                s = re.sub(r':\s*([a-zA-Z_$][a-zA-Z0-9_$]*)(\s*[},])', r': "\1"\2', s)

                # 6. Force $ prefix on pipeline STAGE and OPERATOR names ONLY
                # DO NOT add $ to lookup sub-keys: from, as, localField, foreignField
                mongo_ops = ['match', 'group', 'sort', 'limit', 'project', 'unwind', 'lookup',
                             'sum', 'avg', 'max', 'min', 'push', 'addFields', 'exists', 'expr',
                             'replaceRoot', 'count', 'first', 'last', 'addToSet']

                for op in mongo_ops:
                    s = re.sub(r'\"(?!\$)' + op + r'\"\s*:', r'\"$' + op + r'\":', s)
                
                return s

            def convert_iso_dates(obj):
                """Recursively look for ISO date strings and convert to datetime objects"""
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        obj[k] = convert_iso_dates(v)
                elif isinstance(obj, list):
                    return [convert_iso_dates(i) for i in obj]
                elif isinstance(obj, str):
                    # Check for ISO8601 format: 2024-01-01T... or 2024-01-01
                    if len(obj) >= 10:
                        # Very simple regex match for date-like string
                        if re.match(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?.*', obj):
                            try:
                                # Strip quotes if they were double-tripled during cleaning
                                cleaned_date = obj.strip('"')
                                # Parse to datetime - handles most common ISO formats
                                # Replace Z with +00:00 for fromisoformat if needed
                                dt_str = cleaned_date.replace('Z', '+00:00')
                                return datetime.fromisoformat(dt_str)
                            except:
                                pass
                return obj

            def clean_string_values(obj):
                """Recursively strip accidental double quotes from string values: '"value"' -> 'value'"""
                if isinstance(obj, dict):
                    return {k: clean_string_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_string_values(i) for i in obj]
                elif isinstance(obj, str):
                    s = obj.strip()
                    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                        return s[1:-1]
                    return s
                return obj

            def make_case_insensitive(obj):
                """Recursively convert plain string values to case-insensitive regex.
                Excludes sensitive fields like username/password."""
                sensitive_fields = {'username', 'password', 'pwd', 'secret', 'key', 'token', 'email'}
                
                if isinstance(obj, list):
                    # aggregate pipeline: look inside each $match stage
                    result = []
                    for stage in obj:
                        if isinstance(stage, dict) and '$match' in stage:
                            ci_match = {}
                            for k, v in stage['$match'].items():
                                # SKIP sensitive fields or already-regex fields
                                is_sensitive = any(f in k.lower() for f in sensitive_fields)
                                if isinstance(v, str) and not k.startswith('$') and not is_sensitive:
                                    ci_match[k] = {'$regex': v, '$options': 'i'}
                                else:
                                    ci_match[k] = v
                            result.append({'$match': ci_match})
                        else:
                            result.append(stage)
                    return result
                elif isinstance(obj, dict):
                    # find filter: convert top-level string values
                    ci_filter = {}
                    for k, v in obj.items():
                        is_sensitive = any(f in k.lower() for f in sensitive_fields)
                        if isinstance(v, str) and not k.startswith('$') and not is_sensitive:
                            ci_filter[k] = {'$regex': v, '$options': 'i'}
                        else:
                            ci_filter[k] = v
                    return ci_filter
                return obj

            clean_json = fix_json(json_text)
            print(f"🧹 Debug: Cleaned JSON for Parsing:\n{clean_json}\n")
            data = json.loads(clean_json)
            
            # NEW: Clean string values to remove redundant quotes '"Mumbai"' -> 'Mumbai'
            data = clean_string_values(data)

            # CRITICAL: Convert strings to actual BSON Dates for comparison
            data = convert_iso_dates(data)

            # ── Execute with smart fallback chain ──────────────────────────────────
            results = []
            attempted = []

            def try_execute(op, d, col):
                """Run one operation and return results list or raise."""
                if op == 'find':
                    return list(self.db_connector.db[col].find(d).limit(10))
                elif op == 'aggregate':
                    return list(self.db_connector.db[col].aggregate(d))
                elif op in ['count', 'countDocuments']:
                    cnt = self.db_connector.db[col].count_documents(d if d else {})
                    return [{'count': cnt}]
                return []

            # Step A: Run with original data
            try:
                results = try_execute(operation, data, collection_name)
                attempted.append(f"{operation} (exact)")
                print(f"▶ Executed as '{operation}' — got {len(results)} doc(s)")
            except Exception as exec_err:
                print(f"⚠ First attempt failed ({operation}): {exec_err}")
                # If operation was 'find' but data is a list, switch to aggregate
                if operation == 'find' and isinstance(data, list):
                    try:
                        results = try_execute('aggregate', data, collection_name)
                        operation = 'aggregate'
                        attempted.append("aggregate (auto-switched from find)")
                        print(f"▶ Auto-switched to aggregate — got {len(results)} doc(s)")
                    except Exception as e2:
                        return f"❌ Execution Failed: {e2}\n\nQuery tried: {query_code}"
                # If operation was 'aggregate' but data is a dict, switch to find
                elif operation == 'aggregate' and isinstance(data, dict):
                    try:
                        results = try_execute('find', data, collection_name)
                        operation = 'find'
                        attempted.append("find (auto-switched from aggregate)")
                        print(f"▶ Auto-switched to find — got {len(results)} doc(s)")
                    except Exception as e2:
                        return f"❌ Execution Failed: {e2}\n\nQuery tried: {query_code}"
                else:
                    return f"❌ Execution Failed: {exec_err}\n\nQuery tried: {query_code}"

            # Step B: If no results, retry with case-insensitive matching
            if not results and operation in ('find', 'aggregate'):
                print(f"🔄 No results found — retrying with case-insensitive matching...")
                try:
                    ci_data = make_case_insensitive(copy.deepcopy(data))
                    results = try_execute(operation, ci_data, collection_name)
                    attempted.append(f"{operation} (case-insensitive retry)")
                    print(f"▶ Case-insensitive retry — got {len(results)} doc(s)")
                except Exception as ci_err:
                    print(f"⚠ Case-insensitive retry failed: {ci_err}")

            # Step C: Final check
            if not results:
                print(f"❌ All attempts returned 0 results. Tried: {' → '.join(attempted)}")
                return (
                    f"No documents found matching your query.\n"
                    f"_Tried: {', '.join(attempted)}_\n"
                    f"_Collection: `{collection_name}` | Filter used: `{json_text[:120]}`_"
                )

            return self.formatter.format_results(results, user_question)

        except Exception as e:
            return f"❌ Execution Failed: {str(e)}\n\nQuery tried: {query_code}"
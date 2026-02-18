import os
import json
import pickle
import numpy as np
import requests  # ✅ FIX 1: Added missing import
import ast  # ✅ FIX 2: Added missing import
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId
import subprocess

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
# EMBEDDING & VECTOR SEARCH
# ============================================================================

class VectorSearchEngine:
    """Generate embeddings and perform semantic search"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings_db = {}
    
    def generate_embeddings(self, schema: Dict) -> Dict:
        """Generate embeddings for all collections"""
        for collection_name, collection_info in schema.items():
            # Collection embedding
            collection_text = f"{collection_name}: {collection_info['description']}"
            collection_embedding = self.model.encode(collection_text)
            
            # Field embeddings
            field_embeddings = {}
            for field_name, field_desc in collection_info['fields'].items():
                field_text = f"{field_name}: {field_desc}"
                field_embedding = self.model.encode(field_text)
                field_embeddings[field_name] = field_embedding.tolist()
            
            # Store
            self.embeddings_db[collection_name] = {
                "description": collection_info['description'],
                "collection_embedding": collection_embedding.tolist(),
                "fields": collection_info['fields'],
                "field_embeddings": field_embeddings,
                "doc_count": collection_info['doc_count'],
                "indexed_fields": collection_info['indexed_fields']
            }
        
        return self.embeddings_db
    
    def save_embeddings(self, filepath='embeddings.pkl'):
        """Save embeddings to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
    
    def load_embeddings(self, filepath='embeddings.pkl'):
        """Load embeddings from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.embeddings_db = pickle.load(f)
            return True
        return False
    
    def search_collections(self, user_question: str, top_k: int = 3) -> List[Tuple]:
        """Find most relevant collections for user question"""
        question_embedding = self.model.encode(user_question)
        question_embedding = np.array(question_embedding)
        
        similarities = {}
        print(f"\nDEBUG: Searching for matches for: '{user_question}'")
        for collection_name, collection_data in self.embeddings_db.items():
            collection_embedding = np.array(collection_data['collection_embedding'])
            norm1 = np.linalg.norm(question_embedding)
            norm2 = np.linalg.norm(collection_embedding)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(question_embedding, collection_embedding) / (norm1 * norm2)
            
            # STRONG KEYWORD FALLBACK
            q_lower = user_question.lower()
            c_lower = collection_name.lower()
            
            # Smart Aliases
            check_name = c_lower
            if c_lower == 'customers' and ('user' in q_lower or 'client' in q_lower):
                # If user asks for 'users', boost 'customers' too because that's where data often is
                print(f"  -> ALIAS MATCH: 'users' mapped to '{collection_name}'")
                similarity = max(similarity, 0.75)
            
            # Check for plural and singular
            if c_lower in q_lower or (c_lower.rstrip('s') in q_lower and len(c_lower) > 3):
                print(f"  -> KEYWORD MATCH FOUND: {collection_name}")
                similarity = max(similarity, 0.8) # Set very high score
            
            similarities[collection_name] = similarity
            print(f"  -> Collection {collection_name}: Score {similarity:.4f}")
        
        top_collections = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        print(f"✅ FOUND {len(top_collections)} RELEVANT TABLES: {', '.join([c[0] for c in top_collections])}")
        return top_collections
    
    def search_fields(self, user_question: str, collection_name: str, top_k: int = 3) -> List[Tuple]:
        """Find most relevant fields in a collection"""
        if collection_name not in self.embeddings_db:
            return []
        
        question_embedding = self.model.encode(user_question)
        question_embedding = np.array(question_embedding)
        
        collection_data = self.embeddings_db[collection_name]
        field_embeddings = collection_data['field_embeddings']
        
        field_similarities = {}
        for field_name, field_embedding in field_embeddings.items():
            field_embedding = np.array(field_embedding)
            norm1 = np.linalg.norm(question_embedding)
            norm2 = np.linalg.norm(field_embedding)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(question_embedding, field_embedding) / (norm1 * norm2)
            
            field_similarities[field_name] = similarity
        
        top_fields = sorted(field_similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_fields


# ============================================================================
# PROMPT BUILDER FOR LLM
# ============================================================================

class PromptBuilder:
    """Build optimized prompts for MongoDB queries"""
    
    def __init__(self, embeddings_db: Dict, db_connector):
        self.embeddings_db = embeddings_db
        self.db_connector = db_connector
        self.search_engine = VectorSearchEngine()
        self.search_engine.embeddings_db = embeddings_db
        self.examples = self.load_examples()
    
    def load_examples(self, filepath='chat_examples.json'):
        """Load few-shot examples from file"""
        import json
        import os
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load examples: {e}")
        return []
    
    def build_context(self, user_question: str, top_k_collections: int = 3) -> Tuple[str, List[Tuple]]:
        print(f"🔍 STEP 1: Vectorizing query for Table Mapping...")
        relevant_collections = self.search_engine.search_collections(user_question, top_k=top_k_collections)
        print(f"📊 Mapping Result: Question relates to tables -> {[c[0] for c in relevant_collections]}")
        
        if not relevant_collections or relevant_collections[0][1] < 0.2:
            return "NO_RELEVANT_COLLECTION", []
        
        # SMART CONTEXT: If 'orders' is found, ALWAYS include 'customers' to enable joins
        found_names = [n for n, s in relevant_collections]
        if 'orders' in found_names and 'customers' not in found_names:
            relevant_collections.append(('customers', 0.99))
            print("🔗 AUTO-INJECTION: Adding 'customers' table for relational mapping.")
        if 'customers' in found_names and 'orders' not in found_names:
             relevant_collections.append(('orders', 0.5))
             print("🔗 AUTO-INJECTION: Adding 'orders' table for relational mapping.")

        context = "## RELEVANT DATABASE SCHEMA MAPPING\n\n"
        
        for collection_name, similarity_score in relevant_collections:
            if similarity_score < 0.2:
                continue
            
            print(f"🔍 STEP 2: Extracting relevant fields for table '{collection_name}'...")
            collection_data = self.embeddings_db[collection_name]
            context += f"### TABLE: `{collection_name}`\n"
            
            # CONTEXT PRUNING: Only show the top 5 most relevant fields
            relevant_fields = self.search_engine.search_fields(user_question, collection_name, top_k=5)
            
            if relevant_fields:
                print(f"   -> Top fields found: {[f[0] for f in relevant_fields]}")
                for field_name, score in relevant_fields:
                    field_desc = collection_data['fields'].get(field_name, "Data field")
                    context += f"  - {field_name} ({field_desc})\n"
            else:
                # Fallback to first few fields if search fails
                for i, (field_name, field_desc) in enumerate(collection_data['fields'].items()):
                    if i >= 5: break
                    context += f"  - {field_name} ({field_desc})\n"
            
            context += "\n"
        
        return context, relevant_collections
        
        return context
    
    def get_reliable_prompt(self, user_question: str) -> Tuple[str, str]:
        """Generate a reliable prompt that prevents hallucination"""
        schema_context = self.build_context(user_question)
        
        if schema_context == "NO_RELEVANT_COLLECTION":
            return (
                "You are a MongoDB query assistant. If the user asks something not related to the database, politely decline and suggest a relevant database query.",
                f"User asked: {user_question}\n\nThis question doesn't seem related to the database. Please ask something about the data in the MongoDB collections."
            )
        
        system_prompt = """You are a MongoDB expert. Rules:
1. Return ONLY the code starting with 'db.'
2. Use ONLY the provided collections and fields.
3. For Joins: use $lookup -> $unwind -> $match.
4. For Searches: use $regex with 'i' option.
5. NO data modification. If unsure, say UNABLE_TO_QUERY.
Return ONLY valid code, no text."""
        
        # Add Few-Shot Examples
        examples_text = "\n### Reference Examples (Follow this style):\n"
        for ex in self.examples[:3]: # Use top 3 examples to keep prompt short
            examples_text += f"Question: \"{ex['question']}\"\nQuery: {ex['query']}\n\n"

        user_prompt = f"""Available Database Schema:
{schema_context}

{examples_text}

User Question: "{user_question}"

Generate the MongoDB query to answer this question. Use ONLY the collections and fields shown above.
Return ONLY the full MongoDB query code starting with 'db.', no explanation.
"""
        
        print(f"DEBUG: System Prompt Length: {len(system_prompt)} chars")
        print(f"DEBUG: User Prompt Length: {len(user_prompt)} chars")
        
        return system_prompt, user_prompt


# ============================================================================
# LLM INTEGRATION (Qwen2.5 3B Local via Ollama API)
# ============================================================================

class LocalLLMInterface:
    """Interface with local Qwen2.5 3B model via Ollama API"""
    
    def __init__(self, model_name='qwen2.5:3b', api_url='http://localhost:11434'):
        self.model_name = model_name
        self.api_url = api_url
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        """Generate response using local Qwen2.5 model via Ollama API"""
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 256
                },
                "keep_alive": "5m"  # Keep model in memory for 5 mins after request
            }
            
            print(f"\n--- DATA SENDING TO LLM ---")
            print(f"System Prompt: {system_prompt[:200]}...")
            print(f"User Question: {user_prompt.split('User Question: ')[-1]}")
            print(f"---------------------------\n")

            response = requests.post(
                f"{self.api_url}/api/generate", 
                json=payload, 
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                print(f"--- RECEIVED FROM LLM ---\n{result}\n--------------------------")
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
            return "No documents found matching your query."
        
        count = len(results)
        response = f"Found {count} result{'s' if count != 1 else ''}:\n\n"
        # Format each result
        for i, doc in enumerate(results, 1):
            # Special handling for aggregation results (single result with calculated fields)
            if count == 1 and isinstance(doc, dict) and '_id' in doc and doc['_id'] is None:
                # This is likely an aggregation result like { _id: null, total: 100 }
                clean_doc = {k: v for k, v in doc.items() if k != '_id'}
                response = "Analysis Result:\n"
                for k, v in clean_doc.items():
                    # Format numbers nicely
                    if isinstance(v, (int, float)):
                         if "price" in k.lower() or "spending" in k.lower() or "amount" in k.lower():
                            response += f"• {k.replace('_', ' ').title()}: ${v:,.2f}\n"
                         else:
                            response += f"• {k.replace('_', ' ').title()}: {v:,}\n"
                    else:
                        response += f"• {k.replace('_', ' ').title()}: {v}\n"
                return response

            response += f"{i}. "
            
            if isinstance(doc, dict):
                key_fields = []
                # Extended list of common fields
                display_keys = [
                    'name', 'title', 'email', 'username', 'product', 'status', 
                    'value', 'amount', 'price', 'totalSales', 'orderCount', 
                    'stock', 'inventory', 'quantity', 'category', 'role', 'city'
                ]
                
                # Helper to flatten and check keys
                def extract_relevant(d, prefix=""):
                    found = []
                    for k, v in d.items():
                        if isinstance(v, dict):
                            found.extend(extract_relevant(v, prefix + k + "."))
                        elif k in display_keys or (prefix + k) in display_keys or 'name' in k.lower():
                             found.append(f"{k}: {v}")
                    return found
                
                key_fields = extract_relevant(doc)
                
                if key_fields:
                    response += ", ".join(key_fields)
                else:
                    response += str(doc)[:200]
            else:
                response += str(doc)[:200]
            
            response += "\n"
        
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
        self.vector_search = VectorSearchEngine()
        self.llm = LocalLLMInterface()
        self.formatter = ResponseFormatter()
        
        # Connect to MongoDB
        if not self.db_connector.connect(db_name):
            print("Warning: Failed to connect to MongoDB")
        
        # Try Loading Embeddings first
        if not self.vector_search.load_embeddings('embeddings.pkl'):
            schema = self.db_connector.extract_schema()
            self.embeddings = self.vector_search.generate_embeddings(schema)
            self.vector_search.save_embeddings('embeddings.pkl')
        else:
            self.embeddings = self.vector_search.embeddings_db
            
        self.prompt_builder = PromptBuilder(self.embeddings, self.db_connector)
        
    def validate_query_fields(self, collection_name: str, query_data: any) -> Optional[str]:
        """Validate that all fields in the query exist in the collection schema"""
        if collection_name not in self.embeddings:
            return None # Skip if metadata is missing
            
        allowed_fields = set(self.embeddings[collection_name]['fields'].keys())
        allowed_fields.update(['_id', 'id', 'count', 'total', 'avg', 'sum', 'min', 'max'])
        
        def get_all_keys(obj, is_top_level_def=False):
            """Extract all keys from a query object, ignoring $ operators unless they define new fields"""
            keys = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if is_top_level_def:
                        # In stages like $group or $project, top-level keys ARE the field definitions
                        if k != '_id': keys.append(k)
                    elif not k.startswith('$'):
                        keys.append(k)
                    
                    # Recursive check
                    keys.extend(get_all_keys(v))
            elif isinstance(obj, list):
                for item in obj:
                    keys.extend(get_all_keys(item))
            return keys

        if isinstance(query_data, list): # Aggregation Pipeline
            virtual_fields = set()
            for stage in query_data:
                if not stage: continue
                op = list(stage.keys())[0]
                
                # 1. Register new fields created by this stage
                if op == '$lookup':
                    as_field = stage['$lookup'].get('as')
                    if as_field: virtual_fields.add(as_field)
                elif op in ['$group', '$project', '$addFields']:
                    new_keys = get_all_keys(stage[op], is_top_level_def=True)
                    virtual_fields.update(new_keys)
                
                # 2. Validate all keys used in this stage
                for key in get_all_keys(stage[op]):
                    root_key = key.split('.')[0]
                    # We allow $ operators within the values, and we allow fields that exist or were created
                    if root_key not in allowed_fields and root_key not in virtual_fields:
                        return f"❌ Hallucination Detected: Field '{key}' does not exist in '{collection_name}'. Available fields: {', '.join(list(allowed_fields)[:10])}..."
        else: # Find Query
            for key in get_all_keys(query_data):
                root_key = key.split('.')[0]
                if root_key not in allowed_fields:
                    return f"❌ Hallucination Detected: Field '{key}' does not exist in '{collection_name}'. Available fields: {', '.join(list(allowed_fields)[:10])}..."
        
        return None
    def process_query(self, user_question: str) -> str:
        """Process user query and return answer"""
        
        # Check for prohibited operations
        prohibited_words = ['drop', 'delete', 'update', 'insert', 'modify', 'remove', 'truncate', 'alter']
        if any(word in user_question.lower() for word in prohibited_words):
            return "❌ Database modification queries are not allowed. You can only view/query data."
        
        # SMART JOIN DETECTION: Handle common cross-table queries with templates
        question_lower = user_question.lower()
        
        # Pattern: Orders with Customer info
        if ('order' in question_lower and any(word in question_lower for word in ['customer', 'user', 'client'])):
            # Check if asking for customer details
            if any(word in question_lower for word in ['name', 'email', 'detail', 'info']):
                # Build join query automatically
                match_filter = {}
                if 'completed' in question_lower:
                    match_filter['status'] = 'completed'
                elif 'pending' in question_lower:
                    match_filter['status'] = 'pending'
                elif 'cancelled' in question_lower:
                    match_filter['status'] = 'cancelled'
                
                pipeline = [
                    {"$match": match_filter} if match_filter else {"$match": {}},
                    {"$lookup": {
                        "from": "customers",
                        "localField": "customerId",
                        "foreignField": "_id",
                        "as": "customer_info"
                    }},
                    {"$unwind": {"path": "$customer_info", "preserveNullAndEmptyArrays": True}},
                    {"$project": {
                        "_id": 0,
                        "status": 1,
                        "amount": 1,
                        "quantity": 1,
                        "customer_name": "$customer_info.name",
                        "customer_email": "$customer_info.email"
                    }},
                    {"$limit": 10}
                ]
                
                try:
                    results = list(self.db_connector.db['orders'].aggregate(pipeline))
                    return self.formatter.format_results(results, user_question)
                except Exception as e:
                    # Fall through to LLM if template fails
                    print(f"Template join failed: {e}, falling back to LLM")
        
        # Build prompt with relevant schema
        schema_context, relevant_collections = self.prompt_builder.build_context(user_question)
        
        if schema_context == "NO_RELEVANT_COLLECTION":
            return "❌ This question is not related to the database. Please ask something about the data in MongoDB."
        
        system_prompt, user_prompt = self.prompt_builder.get_reliable_prompt(user_question)
        
        # Get query from Qwen2.5
        query_code = self.llm.generate(system_prompt, user_prompt, temperature=0)
        
        # DEBUG: Print query to terminal
        print(f"\n🔹 MODEL OUTPUT:\n{query_code}\n")
        
        # Check for errors
        if "ERROR:" in query_code:
            return f"⚠ {query_code}"
        
        if "UNABLE_TO_QUERY" in query_code:
            return "⚠ I cannot construct a valid query for this question with the available schema."
        
        if "MODIFICATION_NOT_ALLOWED" in query_code:
            return "❌ Database modification is not allowed. This system is read-only."
        
        if "OUT_OF_SCOPE" in query_code:
            return "❌ This question is not related to the database."
        
        if "OUT_OF_SCOPE" in query_code:
            return "❌ This question is not related to the database."

        # FLEXIBLE PARSING LOGIC
        query_code = query_code.strip()
        
        # 1. Clean Markdown
        if "```" in query_code:
            # Extract first code block
            blocks = re.findall(r'```(?:mongodb|json|python|javascript)?\s*(.*?)\s*```', query_code, re.DOTALL)
            if blocks:
                query_code = blocks[0].strip()
        
        results = []
        collection_name = "Unknown"
        
        try:
            # 2. Try to find the pattern db.collection.operation(...) or collection.operation(...)
            # Note: collection name can include - or _ and we look for find, aggregate, countDocuments
            match = re.search(r'(?:db\.)?([a-zA-Z0-9_\-]+)\.(find|aggregate|countDocuments|count)\s*\(', query_code)
            
            collection_name = None
            operation = None
            content = ""
            
            if match:
                collection_name = match.group(1)
                operation = match.group(2)
                content = query_code[match.end():]
                # Try to balance parentheses
                stack = 1
                end_pos = -1
                for i, char in enumerate(content):
                    if char == '(': stack += 1
                    elif char == ')': stack -= 1
                    if stack == 0:
                        end_pos = i
                        break
                if end_pos != -1:
                    content = content[:end_pos]
            else:
                # 3. Handle "Naked" output (just the JSON object or array)
                # Guess collection from relevant_collections
                if relevant_collections:
                     collection_name = relevant_collections[0][0]
                else:
                     return f"⚠ No relevant collection found for query:\n{query_code}"
                
                if query_code.startswith('['):
                    operation = 'aggregate'
                    content = query_code
                elif query_code.startswith('{'):
                    operation = 'find'
                    content = query_code
                else:
                    return f"⚠ Could not identify MongoDB command or valid JSON in response:\n{query_code}"

            # 4. Normalize JSON content
            # Fix unquoted keys, handle single quotes, etc.
            # Using a simplified version of the previous logic
            def clean_json(s):
                # Replace single quotes with double quotes (carefully)
                s = s.replace("'", '"')
                # Add quotes to unquoted keys
                s = re.sub(r'([{,\s])([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', s)
                
                # SAFETY FIX: Trim Regex patterns (fixes " udaipur" issue)
                s = re.sub(r'("\$regex"\s*:\s*)"\s*(\S.*?)\s*"', r'\1"\2"', s)
                
                # SAFETY FIX: Fix quoted objects in arrays (fixes "{$limit":10})
                # Turns "{$limit":10} into {"$limit":10}
                s = re.sub(r'"\s*(\{[^"]+?\})\s*"', r'\1', s)
                s = re.sub(r'"\s*(\{\$[^"]+?\})\s*"', r'\1', s)
                
                # SAFETY FIX: Auto-close brackets if missing
                open_sq = s.count('[')
                close_sq = s.count(']')
                if open_sq > close_sq:
                    s += int(open_sq - close_sq) * ']'
                
                open_br = s.count('{')
                close_br = s.count('}')
                if open_br > close_br:
                    s += int(open_br - close_br) * '}'
                
                return s

            clean_content = clean_json(content)
            
            try:
                data = json.loads(clean_content)
            except:
                try:
                    import ast
                    data = ast.literal_eval(content)
                except:
                    return f"❌ Failed to parse query data: {content[:100]}..."

            # 5. Schema Guardrail: Validate Fields
            validation_error = self.validate_query_fields(collection_name, data)
            if validation_error:
                print(f"🚩 SCHEMA GUARDRAIL TRIGGERED: {validation_error}")
                # We could try to auto-retry here, but for now, we'll return the error
                return f"{validation_error}\n\nAsk me again, but try to use fields from the schema list above."

            # 6. Execute based on operation
            if operation == 'find':
                results = self.db_connector.execute_query(collection_name, data)
            elif operation in ['countDocuments', 'count']:
                count = self.db_connector.db[collection_name].count_documents(data if isinstance(data, dict) else {})
                results = [{"count": count}]
            elif operation == 'aggregate':
                # Robust extraction for aggregate pipeline
                    try:
                         # Find outermost brackets
                        start_idx = query_code.find('[')
                        end_idx = query_code.rfind(']') + 1
                        
                        if start_idx != -1 and end_idx > start_idx:
                            raw_content = query_code[start_idx:end_idx]
                            clean_content = clean_json(raw_content)
                            pipeline = json.loads(clean_content)
                        else:
                            raise ValueError("No brackets found")
                    except:
                        # Fallback: AST Literal Eval (handles single quotes/messy syntax better)
                        try:
                            # Try to find list in the string manually
                            pass # use content logic below
                            content = query_code[match.end():] # fallback
                            start_idx = content.find('[')
                            end_idx = content.rfind(']') + 1
                            if start_idx != -1: 
                                content = content[start_idx:end_idx]
                                import ast
                                pipeline = ast.literal_eval(content)
                            else:
                                pipeline = []
                        except:
                            pipeline = []
                    
                    if not pipeline:
                         return f"⚠ Parsing failed for pipeline:\n{query_code}"

                    # INTELLIGENT PIPELINE REORDERING (Fixing 'Match before Unwind' bug)
                    # If we see LOOKUP -> MATCH -> UNWIND, we must swap Match and Unwind
                    for i in range(len(pipeline) - 1):
                        stage_curr = list(pipeline[i].keys())[0]
                        stage_next = list(pipeline[i+1].keys())[0]
                        
                        if stage_curr == '$match' and stage_next == '$unwind':
                            # Check if match refers to a joined field (contains dot)
                            match_query = pipeline[i]['$match']
                            is_joined_field = any('.' in k for k in match_query.keys())
                            
                            if is_joined_field:
                                # SWAP THEM!
                                print("🔄 AUTO-FIXING PIPELINE: Swapping $match and $unwind")
                                pipeline[i], pipeline[i+1] = pipeline[i+1], pipeline[i]

                    # Execute aggregate
                    # Increase safety limit to 50 (was 10) so user requests for "Top 20" work
                    results = list(self.db_connector.db[collection_name].aggregate(pipeline))[:50]
            
            else:
                return f"⚠ Could not identify MongoDB command (find/aggregate/countDocuments) in response:\n{query_code}"
        
        except Exception as e:
            return f"❌ Query execution error: {str(e)}\n\nGenerated query was for '{collection_name}'"

        # Format results
        formatted_response = self.formatter.format_results(results, user_question)
        
        return formatted_response
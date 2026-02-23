#!/usr/bin/env python3
"""
Simplified Chat with Database Flow
User Question -> LLM -> Query -> Validate -> Execute -> Results
"""

import json
import re
import requests
from pymongo import MongoClient
from bson import ObjectId, errors as bson_errors
from datetime import datetime

class SimpleChatFlow:
    """Direct chat flow: Question -> LLM -> Validate -> Execute"""
    
    def __init__(self, mongodb_uri='mongodb://localhost:27017', db_name='ai_test_db'):
        # MongoDB Connection
        self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        
        # LLM Configuration
        self.llm_url = "http://localhost:11434/api/generate"
        self.model_name = "db-assistant"
        
        # Load schema from schema.md
        self.schema = self._load_schema()
        
        print(f"✓ Connected to MongoDB: {db_name}")
        print(f"✓ LLM Model: {self.model_name}")
        print(f"✓ Schema loaded: {list(self.schema.keys())}")
    
    def _load_schema(self):
        """Load schema from database_schema.md"""
        schema = {}
        try:
            with open('database_schema.md', 'r') as f:
                content = f.read()
                
            # Parse schema format: Collection 'name': {field(type), ...}
            pattern = r"Collection '(\w+)': \{([^}]+)\}"
            matches = re.findall(pattern, content)
            
            for collection_name, fields_str in matches:
                fields = {}
                # Parse fields: name(type), price(number), etc.
                field_pattern = r'(\w+)\((\w+)\)'
                field_matches = re.findall(field_pattern, fields_str)
                
                for field_name, field_type in field_matches:
                    fields[field_name] = field_type
                
                schema[collection_name] = fields
                
        except FileNotFoundError:
            print("⚠ database_schema.md not found, extracting from database...")
            # Fallback: extract from database
            for coll_name in self.db.list_collection_names():
                if not coll_name.startswith('system.'):
                    sample = self.db[coll_name].find_one()
                    if sample:
                        schema[coll_name] = {k: type(v).__name__ for k, v in sample.items()}
        
        return schema
    
    def ask(self, user_question: str, max_retries: int = 3) -> dict:
        """
        Main flow: Question -> LLM -> Validate -> Execute
        
        Args:
            user_question: Natural language question
            max_retries: Number of times to retry if query has syntax errors
            
        Returns:
            dict with 'success', 'results', 'query', 'error'
        """
        print(f"\n{'='*60}")
        print(f"Question: {user_question}")
        print(f"{'='*60}")
        
        for attempt in range(1, max_retries + 1):
            print(f"\n[Attempt {attempt}/{max_retries}]")
            
            # Step 1: Generate query from LLM
            print("→ Sending to LLM...")
            previous_error = None if attempt == 1 else error_msg
            query_code = self._generate_query(user_question, attempt, previous_error)
            
            if not query_code or "ERROR" in query_code:
                return {
                    'success': False,
                    'error': f"LLM failed to generate query: {query_code}",
                    'query': None,
                    'results': None
                }
            
            print(f"→ LLM returned:\n{query_code}\n")
            
            # Step 2: Validate syntax
            print("→ Validating syntax...")
            is_valid, error_msg, parsed_query = self._validate_syntax(query_code)
            
            if not is_valid:
                print(f"✗ Syntax Error: {error_msg}")
                if attempt < max_retries:
                    print(f"→ Regenerating query (attempt {attempt + 1})...")
                    continue
                else:
                    return {
                        'success': False,
                        'error': f"Syntax validation failed after {max_retries} attempts: {error_msg}",
                        'query': query_code,
                        'results': None
                    }
            
            print("✓ Syntax valid")
            
            # Step 3: Execute query
            print("→ Executing query...")
            results = self._execute_query(parsed_query)
            
            if isinstance(results, dict) and 'error' in results:
                print(f"✗ Execution Error: {results['error']}")
                if attempt < max_retries:
                    print(f"→ Regenerating query (attempt {attempt + 1})...")
                    continue
                else:
                    return {
                        'success': False,
                        'error': f"Execution failed: {results['error']}",
                        'query': query_code,
                        'results': None
                    }
            
            # Success!
            print(f"✓ Query executed successfully")
            print(f"✓ Found {len(results)} results")
            
            return {
                'success': True,
                'results': results,
                'query': query_code,
                'error': None
            }
        
        return {
            'success': False,
            'error': f"Failed after {max_retries} attempts",
            'query': None,
            'results': None
        }
    
    def _generate_query(self, user_question: str, attempt: int, previous_error: str = None) -> str:
        """Send question to LLM and get MongoDB query"""
        
        # Build schema context
        schema_text = "Available Collections:\n"
        for coll_name, fields in self.schema.items():
            schema_text += f"\n{coll_name}: "
            schema_text += ", ".join([f"{fname}({ftype})" for fname, ftype in fields.items()])
        
        # System prompt
        system_prompt = f"""You are a MongoDB query expert. Generate ONLY valid MongoDB queries.

{schema_text}

RULES:
1. Return ONLY the query code, no explanation
2. Use format: db.collection.find({{}}) or db.collection.aggregate([])
3. Use ONLY collections and fields from the schema above
4. For dates, use ISODate("YYYY-MM-DD") format
5. Always limit results to 10 documents
6. Return valid JSON syntax
7. Use double quotes for JSON strings
8. Do NOT use fields that don't exist in the schema

Example outputs:
db.orders.find({{"status": "completed"}})
db.orders.aggregate([{{"$group": {{"_id": null, "total": {{"$sum": "$amount"}}}}}}])
"""
        
        # Add retry context if this is not the first attempt
        if attempt > 1:
            system_prompt += f"\n\nPREVIOUS ATTEMPT FAILED"
            if previous_error:
                system_prompt += f" with error: {previous_error}"
            system_prompt += "\nGenerate a DIFFERENT, SIMPLER query. Fix the error."
        
        user_prompt = f"Question: {user_question}\n\nMongoDB Query:"
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.1 if attempt == 1 else 0.3,  # Increase randomness on retries
                        "num_predict": 300
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"ERROR: LLM API returned {response.status_code}"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _validate_syntax(self, query_code: str) -> tuple:
        """
        Validate MongoDB query syntax with enhanced error detection
        
        Returns:
            (is_valid, error_message, parsed_query)
        """
        try:
            # Clean the code
            query_code = query_code.strip()
            
            # Remove markdown code blocks
            if "```" in query_code:
                blocks = re.findall(r'```(?:javascript|json|mongodb)?\s*(.*?)\s*```', query_code, re.DOTALL)
                if blocks:
                    query_code = blocks[0].strip()
            
            # Parse collection and operation
            # Pattern: db.collection.operation(...)
            match = re.search(r'db\.(\w+)\.(find|aggregate|countDocuments)\s*\(', query_code)
            
            if not match:
                return False, "Invalid format. Expected: db.collection.find() or db.collection.aggregate()", None
            
            collection_name = match.group(1)
            operation = match.group(2)
            
            # Check if collection exists
            if collection_name not in self.schema:
                available = ", ".join(self.schema.keys())
                return False, f"Collection '{collection_name}' not found. Available: {available}", None
            
            # Extract the query/pipeline content
            start_idx = query_code.find('(', match.start()) + 1
            paren_count = 1
            end_idx = start_idx
            
            for i in range(start_idx, len(query_code)):
                if query_code[i] == '(':
                    paren_count += 1
                elif query_code[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        end_idx = i
                        break
            
            json_text = query_code[start_idx:end_idx].strip()
            
            # Fix common JSON issues
            json_text = self._fix_json(json_text)
            
            # Try to parse as JSON
            try:
                if operation == 'aggregate':
                    pipeline = json.loads(json_text)
                    if not isinstance(pipeline, list):
                        return False, "Aggregate pipeline must be a list", None
                    
                    # Validate fields in pipeline
                    field_error = self._validate_fields(collection_name, pipeline)
                    if field_error:
                        return False, field_error, None
                    
                    parsed_query = {
                        'collection': collection_name,
                        'operation': 'aggregate',
                        'pipeline': pipeline
                    }
                else:  # find
                    query_dict = json.loads(json_text) if json_text else {}
                    if not isinstance(query_dict, dict):
                        return False, "Find query must be a dictionary", None
                    
                    # Validate fields in query
                    field_error = self._validate_fields(collection_name, query_dict)
                    if field_error:
                        return False, field_error, None
                    
                    parsed_query = {
                        'collection': collection_name,
                        'operation': 'find',
                        'query': query_dict
                    }
                
                return True, None, parsed_query
                
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}", None
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _validate_fields(self, collection_name: str, query_data) -> str:
        """
        Validate that fields used in query exist in schema
        Returns error message if invalid, None if valid
        """
        valid_fields = set(self.schema[collection_name].keys())
        # MongoDB operators and special fields
        allowed_special = {'_id', '$match', '$group', '$project', '$sort', '$limit', '$lookup', 
                          '$unwind', '$sum', '$avg', '$count', '$push', '$addToSet', '$first', 
                          '$last', '$max', '$min', '$and', '$or', '$not', '$in', '$nin', '$gt', 
                          '$gte', '$lt', '$lte', '$eq', '$ne', '$exists', '$regex', '$options',
                          'from', 'localField', 'foreignField', 'as', 'preserveNullAndEmptyArrays'}
        
        def check_fields(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Skip MongoDB operators
                    if key.startswith('$') or key in allowed_special:
                        check_fields(value, path)
                        continue
                    
                    # Check if field exists
                    field_name = key.split('.')[0]  # Handle nested fields like "customer.name"
                    if field_name not in valid_fields and field_name not in allowed_special:
                        available = ", ".join(sorted(valid_fields)[:10])
                        return f"Field '{key}' doesn't exist in '{collection_name}'. Available: {available}..."
                    
                    check_fields(value, f"{path}.{key}")
                    
            elif isinstance(obj, list):
                for item in obj:
                    error = check_fields(item, path)
                    if error:
                        return error
            
            elif isinstance(obj, str) and obj.startswith('$'):
                # Field reference like "$amount"
                field_name = obj[1:].split('.')[0]
                if field_name and field_name not in valid_fields and field_name not in allowed_special:
                    available = ", ".join(sorted(valid_fields)[:10])
                    return f"Field '{obj}' doesn't exist in '{collection_name}'. Available: {available}..."
            
            return None
        
        return check_fields(query_data)
    
    def _fix_json(self, json_text: str) -> str:
        """Fix common JSON syntax issues"""
        # Replace single quotes with double quotes
        json_text = json_text.replace("'", '"')
        
        # Fix unquoted keys: {name: "value"} -> {"name": "value"}
        json_text = re.sub(r'(\w+):', r'"\1":', json_text)
        
        # Fix MongoDB date constructors
        json_text = re.sub(r'ISODate\("([^"]+)"\)', r'"\1"', json_text)
        json_text = re.sub(r'new Date\("([^"]+)"\)', r'"\1"', json_text)
        json_text = re.sub(r'ObjectId\("([^"]+)"\)', r'"\1"', json_text)
        
        return json_text
    
    def _execute_query(self, parsed_query: dict) -> list:
        """Execute the validated MongoDB query"""
        try:
            collection = self.db[parsed_query['collection']]
            
            if parsed_query['operation'] == 'aggregate':
                results = list(collection.aggregate(parsed_query['pipeline']))
            else:  # find
                results = list(collection.find(parsed_query['query']).limit(10))
            
            # Convert ObjectId to string for JSON serialization
            for doc in results:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def format_results(self, response: dict) -> str:
        """Format results for display"""
        if not response['success']:
            return f"❌ Error: {response['error']}"
        
        results = response['results']
        
        if not results:
            return "No results found."
        
        output = f"\n✓ Found {len(results)} result(s):\n\n"
        
        for i, doc in enumerate(results, 1):
            output += f"{i}. "
            if isinstance(doc, dict):
                # Show key fields
                parts = []
                for key, value in doc.items():
                    if key != '_id':
                        parts.append(f"{key}: {value}")
                output += ", ".join(parts[:5])  # Show first 5 fields
            else:
                output += str(doc)
            output += "\n"
        
        return output
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("MongoDB Chat Assistant - Simplified Flow")
        print("="*60)
        print("Type your question or 'exit' to quit\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break
                
                # Process the question
                response = self.ask(question)
                
                # Display results
                print("\n" + self.format_results(response))
                
                # Show the query that was used
                if response['query']:
                    print(f"\n[Query used: {response['query'][:100]}...]")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    try:
        chat = SimpleChatFlow()
        chat.chat()
    except Exception as e:
        print(f"Failed to start: {e}")
        print("\nMake sure:")
        print("  1. MongoDB is running")
        print("  2. Ollama is running with db-assistant model")
        print("  3. database_schema.md exists")

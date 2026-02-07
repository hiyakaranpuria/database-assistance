"""
MongoDB AI-Powered Query Assistant
Single File - Ready to Paste into AI IDE
Uses: Qwen2.5 3B (Local), MongoDB (ai-test-db), Vector Search
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from bson import ObjectId

# ============================================================================
# MONGODB CONNECTION & SCHEMA EXTRACTION
# ============================================================================

class MongoDBConnector:
    """Connect to MongoDB and extract schema"""
    
    def __init__(self, connection_string='mongodb://localhost:27017'):
        self.connection_string = connection_string
        self.client = None
        self.db = None
    
    def connect(self, database_name='ai-test-db'):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            print(f"✓ Connected to MongoDB: {database_name}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def extract_schema(self) -> Dict:
        """Extract complete schema from all collections"""
        schema = {}
        collections = self.db.list_collection_names()
        
        for collection_name in collections:
            if collection_name.startswith('system.'):
                continue
            
            collection = self.db[collection_name]
            
            try:
                doc_count = collection.count_documents({})
                sample_docs = list(collection.find().limit(50))
                
                # Infer field types
                field_types = {}
                for doc in sample_docs:
                    for field_name, value in doc.items():
                        if field_name not in field_types:
                            field_type = type(value).__name__
                            field_types[field_name] = field_type
                
                # Get indexes
                indexes = list(collection.list_indexes())
                indexed_fields = set()
                for idx in indexes:
                    for field, _ in idx['key']:
                        if field != '_id':
                            indexed_fields.add(field)
                
                # Build description
                fields_desc = {}
                for field_name, field_type in field_types.items():
                    type_map = {
                        'str': 'String', 'int': 'Integer', 'float': 'Double',
                        'bool': 'Boolean', 'dict': 'Document', 'list': 'Array',
                        'ObjectId': 'ObjectId', 'datetime': 'Date'
                    }
                    field_desc = f"{type_map.get(field_type, field_type)}"
                    if field_name in indexed_fields:
                        field_desc += " [Indexed]"
                    fields_desc[field_name] = field_desc
                
                schema[collection_name] = {
                    "description": f"Collection with {len(fields_desc)} fields and {doc_count} documents",
                    "fields": fields_desc,
                    "doc_count": doc_count,
                    "indexed_fields": list(indexed_fields)
                }
            except Exception as e:
                print(f"Warning: Could not extract {collection_name}: {e}")
        
        return schema
    
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
        print(f"✓ Loaded embedding model: {model_name}")
    
    def generate_embeddings(self, schema: Dict) -> Dict:
        """Generate embeddings for all collections"""
        print("Generating embeddings...")
        
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
        
        print(f"✓ Generated embeddings for {len(self.embeddings_db)} collections")
        return self.embeddings_db
    
    def save_embeddings(self, filepath='embeddings.pkl'):
        """Save embeddings to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print(f"✓ Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath='embeddings.pkl'):
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            self.embeddings_db = pickle.load(f)
        print(f"✓ Embeddings loaded from {filepath}")
        return self.embeddings_db
    
    def search_collections(self, user_question: str, top_k: int = 3) -> List[Tuple]:
        """Find most relevant collections for user question"""
        question_embedding = self.model.encode(user_question)
        question_embedding = np.array(question_embedding)
        
        similarities = {}
        for collection_name, collection_data in self.embeddings_db.items():
            collection_embedding = np.array(collection_data['collection_embedding'])
            norm1 = np.linalg.norm(question_embedding)
            norm2 = np.linalg.norm(collection_embedding)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(question_embedding, collection_embedding) / (norm1 * norm2)
            
            similarities[collection_name] = similarity
        
        top_collections = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
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
    
    def __init__(self, embeddings_db: Dict):
        self.embeddings_db = embeddings_db
        self.search_engine = VectorSearchEngine()
        self.search_engine.embeddings_db = embeddings_db
    
    def build_context(self, user_question: str, top_k_collections: int = 3) -> str:
        """Build relevant schema context"""
        relevant_collections = self.search_engine.search_collections(user_question, top_k=top_k_collections)
        
        if not relevant_collections or relevant_collections[0][1] < 0.2:
            return "NO_RELEVANT_COLLECTION"
        
        context = "## Available MongoDB Collections\n\n"
        
        for collection_name, similarity_score in relevant_collections:
            if similarity_score < 0.2:  # Skip very low relevance
                continue
            
            collection_data = self.embeddings_db[collection_name]
            
            context += f"### Collection: `{collection_name}`\n"
            context += f"Document Count: {collection_data['doc_count']}\n"
            context += f"Fields:\n"
            
            # Get relevant fields
            relevant_fields = self.search_engine.search_fields(user_question, collection_name, top_k=5)
            
            if relevant_fields:
                for field_name, field_similarity in relevant_fields:
                    if field_similarity > 0.1:
                        field_desc = collection_data['fields'][field_name]
                        context += f"  - `{field_name}`: {field_desc}\n"
            else:
                # Show all fields if no good match
                for field_name, field_desc in list(collection_data['fields'].items())[:5]:
                    context += f"  - `{field_name}`: {field_desc}\n"
            
            if collection_data['indexed_fields']:
                context += f"Indexed Fields: {', '.join(collection_data['indexed_fields'])}\n"
            
            context += "\n"
        
        return context
    
    def get_reliable_prompt(self, user_question: str) -> Tuple[str, str]:
        """
        Generate a reliable prompt that prevents hallucination
        Returns: (system_prompt, user_prompt)
        """
        schema_context = self.build_context(user_question)
        
        if schema_context == "NO_RELEVANT_COLLECTION":
            return (
                "You are a MongoDB query assistant. If the user asks something not related to the database, politely decline and suggest a relevant database query.",
                f"User asked: {user_question}\n\nThis question doesn't seem related to the database. Please ask something about the data in the MongoDB collections."
            )
        
        system_prompt = """You are a MongoDB query expert. Your ONLY job is to:
1. Generate MongoDB queries based on user questions
2. Use ONLY the collections and fields provided
3. Return ONLY valid MongoDB query syntax
4. Do NOT make up field names or collections
5. Do NOT attempt data modifications (no $set, $unset, deleteOne, deleteMany, drop, etc.)
6. Do NOT hallucinate - if you can't construct a valid query, say "UNABLE_TO_QUERY"
7. For simple queries use: db.collection.find({...})
8. For complex queries use: db.collection.aggregate([...])
9. Always limit results to maximum 10 documents

CRITICAL RULES:
- NEVER suggest DROP, DELETE, UPDATE, or MODIFY operations
- NEVER create new collections or fields
- If user asks for data modification, respond: "MODIFICATION_NOT_ALLOWED: Cannot modify database"
- If user asks something unrelated to database, respond: "OUT_OF_SCOPE: This question is not related to the database"

Return ONLY the MongoDB query code, nothing else."""
        
        user_prompt = f"""Available Database Schema:
{schema_context}

User Question: "{user_question}"

Generate the MongoDB query to answer this question. Use ONLY the collections and fields shown above.
Return ONLY the query code, no explanation."""
        
        return system_prompt, user_prompt


# ============================================================================
# LLM INTEGRATION (Qwen2.5 3B Local)
# ============================================================================

class LocalLLMInterface:
    """Interface with local Qwen2.5 3B model"""
    
    def __init__(self, model_name='qwen2.5:3b'):
        self.model_name = model_name
        self.check_model_available()
    
    def check_model_available(self):
        """Check if Ollama and model are available"""
        try:
            import subprocess
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Ollama available")
            else:
                print("⚠ Ollama not found. Install from: https://ollama.ai")
        except FileNotFoundError:
            print("⚠ Ollama not installed. Install from: https://ollama.ai")
    
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        """
        Generate response using local Qwen2.5 model
        
        Args:
            system_prompt: System instructions for the model
            user_prompt: User's question/request
            temperature: 0 = deterministic (best for queries)
        
        Returns:
            Generated MongoDB query or response
        """
        try:
            import subprocess
            import json
            
            # Combine prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call Ollama
            result = subprocess.run(
                [
                    'ollama', 'run', self.model_name,
                    '--num-predict', '200',  # Limit output tokens
                    '--temperature', str(temperature)
                ],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"ERROR: {result.stderr}"
        
        except subprocess.TimeoutExpired:
            return "ERROR: Model response timeout (low latency not met)"
        except Exception as e:
            return f"ERROR: {str(e)}"


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
        
        # Count results
        count = len(results)
        response = f"Found {count} result{'s' if count != 1 else ''}:\n\n"
        
        # Format each result
        for i, doc in enumerate(results, 1):
            response += f"{i}. "
            
            # Try to create readable summary
            if isinstance(doc, dict):
                # Extract key fields
                key_fields = []
                for key in ['name', 'title', 'email', 'username', 'product', 'status', 'value', 'amount', 'price']:
                    if key in doc:
                        key_fields.append(f"{key}: {doc[key]}")
                
                if key_fields:
                    response += ", ".join(key_fields[:3])
                else:
                    response += str(doc)[:100]
            else:
                response += str(doc)[:100]
            
            response += "\n"
        
        return response


# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

class MongoDBChatAssistant:
    """Main chat interface - handles user queries end-to-end"""
    
    def __init__(self, mongodb_uri='mongodb://localhost:27017', db_name='ai_test_db'):
        """Initialize the assistant"""
        print("=" * 70)
        print("MongoDB AI Chat Assistant")
        print("=" * 70)
        
        # Initialize components
        self.db_connector = MongoDBConnector(mongodb_uri)
        self.vector_search = VectorSearchEngine()
        self.llm = LocalLLMInterface()
        self.formatter = ResponseFormatter()
        
        # Connect to MongoDB
        if not self.db_connector.connect(db_name):
            raise Exception("Failed to connect to MongoDB")
        
        # Extract schema and generate embeddings
        print("\nExtracting database schema...")
        schema = self.db_connector.extract_schema()
        
        print("Generating embeddings...")
        self.embeddings = self.vector_search.generate_embeddings(schema)
        
        # Save embeddings
        self.vector_search.save_embeddings('embeddings.pkl')
        
        self.prompt_builder = PromptBuilder(self.embeddings)
        
        print("\n✓ Assistant ready!")
        print("Type 'exit' to quit\n")
    
    def process_query(self, user_question: str) -> str:
        """
        Process user query and return answer
        
        Flow:
        1. User asks question
        2. Vector search finds relevant collections
        3. Build optimized prompt
        4. Send to Qwen2.5 local model
        5. Parse MongoDB query
        6. Execute query
        7. Format results
        8. Return human-readable answer
        """
        
        # Check for prohibited operations
        prohibited_words = ['drop', 'delete', 'update', 'insert', 'modify', 'remove', 'truncate', 'alter']
        if any(word in user_question.lower() for word in prohibited_words):
            return "❌ Database modification queries are not allowed. You can only view/query data."
        
        # Step 1: Build prompt with relevant schema
        schema_context, relevant_collections = self.prompt_builder.build_context(user_question)
        
        if schema_context == "NO_RELEVANT_COLLECTION":
            return "❌ This question is not related to the database. Please ask something about the data in MongoDB."
        
        system_prompt, user_prompt = self.prompt_builder.get_reliable_prompt(user_question)
        
        # Step 2: Get query from Qwen2.5
        print("🤖 Qwen2.5 thinking...", end="", flush=True)
        query_code = self.llm.generate(system_prompt, user_prompt, temperature=0)
        print(" ✓")
        
        # Check for errors
        if "ERROR:" in query_code:
            return f"⚠ {query_code}"
        
        if "UNABLE_TO_QUERY" in query_code:
            return "❌ Unable to construct a query for this question. Try rephrasing or ask something else."
        
        if "MODIFICATION_NOT_ALLOWED" in query_code:
            return "❌ Database modification queries are not allowed."
        
        if "OUT_OF_SCOPE" in query_code:
            return "❌ This question is not related to the database."
        
        # Step 3: Parse the MongoDB query
        print("📊 Executing query...", end="", flush=True)
        
        # Extract MongoDB commands
        query_code = query_code.strip()
        
        # Parse collection name and query
        try:
            if 'find(' in query_code:
                # Extract collection name and find query
                parts = query_code.split('find(')
                collection_name = parts[0].replace('db.', '').strip()
                query_str = 'find(' + parts[1]
                
                # Extract query dict
                query_dict = {}
                if '{' in query_str:
                    import json
                    start = query_str.find('{')
                    end = query_str.rfind('}') + 1
                    query_dict = json.loads(query_str[start:end])
                
                # Execute query
                results = self.db_connector.execute_query(collection_name, query_dict)
            
            elif 'aggregate(' in query_code:
                # For aggregate queries, execute manually
                collection_name = query_code.split('aggregate(')[0].replace('db.', '').strip()
                # Parse pipeline
                pipeline_str = query_code.split('aggregate(')[1].rsplit(']', 1)[0] + ']'
                import json
                pipeline = json.loads(pipeline_str)
                results = list(self.db_connector.db[collection_name].aggregate(pipeline))[:10]
            
            else:
                return f"⚠ Query format not recognized:\n{query_code}"
        
        except Exception as e:
            print(" ✗")
            return f"❌ Query execution error: {str(e)}\n\nGenerated query:\n{query_code}"
        
        print(" ✓")
        
        # Step 4: Format results
        print("📝 Formatting results...", end="", flush=True)
        formatted_response = self.formatter.format_results(results, user_question)
        print(" ✓")
        
        return formatted_response
    
    def chat(self):
        """Interactive chat loop"""
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                response = self.process_query(user_input)
                print(f"\n🤖 Assistant:\n{response}")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Initialize assistant
        assistant = MongoDBChatAssistant(
            mongodb_uri='mongodb://localhost:27017',
            db_name='ai_test_db'
        )
        
        # Start chat
        assistant.chat()
    
    except Exception as e:
        print(f"\n❌ Failed to start assistant: {e}")
        print("\nMake sure:")
        print("  1. MongoDB is running (mongod)")
        print("  2. Database 'ai-test-db' exists")
        print("  3. Ollama is running with Qwen2.5 model")
        print("\nSetup commands:")
        print("  # Start MongoDB")
        print("  mongod")
        print("\n  # Pull and run Ollama")
        print("  ollama pull qwen2.5:3b")
        print("  ollama serve")

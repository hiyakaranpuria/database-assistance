from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# ============================================================================
# STEP 1: MONGODB SCHEMA EXTRACTOR
# ============================================================================

class MongoDBSchemaExtractor:
    """Automatically extract schema from MongoDB collections"""
    
    def __init__(self, connection_string: str = 'mongodb://localhost:27017'):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB connection URI
            Examples:
                - 'mongodb://localhost:27017'
                - 'mongodb+srv://user:pass@cluster.mongodb.net'
                - 'mongodb://user:pass@host:port'
        """
        self.connection_string = connection_string
        self.client = None
        self.db = None
    
    def connect(self, database_name: str):
        """Connect to MongoDB database"""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            print(f"✓ Connected to MongoDB database: {database_name}")
            return self.db
        except Exception as e:
            print(f"✗ Connection error: {e}")
            print(f"  Connection string: {self.connection_string}")
            raise
    
    def extract_collection_schema(self, collection_name: str) -> Dict:
        """
        Extract schema from a single MongoDB collection
        Analyzes field types, indexes, and sample documents
        """
        if not self.db:
            raise ValueError("Database not connected. Call connect() first.")
        
        collection = self.db[collection_name]
        
        # Get collection statistics
        try:
            stats = self.db.command('collStats', collection_name)
            doc_count = stats['count']
            avg_doc_size = stats.get('avgObjSize', 0)
            total_size = stats.get('size', 0)
        except Exception as e:
            print(f"  Warning: Could not get stats for {collection_name}: {e}")
            doc_count = collection.count_documents({})
            avg_doc_size = 0
            total_size = 0
        
        # Get indexes
        try:
            indexes = list(collection.list_indexes())
            index_fields = set()
            for idx in indexes:
                for field, _ in idx['key']:
                    if field != '_id':  # Skip _id index
                        index_fields.add(field.split('.')[0])  # Get top-level field
        except Exception as e:
            print(f"  Warning: Could not get indexes for {collection_name}: {e}")
            index_fields = set()
        
        # Sample documents to infer schema
        try:
            sample_docs = list(collection.find().limit(100))
        except Exception as e:
            print(f"  Warning: Could not sample documents from {collection_name}: {e}")
            sample_docs = []
        
        # Infer field types and descriptions
        field_types = {}
        
        for doc in sample_docs:
            for field_name, value in doc.items():
                if field_name == '_id':
                    continue
                
                field_type = type(value).__name__
                if field_name not in field_types:
                    field_types[field_name] = field_type
        
        # Build field descriptions
        fields = {}
        for field_name, field_type in field_types.items():
            is_indexed = field_name in index_fields
            type_desc = self._map_type_description(field_type)
            
            field_desc = f"Field type: {type_desc}"
            if is_indexed:
                field_desc += " [Indexed]"
            
            fields[field_name] = field_desc
        
        schema = {
            "description": f"MongoDB collection with {len(fields)} fields and {doc_count} documents",
            "fields": fields,
            "doc_count": doc_count,
            "avg_doc_size_bytes": avg_doc_size,
            "indexed_fields": list(index_fields),
            "total_size_bytes": total_size
        }
        
        return schema
    
    def extract_database_schema(self) -> Dict:
        """
        Extract complete schema from all collections in the database
        """
        if not self.db:
            raise ValueError("Database not connected. Call connect() first.")
        
        schema = {}
        collections = self.db.list_collection_names()
        
        print(f"\nExtracting schema from {len(collections)} collections...")
        
        for collection_name in collections:
            # Skip system collections
            if collection_name.startswith('system.'):
                continue
            
            try:
                print(f"  Extracting: {collection_name}...", end=" ")
                collection_schema = self.extract_collection_schema(collection_name)
                schema[collection_name] = collection_schema
                print(f"✓ ({len(collection_schema['fields'])} fields)")
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        return schema
    
    @staticmethod
    def _map_type_description(python_type: str) -> str:
        """Map Python type names to MongoDB/BSON type descriptions"""
        type_map = {
            'str': 'String',
            'int': 'Integer',
            'float': 'Double',
            'bool': 'Boolean',
            'dict': 'Document/Object',
            'list': 'Array',
            'ObjectId': 'ObjectId',
            'datetime': 'Date',
            'NoneType': 'Null',
            'Decimal128': 'Decimal128',
            'bytes': 'Binary',
        }
        return type_map.get(python_type, python_type)


# ============================================================================
# STEP 2: EMBEDDING GENERATOR
# ============================================================================

class EmbeddingGenerator:
    """Generate and manage schema embeddings for MongoDB"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model"""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embeddings_db = {}
        self.model_name = model_name
        print("✓ Model loaded")
    
    def generate_embeddings(self, schema: Dict) -> Dict:
        """Generate embeddings for all collections and fields"""
        print(f"\nGenerating embeddings for {len(schema)} collections...")
        
        for collection_name, collection_info in schema.items():
            # Create embedding text combining collection name and description
            collection_text = f"{collection_name}: {collection_info['description']}"
            collection_embedding = self.model.encode(collection_text)
            
            # Generate embeddings for fields too
            field_embeddings = {}
            for field_name, field_desc in collection_info['fields'].items():
                field_text = f"{field_name}: {field_desc}"
                field_embedding = self.model.encode(field_text)
                field_embeddings[field_name] = field_embedding.tolist()
            
            # Store in embeddings database
            self.embeddings_db[collection_name] = {
                "description": collection_info['description'],
                "collection_embedding": collection_embedding.tolist(),
                "fields": collection_info['fields'],
                "field_embeddings": field_embeddings,
                "doc_count": collection_info['doc_count'],
                "indexed_fields": collection_info['indexed_fields'],
                "avg_doc_size_bytes": collection_info['avg_doc_size_bytes'],
                "total_size_bytes": collection_info['total_size_bytes']
            }
            
            print(f"  ✓ {collection_name}")
        
        print(f"✓ Generated embeddings for all collections")
        return self.embeddings_db
    
    def save_embeddings(self, filepath: str):
        """Save embeddings to binary file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.embeddings_db, f)
        print(f"✓ Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            self.embeddings_db = pickle.load(f)
        print(f"✓ Embeddings loaded from {filepath}")
        return self.embeddings_db
    
    def save_schema_json(self, filepath: str):
        """Save schema details as readable JSON"""
        schema_json = {}
        for collection_name, collection_data in self.embeddings_db.items():
            schema_json[collection_name] = {
                "description": collection_data['description'],
                "fields": collection_data['fields'],
                "doc_count": collection_data['doc_count'],
                "indexed_fields": collection_data['indexed_fields'],
                "avg_doc_size_bytes": collection_data['avg_doc_size_bytes'],
                "total_size_bytes": collection_data['total_size_bytes']
            }
        
        with open(filepath, 'w') as f:
            json.dump(schema_json, f, indent=2)
        print(f"✓ Schema saved as JSON to {filepath}")


# ============================================================================
# STEP 3: SEMANTIC SEARCH
# ============================================================================

class SemanticSchemaSearch:
    """Search for relevant collections using semantic similarity"""
    
    def __init__(self, embeddings_db: Dict):
        self.embeddings_db = embeddings_db
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search_collections(self, user_question: str, top_k: int = 3) -> List[Tuple]:
        """Find most relevant collections for user question"""
        
        # Encode user question
        question_embedding = self.model.encode(user_question)
        question_embedding = np.array(question_embedding)
        
        # Calculate similarity with each collection
        similarities = {}
        for collection_name, collection_data in self.embeddings_db.items():
            collection_embedding = np.array(collection_data['collection_embedding'])
            
            # Cosine similarity
            similarity = self._cosine_similarity(question_embedding, collection_embedding)
            similarities[collection_name] = similarity
        
        # Get top K collections
        top_collections = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return top_collections
    
    def search_fields(self, user_question: str, collection_name: str, top_k: int = 3) -> List[Tuple]:
        """Find most relevant fields within a collection"""
        
        if collection_name not in self.embeddings_db:
            return []
        
        question_embedding = self.model.encode(user_question)
        question_embedding = np.array(question_embedding)
        
        collection_data = self.embeddings_db[collection_name]
        field_embeddings = collection_data['field_embeddings']
        
        # Calculate similarity with each field
        field_similarities = {}
        for field_name, field_embedding in field_embeddings.items():
            field_embedding = np.array(field_embedding)
            similarity = self._cosine_similarity(question_embedding, field_embedding)
            field_similarities[field_name] = similarity
        
        # Get top K fields
        top_fields = sorted(
            field_similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return top_fields
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)


# ============================================================================
# STEP 4: BUILD LLM CONTEXT
# ============================================================================

class LLMContextBuilder:
    """Build context for LLM based on relevant MongoDB schema"""
    
    def __init__(self, embeddings_db: Dict, search: SemanticSchemaSearch):
        self.embeddings_db = embeddings_db
        self.search = search
    
    def build_context(self, user_question: str, top_k_collections: int = 3) -> str:
        """Build schema context for LLM"""
        
        # Find relevant collections
        relevant_collections = self.search.search_collections(user_question, top_k=top_k_collections)
        
        context = "## Relevant MongoDB Schema\n\n"
        
        for collection_name, similarity_score in relevant_collections:
            collection_data = self.embeddings_db[collection_name]
            
            context += f"### Collection: `{collection_name}`\n"
            context += f"**Description:** {collection_data['description']}\n"
            context += f"**Relevance Score:** {similarity_score:.2%}\n"
            context += f"**Document Count:** {collection_data['doc_count']:,}\n"
            
            # Find most relevant fields
            relevant_fields = self.search.search_fields(user_question, collection_name, top_k=5)
            
            context += "**Fields:**\n"
            for field_name, field_similarity in relevant_fields:
                field_desc = collection_data['fields'][field_name]
                context += f"  - `{field_name}`: {field_desc}\n"
            
            # Add indexed fields info
            if collection_data['indexed_fields']:
                context += f"**Indexed Fields:** {', '.join(collection_data['indexed_fields'])}\n"
            
            context += "\n"
        
        return context
    
    def get_full_prompt(self, user_question: str) -> str:
        """Get complete prompt for LLM"""
        schema_context = self.build_context(user_question)
        
        prompt = f"""You are a MongoDB expert. Use the following MongoDB schema to answer the user's question.

{schema_context}

**User Question:** {user_question}

Generate an accurate MongoDB query to answer this question. You can use:
- db.collection.find() for simple queries
- db.collection.aggregate() for complex queries with pipelines
- Only use the collections and fields provided above
- Return just the MongoDB query without any explanation

Examples:
db.users.find({{ email: "user@example.com" }})

db.orders.aggregate([
  {{ $match: {{ status: "completed" }} }},
  {{ $group: {{ _id: "$user_id", total: {{ $sum: "$amount" }} }} }},
  {{ $sort: {{ total: -1 }} }}
])

Now generate the query for: {user_question}"""
        
        return prompt


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete workflow: Extract -> Embed -> Search -> Use"""
    
    print("=" * 80)
    print(" " * 15 + "MONGODB SCHEMA EMBEDDING SYSTEM")
    print(" " * 20 + "AUTOMATED WORKFLOW")
    print("=" * 80)
    
    # Configuration
    MONGODB_URI = 'mongodb://localhost:27017'  # Update if using different connection
    DATABASE_NAME = 'ai-test-db'  # Your existing database
    
    # Step 1: Extract Schema from MongoDB
    print(f"\n[STEP 1] Extracting Schema from MongoDB '{DATABASE_NAME}'...")
    print("-" * 80)
    
    extractor = MongoDBSchemaExtractor(connection_string=MONGODB_URI)
    
    try:
        extractor.connect(database_name=DATABASE_NAME)
        schema = extractor.extract_database_schema()
        
        if not schema:
            print("⚠ No collections found in database!")
            return
        
        print(f"\n✓ Successfully extracted {len(schema)} collections")
        
    except Exception as e:
        print(f"✗ Could not connect to MongoDB")
        print(f"  Error: {e}")
        print(f"\n  Make sure:")
        print(f"    - MongoDB is running at {MONGODB_URI}")
        print(f"    - Database '{DATABASE_NAME}' exists")
        print(f"    - You have network access to MongoDB")
        return
    
    # Step 2: Generate Embeddings
    print(f"\n[STEP 2] Generating Embeddings...")
    print("-" * 80)
    
    embedding_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    embeddings_db = embedding_gen.generate_embeddings(schema)
    
    # Save embeddings for reuse
    embedding_gen.save_embeddings('schema_embeddings.pkl')
    embedding_gen.save_schema_json('schema_info.json')
    
    # Step 3: Initialize Search
    print(f"\n[STEP 3] Initializing Semantic Search...")
    print("-" * 80)
    search = SemanticSchemaSearch(embeddings_db)
    print("✓ Search engine ready")
    
    # Step 4: Demo - Search for relevant collections
    print(f"\n[STEP 4] Testing Semantic Search...")
    print("-" * 80)
    
    test_questions = [
        "What are the recent orders?",
        "Show me all users",
        "Find products with high ratings",
        "List all payment transactions"
    ]
    
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        relevant_collections = search.search_collections(question, top_k=3)
        print("   Top relevant collections:")
        for i, (collection_name, score) in enumerate(relevant_collections, 1):
            print(f"   {i}. {collection_name:20} (relevance: {score:6.1%})")
    
    # Step 5: Build LLM Context
    print(f"\n[STEP 5] Building LLM Context Examples...")
    print("-" * 80)
    
    context_builder = LLMContextBuilder(embeddings_db, search)
    
    # Example 1
    sample_question_1 = "Show me all products"
    prompt_1 = context_builder.get_full_prompt(sample_question_1)
    
    print(f"\n📝 Example 1: {sample_question_1}")
    print("\nGenerated Context (first 500 chars):")
    print("=" * 80)
    print(prompt_1[:500] + "...")
    
    print("\n" + "=" * 80)
    print("✓ WORKFLOW COMPLETE")
    print("=" * 80)
    print("\n📁 Output Files Created:")
    print("  - schema_embeddings.pkl (binary embeddings for fast search)")
    print("  - schema_info.json (human-readable schema information)")
    print("\n💡 Next steps:")
    print("  1. Use the embeddings in your application")
    print("  2. For each user query, call LLMContextBuilder.get_full_prompt()")
    print("  3. Send the prompt to your LLM (OpenAI, Claude, etc.)")
    print("  4. Execute the returned MongoDB query")


if __name__ == "__main__":
    main()

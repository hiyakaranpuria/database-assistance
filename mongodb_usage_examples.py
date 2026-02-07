"""
Real-world Usage Examples for MongoDB Schema Embedding System
Demonstrates how to integrate this into your LLM pipeline
"""

from mongodb_schema_embedding_system import (
    MongoDBSchemaExtractor,
    EmbeddingGenerator,
    SemanticSchemaSearch,
    LLMContextBuilder
)
import json
import pickle


# ============================================================================
# EXAMPLE 1: One-time Setup
# ============================================================================

def setup_schema_embeddings(
    mongodb_uri: str = 'mongodb://localhost:27017',
    database_name: str = 'ai-test-db',
    embeddings_output: str = 'schema_embeddings.pkl'
):
    """
    Run this ONCE to extract schema and generate embeddings from your MongoDB.
    After this, you only load the embeddings for queries.
    """
    print("=" * 70)
    print("SETUP: Extracting MongoDB schema and generating embeddings...")
    print("=" * 70)
    
    # Step 1: Extract schema from your database
    extractor = MongoDBSchemaExtractor(connection_string=mongodb_uri)
    
    try:
        extractor.connect(database_name=database_name)
        schema = extractor.extract_database_schema()
        
        if not schema:
            print("⚠ No collections found in database!")
            return None
        
        print(f"\n✓ Extracted {len(schema)} collections")
        print("  Collections found:")
        for collection_name in schema.keys():
            cols = len(schema[collection_name]['fields'])
            docs = schema[collection_name]['doc_count']
            print(f"    - {collection_name} ({cols} fields, {docs:,} documents)")
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None
    
    # Step 2: Generate and save embeddings
    embedding_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    embeddings_db = embedding_gen.generate_embeddings(schema)
    
    embedding_gen.save_embeddings(embeddings_output)
    embedding_gen.save_schema_json(embeddings_output.replace('.pkl', '.json'))
    
    print(f"\n✓ Embeddings saved to: {embeddings_output}")
    print("✓ Schema info saved to: {embeddings_output.replace('.pkl', '.json')}")
    print("\n✓ Setup Complete! Ready for queries.")
    
    return embeddings_db


# ============================================================================
# EXAMPLE 2: Query Processing (Use Many Times)
# ============================================================================

def process_user_query(user_query: str, embeddings_path: str = 'schema_embeddings.pkl') -> str:
    """
    Process a user query and return LLM prompt with relevant schema.
    This is what you call repeatedly for each user question.
    
    Args:
        user_query: User's natural language question
        embeddings_path: Path to saved embeddings file
    
    Returns:
        LLM prompt with relevant MongoDB schema
    """
    
    try:
        # Load pre-generated embeddings (fast - just unpickling)
        with open(embeddings_path, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Embeddings file not found: {embeddings_path}")
        print("  Run setup_schema_embeddings() first")
        return None
    
    # Initialize search
    search = SemanticSchemaSearch(embeddings_db)
    
    # Build context with relevant schema only
    context_builder = LLMContextBuilder(embeddings_db, search)
    prompt = context_builder.get_full_prompt(user_query)
    
    # Show what collections were selected
    relevant_collections = search.search_collections(user_query, top_k=3)
    print(f"\n📝 Query: {user_query}")
    print("   Selected collections:")
    for collection, score in relevant_collections:
        print(f"   - {collection} ({score:.1%} relevance)")
    
    return prompt


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Queries
# ============================================================================

def batch_process_queries(queries: list, embeddings_path: str = 'schema_embeddings.pkl'):
    """
    Process multiple queries efficiently using same embeddings.
    
    Args:
        queries: List of user questions
        embeddings_path: Path to saved embeddings file
    
    Returns:
        List of prompts ready for LLM
    """
    
    # Load embeddings ONCE
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Embeddings file not found: {embeddings_path}")
        return []
    
    search = SemanticSchemaSearch(embeddings_db)
    context_builder = LLMContextBuilder(embeddings_db, search)
    
    results = []
    
    for query in queries:
        prompt = context_builder.get_full_prompt(query)
        results.append({
            'query': query,
            'prompt': prompt,
            'relevant_collections': [
                t for t, _ in search.search_collections(query, top_k=3)
            ]
        })
    
    print(f"\n✓ Processed {len(results)} queries")
    return results


# ============================================================================
# EXAMPLE 4: Integration with LLM (OpenAI, Anthropic, etc.)
# ============================================================================

def query_with_llm(
    user_question: str,
    embeddings_path: str = 'schema_embeddings.pkl',
    llm_api_key: str = None,
    llm_model: str = 'gpt-4'
):
    """
    Complete flow: Get schema context -> Send to LLM -> Return MongoDB Query
    
    Requires: pip install openai
    
    Args:
        user_question: User's natural language question
        embeddings_path: Path to saved embeddings
        llm_api_key: Your API key (set as env var for security)
        llm_model: LLM model to use
    
    Returns:
        MongoDB query generated by LLM
    """
    
    # Get schema context
    prompt = process_user_query(user_question, embeddings_path)
    
    if not prompt:
        return None
    
    print(f"\n{'-' * 70}")
    print("CONTEXT SENT TO LLM:")
    print(f"{'-' * 70}")
    print(prompt[:500] + "...[truncated]")
    print(f"{'-' * 70}\n")
    
    # In real usage, uncomment and use actual LLM:
    # try:
    #     from openai import OpenAI
    #     client = OpenAI(api_key=llm_api_key or os.getenv("OPENAI_API_KEY"))
    #     
    #     response = client.chat.completions.create(
    #         model=llm_model,
    #         messages=[
    #             {"role": "system", "content": "You are a MongoDB expert. Generate valid MongoDB queries."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0,
    #         max_tokens=500
    #     )
    #     
    #     mongodb_query = response.choices[0].message.content
    #     return mongodb_query
    # except Exception as e:
    #     print(f"Error calling LLM: {e}")
    #     return None


# ============================================================================
# EXAMPLE 5: Caching for Performance
# ============================================================================

class QueryCache:
    """Cache prompts to avoid regenerating for same questions"""
    
    def __init__(self, cache_file: str = 'query_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_or_generate(self, user_query: str, embeddings_path: str = 'schema_embeddings.pkl') -> str:
        """Return cached prompt or generate new one"""
        
        # Simple hash of query
        query_hash = str(hash(user_query))
        
        if query_hash in self.cache:
            print(f"✓ Using cached prompt for: {user_query}")
            return self.cache[query_hash]
        
        # Generate new prompt
        prompt = process_user_query(user_query, embeddings_path)
        
        if not prompt:
            return None
        
        # Cache it
        self.cache[query_hash] = prompt
        self._save_cache()
        
        return prompt


# ============================================================================
# EXAMPLE 6: Monitor & Adjust Relevance
# ============================================================================

def analyze_relevance(
    embeddings_path: str = 'schema_embeddings.pkl',
    queries: list = None
):
    """Analyze if selected collections are relevant for your queries"""
    
    if queries is None:
        queries = [
            "Show all users",
            "Find recent orders",
            "Get product ratings"
        ]
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Embeddings file not found: {embeddings_path}")
        return
    
    search = SemanticSchemaSearch(embeddings_db)
    
    print("=" * 70)
    print("RELEVANCE ANALYSIS")
    print("=" * 70)
    
    for query in queries:
        print(f"\n📝 {query}")
        
        # Get all collections with scores
        question_embedding = search.model.encode(query)
        similarities = {}
        
        for collection_name, collection_data in embeddings_db.items():
            import numpy as np
            collection_embedding = np.array(collection_data['collection_embedding'])
            similarity = np.dot(question_embedding, collection_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(collection_embedding)
            )
            similarities[collection_name] = similarity
        
        # Show all collections sorted by relevance
        sorted_collections = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (collection, score) in enumerate(sorted_collections, 1):
            threshold = "✓" if score > 0.75 else "△" if score > 0.5 else "✗"
            print(f"  {i}. {collection:20} {score:6.1%}  {threshold}")


# ============================================================================
# EXAMPLE 7: Updating Schema When Collections Change
# ============================================================================

def update_embeddings(
    mongodb_uri: str = 'mongodb://localhost:27017',
    database_name: str = 'ai-test-db',
    embeddings_output: str = 'schema_embeddings.pkl'
):
    """
    Regenerate embeddings when MongoDB schema changes.
    You might call this nightly or when collections are added/modified.
    """
    
    print("Updating embeddings...")
    
    # Re-extract schema
    extractor = MongoDBSchemaExtractor(connection_string=mongodb_uri)
    
    try:
        extractor.connect(database_name=database_name)
        schema = extractor.extract_database_schema()
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Re-generate embeddings
    embedding_gen = EmbeddingGenerator()
    embeddings_db = embedding_gen.generate_embeddings(schema)
    
    # Save
    embedding_gen.save_embeddings(embeddings_output)
    
    print(f"✓ Embeddings updated: {embeddings_output}")


# ============================================================================
# EXAMPLE 8: Connection to Different MongoDB Instances
# ============================================================================

def connect_to_different_mongodb(examples_only: bool = True):
    """Examples of connecting to different MongoDB instances"""
    
    examples = {
        'Local': {
            'uri': 'mongodb://localhost:27017',
            'db': 'ai-test-db'
        },
        'Atlas': {
            'uri': 'mongodb+srv://user:password@cluster.mongodb.net',
            'db': 'ai-test-db'
        },
        'Custom Host': {
            'uri': 'mongodb://user:pass@hostname:27017',
            'db': 'ai-test-db'
        },
        'With Credentials': {
            'uri': 'mongodb://user:password@localhost:27017/authSource=admin',
            'db': 'ai-test-db'
        }
    }
    
    print("Available MongoDB Connection Options:")
    print("-" * 70)
    
    for name, config in examples.items():
        print(f"\n{name}:")
        print(f"  URI: {config['uri']}")
        print(f"  DB:  {config['db']}")
    
    if examples_only:
        print("\nTo use any of these, update the connection_string parameter in your code")
    
    return examples


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """
    Complete workflow showing all examples
    """
    
    print("\n" + "=" * 70)
    print("MONGODB SCHEMA EMBEDDING SYSTEM - USAGE EXAMPLES")
    print("=" * 70)
    
    # Configuration for your existing database
    MONGODB_URI = 'mongodb://localhost:27017'
    DATABASE_NAME = 'ai-test-db'
    EMBEDDINGS_FILE = 'schema_embeddings.pkl'
    
    # -------- EXAMPLE 1: One-time Setup --------
    print("\n[EXAMPLE 1] One-time Setup")
    print("-" * 70)
    print("Run this once to extract schema from your MongoDB database...")
    
    embeddings_db = setup_schema_embeddings(
        mongodb_uri=MONGODB_URI,
        database_name=DATABASE_NAME,
        embeddings_output=EMBEDDINGS_FILE
    )
    
    if not embeddings_db:
        print("\n✗ Setup failed. Check MongoDB connection and try again.")
        return
    
    # -------- EXAMPLE 2: Process Single Query --------
    print("\n[EXAMPLE 2] Process Single Query")
    print("-" * 70)
    
    prompt = process_user_query(
        user_query="Find all active users",
        embeddings_path=EMBEDDINGS_FILE
    )
    
    if prompt:
        print(f"\nGenerated prompt for LLM:\n{prompt[:300]}...")
    
    # -------- EXAMPLE 3: Batch Process Multiple Queries --------
    print("\n[EXAMPLE 3] Batch Process Multiple Queries")
    print("-" * 70)
    
    test_queries = [
        "Show recent orders",
        "List high-rated products",
        "Find payment failures",
    ]
    
    results = batch_process_queries(test_queries, EMBEDDINGS_FILE)
    
    print(f"Processed {len(results)} queries")
    for result in results:
        print(f"  - {result['query']}")
        print(f"    Collections: {', '.join(result['relevant_collections'])}")
    
    # -------- EXAMPLE 5: Caching --------
    print("\n[EXAMPLE 5] Using Query Cache")
    print("-" * 70)
    
    cache = QueryCache()
    
    # First call - generates
    prompt1 = cache.get_or_generate(
        "Show all users",
        EMBEDDINGS_FILE
    )
    
    # Second call - uses cache
    prompt2 = cache.get_or_generate(
        "Show all users",
        EMBEDDINGS_FILE
    )
    
    # -------- EXAMPLE 6: Relevance Analysis --------
    print("\n[EXAMPLE 6] Analyze Relevance Scores")
    print("-" * 70)
    
    analyze_relevance(
        EMBEDDINGS_FILE,
        [
            "Find users by country",
            "Get product inventory",
            "Show payment methods"
        ]
    )
    
    # -------- EXAMPLE 8: Connection Options --------
    print("\n[EXAMPLE 8] Available MongoDB Connections")
    print("-" * 70)
    
    connect_to_different_mongodb(examples_only=True)
    
    print("\n" + "=" * 70)
    print("✓ ALL EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Embeddings are saved - ready for repeated queries")
    print("  2. Call process_user_query() for each user question")
    print("  3. Send returned prompt to your LLM")
    print("  4. Execute the MongoDB query returned by LLM")


if __name__ == "__main__":
    main()

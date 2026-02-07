"""
Comprehensive diagnostic for MongoDB Chat Agent
"""
import sys
print("=" * 70)
print("MONGODB CHAT AGENT DIAGNOSTIC")
print("=" * 70)

# Test 1: MongoDB Connection
print("\n[1/6] Testing MongoDB Connection...")
try:
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017', serverSelectionTimeoutMS=2000)
    client.admin.command('ping')
    print("✓ MongoDB is running")
    
    db = client['ai_test_db']
    collections = db.list_collection_names()
    print(f"✓ Database 'ai_test_db' found with {len(collections)} collections")
    print(f"  Collections: {', '.join(collections)}")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")
    sys.exit(1)

# Test 2: Check data exists
print("\n[2/6] Checking data in collections...")
for coll in ['customers', 'products', 'orders']:
    if coll in collections:
        count = db[coll].count_documents({})
        sample = db[coll].find_one()
        print(f"✓ {coll}: {count} documents")
        if sample:
            print(f"  Sample fields: {', '.join(list(sample.keys())[:5])}")

# Test 3: Sentence Transformers
print("\n[3/6] Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = model.encode("test")
    print(f"✓ Sentence Transformers loaded (embedding size: {len(test_embedding)})")
except Exception as e:
    print(f"✗ Sentence Transformers failed: {e}")
    sys.exit(1)

# Test 4: Generate fresh embeddings
print("\n[4/6] Generating fresh embeddings...")
try:
    import os
    if os.path.exists('embeddings.pkl'):
        os.remove('embeddings.pkl')
        print("  Deleted old embeddings.pkl")
    
    from mongo_chat_agent import MongoDBConnector, VectorSearchEngine
    
    connector = MongoDBConnector('mongodb://localhost:27017')
    connector.connect('ai_test_db')
    
    schema = connector.extract_schema()
    print(f"✓ Extracted schema for {len(schema)} collections")
    
    vector_search = VectorSearchEngine()
    embeddings = vector_search.generate_embeddings(schema)
    vector_search.save_embeddings('embeddings.pkl')
    print(f"✓ Generated and saved embeddings")
    
except Exception as e:
    print(f"✗ Embedding generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test vector search
print("\n[5/6] Testing vector search...")
try:
    import numpy as np
    
    test_queries = [
        "Find all products with a price less than 500",
        "Count the number of users from Mumbai",
        "Show me all customers"
    ]
    
    for query in test_queries:
        question_embedding = model.encode(query)
        
        similarities = {}
        for coll_name, coll_data in embeddings.items():
            coll_embedding = np.array(coll_data['collection_embedding'])
            similarity = np.dot(question_embedding, coll_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(coll_embedding)
            )
            similarities[coll_name] = similarity
        
        top_match = max(similarities.items(), key=lambda x: x[1])
        threshold_pass = "PASS" if top_match[1] >= 0.2 else "FAIL"
        
        print(f"\n  Query: '{query}'")
        print(f"  Top match: {top_match[0]} (similarity: {top_match[1]:.4f}) [{threshold_pass}]")
        
        if threshold_pass == "FAIL":
            print(f"  ⚠ WARNING: Below 0.2 threshold!")
            print(f"  All similarities: {similarities}")

except Exception as e:
    print(f"✗ Vector search test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test full agent
print("\n[6/6] Testing MongoDBChatAgent...")
try:
    from mongo_chat_agent import MongoDBChatAgent
    
    agent = MongoDBChatAgent()
    
    test_query = "Show me all customers"
    print(f"\n  Testing query: '{test_query}'")
    result = agent.process_query(test_query)
    
    if "not related to the database" in result:
        print(f"✗ Agent rejected query as out of scope")
        print(f"  Response: {result}")
    else:
        print(f"✓ Agent processed query successfully")
        print(f"  Response preview: {result[:200]}...")
        
except Exception as e:
    print(f"✗ Agent test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

# MongoDB Schema Embedding System - Quick Start Guide

## Overview

This system automatically extracts schema from your existing MongoDB database (`ai-test-db`), generates semantic embeddings, and enables intelligent search to find relevant collections for LLM-generated MongoDB queries.

## Installation

```bash
# Install required packages
pip install sentence-transformers pymongo numpy
```

## Quick Start (5 Minutes)

### Step 1: Extract Schema & Generate Embeddings (One-time)

```python
from mongodb_schema_embedding_system import (
    MongoDBSchemaExtractor,
    EmbeddingGenerator
)

# Extract schema from your database
extractor = MongoDBSchemaExtractor(connection_string='mongodb://localhost:27017')
extractor.connect(database_name='ai-test-db')
schema = extractor.extract_database_schema()

# Generate embeddings
embedding_gen = EmbeddingGenerator()
embeddings_db = embedding_gen.generate_embeddings(schema)

# Save for later use
embedding_gen.save_embeddings('schema_embeddings.pkl')
embedding_gen.save_schema_json('schema_info.json')
```

### Step 2: Process User Queries (Run Repeatedly)

```python
import pickle
from mongodb_schema_embedding_system import SemanticSchemaSearch, LLMContextBuilder

# Load pre-generated embeddings (fast!)
with open('schema_embeddings.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

# Search and build context
search = SemanticSchemaSearch(embeddings_db)
context_builder = LLMContextBuilder(embeddings_db, search)

# For each user question
user_question = "Show me all products with prices above $100"
prompt = context_builder.get_full_prompt(user_question)

print(prompt)  # Ready to send to LLM
```

### Step 3: Send to LLM (OpenAI, Claude, etc.)

```python
# Using OpenAI (example)
from openai import OpenAI

client = OpenAI(api_key="your-api-key")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a MongoDB expert."},
        {"role": "user", "content": prompt}
    ]
)

mongodb_query = response.choices[0].message.content
print(f"Generated MongoDB query:\n{mongodb_query}")
```

## How It Works

```
Your MongoDB Database (ai-test-db)
    ↓
[Extract Collections & Fields]
    ↓
[Generate Semantic Embeddings]
    ↓
User Question: "Show me all users from New York"
    ↓
[Find Most Relevant Collections]
    ↓
Context: Only include `users` collection
    ↓
[Build Context for LLM]
    ↓
Send to LLM (OpenAI, Claude, Anthropic)
    ↓
MongoDB Query: db.users.find({city: "New York"})
```

## Configuration

### MongoDB Connection

**Local MongoDB:**
```python
extractor = MongoDBSchemaExtractor(
    connection_string='mongodb://localhost:27017'
)
```

**MongoDB Atlas:**
```python
extractor = MongoDBSchemaExtractor(
    connection_string='mongodb+srv://user:password@cluster.mongodb.net'
)
```

**Custom Host with Authentication:**
```python
extractor = MongoDBSchemaExtractor(
    connection_string='mongodb://user:password@hostname:27017/authSource=admin'
)
```

### Embedding Model

```python
# Fast model (default, 384 dims)
embedding_gen = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')

# More accurate model (768 dims)
embedding_gen = EmbeddingGenerator(model_name='all-mpnet-base-v2')

# Multilingual model
embedding_gen = EmbeddingGenerator(model_name='multilingual-e5-small')
```

### Search Parameters

```python
# Select more collections (3-5 recommended)
relevant_collections = search.search_collections(question, top_k=5)

# Select fewer fields per collection (3-5 recommended)
relevant_fields = search.search_fields(question, collection_name, top_k=3)
```

## File Structure

```
project/
├── mongodb_schema_embedding_system.py    # Main system
├── mongodb_usage_examples.py             # Usage examples
├── MONGODB_QUICK_START.md               # This file
├── MONGODB_ARCHITECTURE.md              # System design
├── README.md                            # Project overview
│
├── schema_embeddings.pkl                # Generated - Binary embeddings
├── schema_info.json                     # Generated - Readable schema
└── query_cache.json                     # Generated - Query cache
```

## Common Tasks

### Task 1: Extract Schema Once

```python
from mongodb_schema_embedding_system import MongoDBSchemaExtractor, EmbeddingGenerator

extractor = MongoDBSchemaExtractor('mongodb://localhost:27017')
extractor.connect('ai-test-db')
schema = extractor.extract_database_schema()

embedding_gen = EmbeddingGenerator()
embeddings_db = embedding_gen.generate_embeddings(schema)
embedding_gen.save_embeddings('schema_embeddings.pkl')
```

**Output:**
```
✓ Connected to MongoDB database: ai-test-db
Extracting schema from 5 collections...
  Extracting: users... ✓ (12 fields)
  Extracting: products... ✓ (8 fields)
  Extracting: orders... ✓ (10 fields)
  Extracting: payments... ✓ (7 fields)
  Extracting: reviews... ✓ (6 fields)

✓ Successfully extracted 5 collections

Loading embedding model: all-MiniLM-L6-v2...
✓ Model loaded

Generating embeddings for 5 collections...
  ✓ users
  ✓ products
  ✓ orders
  ✓ payments
  ✓ reviews

✓ Generated embeddings for all collections
✓ Embeddings saved to schema_embeddings.pkl
```

### Task 2: Find Relevant Collections for a Question

```python
import pickle
from mongodb_schema_embedding_system import SemanticSchemaSearch

with open('schema_embeddings.pkl', 'rb') as f:
    embeddings_db = pickle.load(f)

search = SemanticSchemaSearch(embeddings_db)

question = "What are the recent orders from customers?"
relevant = search.search_collections(question, top_k=3)

for collection_name, similarity_score in relevant:
    print(f"{collection_name}: {similarity_score:.1%}")
```

**Output:**
```
orders: 87.5%
users: 72.3%
products: 65.1%
```

### Task 3: Find Relevant Fields in a Collection

```python
fields = search.search_fields(
    "Find products with high ratings",
    collection_name='products',
    top_k=5
)

for field_name, similarity_score in fields:
    print(f"{field_name}: {similarity_score:.1%}")
```

**Output:**
```
rating: 92.3%
product_id: 78.5%
name: 65.2%
price: 42.1%
```

### Task 4: Build Context for LLM

```python
from mongodb_schema_embedding_system import LLMContextBuilder

context_builder = LLMContextBuilder(embeddings_db, search)

question = "Show me products with ratings above 4 stars"
context = context_builder.build_context(question, top_k_collections=3)

print(context)
```

**Output:**
```
## Relevant MongoDB Schema

### Collection: `products`
**Description:** MongoDB collection with 8 fields and 100 documents
**Relevance Score:** 95.23%
**Document Count:** 100

**Fields:**
  - `_id`: Field type: ObjectId [Indexed]
  - `name`: Field type: String
  - `rating`: Field type: Double [Indexed]
  - `price`: Field type: Double

**Indexed Fields:** _id, rating

### Collection: `reviews`
...
```

### Task 5: Get Full Prompt for LLM

```python
prompt = context_builder.get_full_prompt(
    "Show me products with ratings above 4 stars"
)

# Send to LLM
response = llm.generate(prompt)  # Your LLM API call
mongodb_query = response

# Execute query
result = db.execute(mongodb_query)
```

### Task 6: Batch Process Multiple Queries

```python
from mongodb_usage_examples import batch_process_queries

queries = [
    "Find all active users",
    "Show recent orders",
    "List high-rated products",
    "Get payment statistics"
]

results = batch_process_queries(queries, 'schema_embeddings.pkl')

for result in results:
    print(f"Query: {result['query']}")
    print(f"Collections: {result['relevant_collections']}")
    # Send result['prompt'] to LLM
```

### Task 7: Update Embeddings When Schema Changes

```python
from mongodb_usage_examples import update_embeddings

# Run this when you add/modify collections
update_embeddings(
    mongodb_uri='mongodb://localhost:27017',
    database_name='ai-test-db',
    embeddings_output='schema_embeddings.pkl'
)
```

### Task 8: Cache Queries for Performance

```python
from mongodb_usage_examples import QueryCache

cache = QueryCache('query_cache.json')

# First call - generates and caches
prompt1 = cache.get_or_generate(
    "Find users from New York",
    'schema_embeddings.pkl'
)

# Second call - uses cache (instant)
prompt2 = cache.get_or_generate(
    "Find users from New York",
    'schema_embeddings.pkl'
)
```

## Troubleshooting

### Issue: "Connection refused" when connecting to MongoDB

**Solution:**
- Make sure MongoDB is running: `mongod`
- Update connection string to match your setup
- For MongoDB Atlas: Use `mongodb+srv://user:pass@cluster.mongodb.net`

### Issue: "No collections found"

**Solution:**
- Verify database name is correct (use `ai-test-db`)
- Check that collections exist: `db.getCollectionNames()`
- Ensure you have read permissions on the database

### Issue: Low relevance scores for some questions

**Solution:**
- Write more descriptive collection names
- Increase `top_k` parameter to see more results
- Use a more accurate embedding model (all-mpnet-base-v2)

### Issue: Embeddings file too large

**Solution:**
- Normal for large schemas (1-5MB for 50 collections)
- File is binary pickle format - will decompress on load
- Can split into multiple database files if needed

## Performance Tips

1. **Generate embeddings once:** Save to disk, load for each query (~30ms)
2. **Use caching:** Avoid regenerating prompts for identical questions
3. **Batch process:** Process multiple queries at once
4. **Adjust top_k:** Use smaller values for speed, larger for coverage
5. **Use faster model:** 'all-MiniLM-L6-v2' is fastest

## Example Output

When you run `python mongodb_schema_embedding_system.py`:

```
================================================================================
                 MONGODB SCHEMA EMBEDDING SYSTEM
                     AUTOMATED WORKFLOW
================================================================================

[STEP 1] Extracting Schema from MongoDB 'ai-test-db'...
--------------------------------------------------------------------------------
✓ Connected to MongoDB database: ai-test-db

Extracting schema from 5 collections...
  Extracting: users... ✓ (12 fields)
  Extracting: products... ✓ (8 fields)
  Extracting: orders... ✓ (10 fields)
  Extracting: payments... ✓ (7 fields)
  Extracting: reviews... ✓ (6 fields)

✓ Successfully extracted 5 collections

[STEP 2] Generating Embeddings...
--------------------------------------------------------------------------------
Loading embedding model: all-MiniLM-L6-v2...
✓ Model loaded

Generating embeddings for 5 collections...
  ✓ users
  ✓ products
  ✓ orders
  ✓ payments
  ✓ reviews

✓ Generated embeddings for all collections
✓ Embeddings saved to schema_embeddings.pkl
✓ Schema saved as JSON to schema_info.json

[STEP 3] Initializing Semantic Search...
--------------------------------------------------------------------------------
✓ Search engine ready

[STEP 4] Testing Semantic Search...
--------------------------------------------------------------------------------

📝 Question: What are the recent orders?
   Top relevant collections:
   1. orders                (relevance:  87.5%)
   2. users                 (relevance:  72.3%)
   3. products              (relevance:  65.1%)

...

================================================================================
✓ WORKFLOW COMPLETE
================================================================================

📁 Output Files Created:
  - schema_embeddings.pkl (binary embeddings for fast search)
  - schema_info.json (human-readable schema information)

💡 Next steps:
  1. Use the embeddings in your application
  2. For each user query, call LLMContextBuilder.get_full_prompt()
  3. Send the prompt to your LLM (OpenAI, Claude, etc.)
  4. Execute the returned MongoDB query
```

## API Reference

See `MONGODB_ARCHITECTURE.md` for:
- Complete class documentation
- Method signatures
- Return types
- Advanced configuration

## Next Steps

1. ✓ Install dependencies: `pip install sentence-transformers pymongo`
2. ✓ Run extraction: `python mongodb_schema_embedding_system.py`
3. ✓ Check generated files: `schema_embeddings.pkl`, `schema_info.json`
4. ✓ Integrate into your application using examples above
5. ✓ Connect your LLM (OpenAI, Claude, Anthropic)
6. ✓ Execute returned MongoDB queries

---

For more details, see:
- `MONGODB_ARCHITECTURE.md` - System design and internals
- `mongodb_usage_examples.py` - Full usage examples
- `README.md` - Project overview

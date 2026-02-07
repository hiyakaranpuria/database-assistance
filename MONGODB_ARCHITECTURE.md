# MongoDB Schema Embedding System - Architecture & Flow

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   Your MongoDB Database                         │
│                       (ai-test-db)                              │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  users   │  │products  │  │ orders   │  │payments  │ ...  │
│  │(N docs)  │  │(M docs)  │  │(P docs)  │  │(Q docs)  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
            [MongoDBSchemaExtractor]
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│          EXTRACTED SCHEMA (Python Dictionary)                  │
│                                                                  │
│  {                                                              │
│    "users": {                                                  │
│      "description": "MongoDB collection with X fields...",     │
│      "fields": {                                               │
│        "_id": "Field type: ObjectId [Indexed]",               │
│        "email": "Field type: String [Indexed]",               │
│        "name": "Field type: String",                          │
│        ...                                                      │
│      },                                                         │
│      "doc_count": 5000,                                        │
│      "indexed_fields": ["_id", "email"],                       │
│      "avg_doc_size_bytes": 512,                                │
│      "total_size_bytes": 2560000                               │
│    },                                                           │
│    "orders": { ... },                                          │
│    ...                                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                          ↓
              [EmbeddingGenerator]
           (sentence-transformers model)
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│       GENERATED EMBEDDINGS (Vector Representations)            │
│                                                                  │
│  embeddings_db = {                                             │
│    "users": {                                                  │
│      "collection_embedding": [0.123, -0.456, ...],  (384 dims)│
│      "fields": {                                               │
│        "_id": "...",                                           │
│        "email": "...",                                         │
│      },                                                         │
│      "field_embeddings": {                                     │
│        "_id": [0.234, -0.567, ...],                           │
│        "email": [0.345, -0.678, ...],                         │
│      },                                                         │
│      "description": "...",                                     │
│      "doc_count": 5000,                                        │
│      "indexed_fields": ["_id", "email"],                       │
│      ...                                                        │
│    },                                                           │
│    "orders": { ... },                                          │
│    ...                                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                          ↓
                [Save to Disk]
                          ↓
          ┌──────────────┴──────────────┐
          ↓                              ↓
schema_embeddings.pkl          schema_info.json
(Binary - Fast Load)          (Human Readable)
(1-5MB typical)               (JSON format)


═════════════════════════════════════════════════════════════════════
                    QUERY TIME WORKFLOW
              (Use Saved Embeddings - Very Fast)
═════════════════════════════════════════════════════════════════════

        [USER ASKS A QUESTION]
              ↓
    "Show me products with
     ratings above 4 stars"
              ↓
    [Load schema_embeddings.pkl]  ← Fast! Pre-computed
          (~10-20ms)
              ↓
        [SemanticSchemaSearch]
        (Find similar collections)
              ↓
    ┌─────────────┬──────────────┬──────────────┐
    ↓             ↓              ↓              ↓
 products      reviews        orders         users
  (95.2%)       (87.3%)        (42.1%)        (35.5%)
                   ↓
      ┌────────────┴────────────┐
      ↓                          ↓
 [Select Top 3]        [Find Relevant Fields]
      ↓                          ↓
 products, reviews,      product_id, name,
 orders                   rating, comment
      ↓
  [LLMContextBuilder]
      ↓
┌──────────────────────────────────────────────┐
│    CONTEXT FOR LLM (Concise & Focused)      │
│                                              │
│  You are a MongoDB expert. Use this schema  │
│  to answer the user's question.             │
│                                              │
│  ## Relevant MongoDB Schema                  │
│                                              │
│  ### Collection: `products`                  │
│  Description: MongoDB collection with...    │
│  Relevance Score: 95.23%                    │
│  Document Count: 1,000                      │
│  Fields:                                    │
│    - product_id: Field type: ObjectId...    │
│    - name: Field type: String               │
│    - rating: Field type: Double             │
│    - price: Field type: Double              │
│                                              │
│  ### Collection: `reviews`                   │
│  Description: MongoDB collection with...    │
│  Relevance Score: 87.34%                    │
│  Fields:                                    │
│    - review_id: Field type: ObjectId...     │
│    - rating: Field type: Integer            │
│    - comment: Field type: String            │
│                                              │
│  User Question: Show me products with       │
│  ratings above 4 stars...                   │
└──────────────────────────────────────────────┘
      ↓
  [Send to LLM]
  (OpenAI, Anthropic, etc.)
      ↓
┌──────────────────────────────────────────────┐
│      LLM GENERATES MONGODB QUERY             │
│                                              │
│  db.products.aggregate([                    │
│    { $match: { rating: { $gt: 4 } } },     │
│    { $sort: { rating: -1 } }                │
│  ])                                         │
│                                              │
│  OR:                                         │
│                                              │
│  db.products.find({ rating: { $gt: 4 } })  │
│    .sort({ rating: -1 })                    │
└──────────────────────────────────────────────┘
      ↓
  [Execute Query on MongoDB]
      ↓
[Return Results to User]
```

## Phase 1: Setup (One-time, ~5-10 seconds)

```
MongoDB ai-test-db
      │
      ├─→ MongoDBSchemaExtractor.connect()
      │         └─→ Establish connection
      │
      └─→ MongoDBSchemaExtractor.extract_database_schema()
                  │
                  ├─→ For each collection:
                  │     ├─→ Get collection stats (count, size)
                  │     ├─→ List indexes
                  │     ├─→ Sample 100 documents
                  │     └─→ Infer field types
                  │
                  └─→ Return schema dict
                          │
                          └─→ EmbeddingGenerator.generate_embeddings()
                                 │
                                 ├─→ For each collection:
                                 │     ├─→ Encode collection name + description
                                 │     ├─→ For each field:
                                 │     │     └─→ Encode field name + type
                                 │     └─→ Store 384-dim vectors
                                 │
                                 └─→ Save embeddings
                                     ├─→ schema_embeddings.pkl (binary)
                                     └─→ schema_info.json (readable)
```

### Setup Timeline
- Extract schema: 1-2 seconds
- Generate embeddings: 2-5 seconds
- Save to disk: <1 second
- **Total: 5-10 seconds (one-time)**

## Phase 2: Query Processing (Per query, ~30-50ms)

```
User Question
    │
    ├─→ Load embeddings ────────→ pickle.load() [10-20ms]
    │
    ├─→ SemanticSchemaSearch.search_collections()
    │     │
    │     ├─→ Encode question ──→ model.encode() [5-10ms]
    │     │
    │     ├─→ For each collection:
    │     │     └─→ Calculate cosine_similarity() [<1ms per collection]
    │     │
    │     └─→ Sort & return top_k [<1ms]
    │
    ├─→ For each selected collection:
    │     └─→ SemanticSchemaSearch.search_fields()
    │           ├─→ Calculate similarity for fields
    │           └─→ Return top_k fields
    │
    ├─→ LLMContextBuilder.build_context()
    │     └─→ Format as readable text [1-2ms]
    │
    └─→ Return prompt
```

### Query Timeline
- Load embeddings: 10-20ms (cached in memory)
- Encode question: 5-10ms
- Calculate similarities: 2-5ms
- Build context: 1-2ms
- **Total: 30-50ms per query** (before LLM)

## How Semantic Search Works

### 1. Embedding Space

All text is converted to 384-dimensional vectors in shared semantic space:

```
Question: "Show me all users from New York"
              ↓
    [Encode with SentenceTransformer]
              ↓
    [0.123, -0.456, 0.789, ..., 0.234]  (384 dimensions)


Collection: "users: MongoDB collection with user account information"
              ↓
    [Encode with SentenceTransformer]
              ↓
    [0.145, -0.412, 0.812, ..., 0.267]

          ↓ Cosine Similarity = 0.952 (High! Similar meaning)


Collection: "products: MongoDB collection with product catalog"
              ↓
    [Encode with SentenceTransformer]
              ↓
    [0.012, -0.765, 0.234, ..., -0.123]

          ↓ Cosine Similarity = 0.234 (Low - Different meaning)
```

### 2. Cosine Similarity Formula

```
           A · B
cos(θ) = ─────────────
         ||A|| × ||B||

Where:
  A = Question vector
  B = Collection vector
  · = Dot product
  || || = Vector magnitude

Result: 0.0 to 1.0
  - 1.0 = Identical meaning
  - 0.75 = Very relevant
  - 0.5 = Somewhat relevant
  - 0.0 = Completely different
```

### 3. Why This Works Better Than Keyword Matching

```
Question: "Show me recent purchases"

Collection: "orders"
  ✓ Keyword match: ✗ (no keyword match)
  ✓ Semantic match: ✓✓✓ (purchases ~ orders)
  ✓ Similarity: 0.89 ← Selected!

Collection: "users"
  ✗ Keyword match: ✗ (no match)
  ✗ Semantic match: ✓ (slightly related)
  ✗ Similarity: 0.42

Collection: "payments"
  ✗ Keyword match: ✗ (no match)
  ✗ Semantic match: ✓ (related to purchases)
  ✗ Similarity: 0.61
```

## Data Flow: Detailed Steps

### Extraction Phase

```
1. Connect to MongoDB
   └─→ Verify credentials
   └─→ Select database 'ai-test-db'

2. Get list of collections
   └─→ Skip system collections (system.*)

3. For EACH collection:
   a) Get Statistics
      └─→ Collection.count_documents()
      └─→ db.command('collStats', collection_name)
   
   b) Get Indexes
      └─→ collection.list_indexes()
      └─→ Extract indexed field names
   
   c) Sample Documents
      └─→ collection.find().limit(100)
      └─→ Analyze field types
   
   d) Build Field Descriptions
      └─→ Map Python type → MongoDB type
      └─→ Mark indexed fields
   
   e) Store Schema
      └─→ {
            "description": "...",
            "fields": {...},
            "doc_count": N,
            "indexed_fields": [...],
            "avg_doc_size_bytes": X,
            "total_size_bytes": Y
          }

4. Return complete schema
```

### Embedding Generation Phase

```
1. For EACH collection in schema:
   
   a) Encode Collection
      └─→ text = collection_name + description
      └─→ vector = model.encode(text)  # 384 dims
   
   b) For EACH field in collection:
      └─→ text = field_name + field_description
      └─→ vector = model.encode(text)  # 384 dims
      └─→ Store in field_embeddings dict
   
   c) Store in embeddings_db
      └─→ {
            "collection_embedding": [...],  # 384 dims
            "field_embeddings": {...},
            "fields": {...},
            "doc_count": N,
            ...
          }

2. Save embeddings
   └─→ Pickle.dump → schema_embeddings.pkl
   └─→ JSON.dump → schema_info.json
```

### Query Processing Phase

```
1. User Question: "Show all products under $50"

2. Load Embeddings
   └─→ pickle.load(schema_embeddings.pkl)
   └─→ Extract embeddings_db dict

3. Initialize Search Engine
   └─→ SemanticSchemaSearch(embeddings_db)

4. Find Relevant Collections
   a) Encode question
      └─→ question_vector = model.encode(question)  # 384 dims
   
   b) Compare with ALL collections
      └─→ For products:
          └─→ similarity = cosine(question_vector, products_embedding)
          └─→ 0.92 ← High!
      └─→ For users:
          └─→ similarity = cosine(question_vector, users_embedding)
          └─→ 0.34 ← Low
      └─→ For orders:
          └─→ similarity = cosine(question_vector, orders_embedding)
          └─→ 0.41 ← Low
   
   c) Sort & Return Top-3
      └─→ [(products, 0.92), (orders, 0.41), (users, 0.34)]

5. For Each Selected Collection (products):
   a) Find Relevant Fields
      └─→ Compare question with field embeddings
      └─→ price: 0.89 ← Selected
      └─→ name: 0.72 ← Selected
      └─→ rating: 0.48
      └─→ _id: 0.22

6. Build Context
   └─→ Format collections + fields as readable text
   └─→ Include relevance scores
   └─→ Include indexed fields info

7. Build Full Prompt
   └─→ System instruction
   └─→ Schema context
   └─→ User question
   └─→ Examples

8. Return Prompt
   └─→ Ready to send to LLM
```

## Performance Characteristics

### Time Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| Extract schema | O(N) | N = num collections |
| Generate embeddings | O(N×M) | M = avg fields per collection |
| Load embeddings | O(1) | Constant time |
| Encode question | O(1) | Fixed 384-dim encoding |
| Calculate similarities | O(N) | N = num collections |
| Build context | O(K) | K = top-k collections |
| **Total per query** | **O(N)** | Very fast! |

### Space Complexity

| Component | Size | Notes |
|-----------|------|-------|
| One embedding | 384 floats | ~1.5 KB |
| Schema embeddings | N×M × 1.5 KB | N = collections, M = fields |
| Typical file | 1-5 MB | For 50 collections |
| In-memory | 10-30 MB | Fully loaded in RAM |

### Throughput

- **Embeddings per second:** 1,000-2,000 (during generation)
- **Queries per second:** 20-50 (with loaded embeddings)
- **Batch processing:** 100+ queries/second (same embeddings)

## MongoDB Collection Statistics Format

```python
stats = {
    "ns": "ai-test-db.users",           # Namespace
    "count": 5000,                      # Document count
    "size": 2560000,                    # Data size bytes
    "avgObjSize": 512,                  # Average doc size
    "storageSize": 2560000,             # Storage allocated
    "nindexes": 3,                      # Number of indexes
    "totalIndexSize": 512000,           # All indexes size
    "indexSizes": {                     # Per-index sizes
        "_id_": 256000,
        "email_1": 256000
    }
}
```

## Index Information Extracted

MongoDB tracks which fields are indexed. Indexed fields are:
- Faster to search
- Better for filtering
- Should be highlighted in schema
- Important for query optimization

```python
indexes = list(collection.list_indexes())
# Returns:
# [
#   {"key": [("_id", 1)]},
#   {"key": [("email", 1)], "unique": True},
#   {"key": [("created_at", -1)]},
# ]

indexed_fields = {"_id", "email", "created_at"}
```

## Token/Cost Reduction

### Without Embeddings (Naive Approach)
```
MongoDB database: 50 collections, 500 total fields
   ↓
For each query, send ALL schema to LLM
   ↓
Tokens per query: 5,000-10,000
Cost per query: $0.15-$0.30 (with GPT-4)
```

### With Embeddings (This System)
```
MongoDB database: 50 collections, 500 total fields
   ↓
Generate embeddings once (5-10 seconds)
   ↓
For each query, find only 3-5 relevant collections
   ↓
Tokens per query: 500-1,000
Cost per query: $0.015-$0.030 (with GPT-4)
   ↓
Savings: 80-90% cost reduction! ✓
```

## System Reliability

### Strengths
✓ Works offline (no external APIs)
✓ Deterministic (same question = same output)
✓ Fast (embeddings pre-computed)
✓ Scalable (handles 100+ collections)
✓ No hallucinations (only actual schema)

### Limitations
✗ Needs 5-10s initial setup
✗ Must regenerate when schema changes
✗ Quality depends on table/field descriptions
✗ Works best in English

### Mitigation Strategies
- Use scheduled jobs to regenerate embeddings nightly
- Write clear, specific collection/field descriptions
- Use higher-quality models if needed (all-mpnet-base-v2)
- Support multiple languages with multilingual models

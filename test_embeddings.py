from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['ai_test_db']

print("Collections in database:", db.list_collection_names())
print("\nProducts count:", db['products'].count_documents({}))

# Sample a product
sample_product = db['products'].find_one()
print("\nSample product:", sample_product)

# Test embeddings
print("\n--- Testing Vector Search ---")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create simple embeddings
collections = db.list_collection_names()
embeddings = {}

for coll in collections:
    if coll.startswith('system.'):
        continue
    count = db[coll].count_documents({})
    text = f"{coll}: Collection with {count} documents"
    embedding = model.encode(text)
    embeddings[coll] = embedding
    print(f"Created embedding for {coll}")

# Test query
question = "Find all products with a price less than 500"
question_embedding = model.encode(question)

print(f"\n--- Testing query: '{question}' ---")
similarities = {}
for coll, emb in embeddings.items():
    similarity = np.dot(question_embedding, emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(emb))
    similarities[coll] = similarity
    print(f"{coll}: {similarity:.4f}")

top_match = max(similarities.items(), key=lambda x: x[1])
print(f"\nTop match: {top_match[0]} with similarity {top_match[1]:.4f}")
print(f"Threshold check (0.2): {'PASS' if top_match[1] >= 0.2 else 'FAIL'}")

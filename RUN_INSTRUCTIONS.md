# How to Run the Simplified Chat Flow

## Prerequisites

### 1. MongoDB Running
```bash
# Check if MongoDB is running
mongosh --eval "db.version()"

# If not running, start it:
mongod
```

### 2. Ollama with db-assistant Model
```bash
# Check if Ollama is running
ollama list

# If not running, start Ollama:
ollama serve

# Make sure db-assistant model exists
ollama list | findstr db-assistant

# If model doesn't exist, you need to create it first
# (See model creation instructions below)
```

### 3. Python Dependencies
```bash
# Install required packages
pip install pymongo requests
```

## Running the Chat

### Option 1: Interactive Chat Mode
```bash
cd database-assistance
python simple_chat_flow.py
```

Then type your questions:
```
You: Show me all customers
You: What is the total revenue?
You: List products with price greater than 100
You: exit
```

### Option 2: Test Mode (Automated)
```bash
cd database-assistance
python test_simple_flow.py
```

This will run predefined test questions automatically.

### Option 3: Use in Your Code
```python
from simple_chat_flow import SimpleChatFlow

# Initialize
chat = SimpleChatFlow(
    mongodb_uri='mongodb://localhost:27017',
    db_name='ai_test_db'
)

# Ask a question
response = chat.ask("Show me all orders from 2024")

# Check results
if response['success']:
    print(f"Found {len(response['results'])} results")
    for doc in response['results']:
        print(doc)
else:
    print(f"Error: {response['error']}")
```

## Creating the db-assistant Model (If Needed)

If you don't have the `db-assistant` model yet, create it:

### Step 1: Create Modelfile
```bash
# Create a file named 'Modelfile' with this content:
FROM qwen2.5:3b

SYSTEM """You are a MongoDB query expert. You have memorized this database schema:

Collection 'products': {name(string), price(number), categoryId(unknown), stock(number), createdAt(date)}
Collection 'reviews': {productId(unknown), rating(number), comment(string), createdAt(date)}
Collection 'payments': {orderId(unknown), amount(number), method(string), status(string), paymentDate(date)}
Collection 'users': {name(string), email(string), role(string), createdAt(date), categoryId(unknown)}
Collection 'orders': {customerId(unknown), productId(unknown), quantity(number), amount(number), status(string), orderDate(date)}
Collection 'categories': {name(string), createdAt(date)}
Collection 'customers': {name(string), email(string), phone(string), city(string), createdAt(date)}

Generate ONLY valid MongoDB queries. Use format: db.collection.find({}) or db.collection.aggregate([])
Return ONLY the query code, no explanation."""

PARAMETER temperature 0.1
PARAMETER num_predict 300
```

### Step 2: Create the Model
```bash
ollama create db-assistant -f Modelfile
```

### Step 3: Test the Model
```bash
ollama run db-assistant "Show me all customers"
```

## Troubleshooting

### Error: "Cannot connect to MongoDB"
```bash
# Start MongoDB
mongod

# Or check if it's running on a different port
mongosh --port 27017
```

### Error: "LLM API returned 404"
```bash
# Make sure Ollama is running
ollama serve

# Check available models
ollama list

# Pull base model if needed
ollama pull qwen2.5:3b
```

### Error: "Collection not found"
```bash
# Check your database name and collections
mongosh
use ai_test_db
show collections
```

### Error: "database_schema.md not found"
The script will auto-extract schema from the database if the file is missing.
But it's better to have the file in the same directory as the script.

## Example Session

```
MongoDB Chat Assistant - Simplified Flow
============================================================
Type your question or 'exit' to quit

You: Show me total sales

============================================================
Question: Show me total sales
============================================================

[Attempt 1/3]
→ Sending to LLM...
→ LLM returned:
db.orders.aggregate([{"$group": {"_id": null, "total": {"$sum": "$amount"}}}])

→ Validating syntax...
✓ Syntax valid
→ Executing query...
✓ Query executed successfully
✓ Found 1 results

✓ Found 1 result(s):

1. total: 15000

[Query used: db.orders.aggregate([{"$group": {"_id": null, "total": {"$sum": "$amount"}}}])...]

You: exit
Goodbye!
```

## Configuration

Edit `simple_chat_flow.py` to change settings:

```python
chat = SimpleChatFlow(
    mongodb_uri='mongodb://localhost:27017',  # Your MongoDB URI
    db_name='ai_test_db'                      # Your database name
)
```

Change LLM settings:
```python
self.model_name = "db-assistant"  # Your model name
self.llm_url = "http://localhost:11434/api/generate"  # Ollama API URL
```

Change retry attempts:
```python
response = chat.ask("your question", max_retries=5)  # Default is 3
```

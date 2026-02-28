# Quick Start Guide

## 🚀 Fast Setup (3 Steps)

### 1. Run Setup
```bash
setup.bat
```
This installs dependencies and creates the db-assistant model.

### 2. Start MongoDB (if not running)
```bash
mongod
```

### 3. Start Chat
```bash
start_chat.bat
```

That's it! Now you can ask questions in natural language.

---

## 📝 Example Questions

```
You: Show me all customers
You: What is the total revenue from orders?
You: List products with stock less than 10
You: Show orders from January 2024
You: Count how many customers are in each city
You: What are the top 5 products by price?
```

---

## 🔧 Manual Setup (if batch files don't work)

### Install Dependencies
```bash
pip install pymongo requests
```

### Create Model
```bash
# Pull base model
ollama pull qwen2.5:3b

# Create Modelfile (copy content from setup.bat)
# Then run:
ollama create db-assistant -f Modelfile
```

### Run Chat
```bash
cd database-assistance
python simple_chat_flow.py
```

---

## 🎯 How It Works

```
Your Question
    ↓
LLM reads schema from memory
    ↓
Generates MongoDB query
    ↓
Python validates syntax
    ↓
If error → Regenerate (max 3 tries)
    ↓
Execute query
    ↓
Show results
```

---

## ⚙️ Configuration

Edit `simple_chat_flow.py`:

```python
# Line 15-16: Change database
mongodb_uri='mongodb://localhost:27017'
db_name='ai_test_db'

# Line 19-20: Change LLM
llm_url = "http://localhost:11434/api/generate"
model_name = "db-assistant"
```

---

## 🐛 Troubleshooting

### MongoDB not connecting?
```bash
# Check if running
mongosh

# Start if needed
mongod
```

### Ollama not responding?
```bash
# Check if running
ollama list

# Start if needed
ollama serve
```

### Model not found?
```bash
# List models
ollama list

# Create model
ollama create db-assistant -f Modelfile
```

### Schema not loading?
Make sure `database_schema.md` is in the same folder as `simple_chat_flow.py`

---

## 📊 Response Format

Success:
```python
{
    'success': True,
    'results': [...],  # List of documents
    'query': 'db.orders.find(...)',
    'error': None
}
```

Error:
```python
{
    'success': False,
    'results': None,
    'query': 'db.orders.find(...)',
    'error': 'Syntax validation failed: ...'
}
```

---

## 🔄 Retry Logic

The system automatically retries if:
- JSON syntax is invalid
- Collection name is wrong
- Query execution fails

It will try up to 3 times with different approaches before giving up.

---

## 💡 Tips

1. **Be specific**: "Show orders from 2024" is better than "Show orders"
2. **Use natural language**: The LLM understands context
3. **Check the query**: The system shows which query was used
4. **Schema matters**: Make sure `database_schema.md` is up to date

---

## 📁 Files

- `simple_chat_flow.py` - Main chat system
- `test_simple_flow.py` - Automated tests
- `database_schema.md` - Schema definition
- `setup.bat` - One-click setup
- `start_chat.bat` - One-click start
- `RUN_INSTRUCTIONS.md` - Detailed instructions

---

## 🎓 Advanced Usage

### Use in Your Code
```python
from simple_chat_flow import SimpleChatFlow

chat = SimpleChatFlow()
response = chat.ask("Show me all customers")

if response['success']:
    for doc in response['results']:
        print(doc)
```

### Change Retry Count
```python
response = chat.ask("your question", max_retries=5)
```

### Get Raw Results
```python
response = chat.ask("your question")
results = response['results']  # List of dicts
query = response['query']      # Query that was used
```

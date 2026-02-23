import json
import os
import subprocess
from pymongo import MongoClient
from datetime import datetime
from metadata_provider import infer_type

def extract_full_metadata():
    """Extract schema from all collections and return as a structured dictionary"""
    client = MongoClient("mongodb://localhost:27017")
    db = client["ai_test_db"]
    
    metadata = {}
    for collection_name in db.list_collection_names():
        # Clean up: Ignore system, backup, and chat history for the core "brain"
        if any(x in collection_name.lower() for x in ['system.', 'backup', 'chat_history']):
            continue
            
        collection = db[collection_name]
        field_info = {}
        
        # Sample document for schema
        doc = collection.find_one()
        if doc:
            for field, value in doc.items():
                if field == "_id": continue
                field_info[field] = infer_type(value)
        
        metadata[collection_name] = field_info
    return metadata

def generate_schema_markdown(metadata):
    """Compact string for the LLM"""
    md = ""
    for coll, fields in metadata.items():
        field_list = [f"{f}({t})" for f, t in fields.items()]
        md += f"Collection '{coll}': {{{', '.join(field_list)}}}\n"
    return md.strip()

def create_ollama_model(schema_md):
    """Create a custom Ollama model with the schema baked into the system prompt"""
    model_name = "db-assistant"
    
    system_prompt = f"""You are an expert MongoDB Query Generator for a local-first analytics application.
You have PERMANENT access to the database schema provided below. 

DATABASE SCHEMA:
{schema_md}

CORE LOGIC RULES (MANDATORY):
1. STARTING COLLECTION:
   - For queries about 'sales', 'revenue', 'income', 'money', or 'status': ALWAYS START WITH `db.orders`.
   - For queries about 'names of products' or 'inventory': START WITH `db.products` and lookup orders.
   - For queries about 'customers': START WITH `db.customers` and lookup orders.

2. FILTERING:
   - Sales/Revenue queries MUST include: {{"$match": {{"status": "completed"}}}}
   - Date ranges MUST be applied to `orderDate` using ISODate.

3. CROSS-TABLE RULES:
   - If you need a product name while looking at orders: Use $lookup with from: "products", localField: "productId", foreignField: "_id".
   - If you need a customer name while looking at orders: Use $lookup with from: "customers", localField: "customerId", foreignField: "_id".

4. SYNTAX RULES (CRITICAL):
   - Use standard JSON only: ALWAYS quote both keys and values with double quotes (").
   - NO JAVASCRIPT: NEVER use `new Date()`, `ISODate()`, or `ObjectId()`.
   - Use plain strings for dates: e.g., "2024-01-01" instead of ISODate("2024-01-01").
   - Ensure all MongoDB operators (like $match, $group) have the $ prefix and are quoted.
   - Return the entire command on a SINGLE LINE.

5. OUTPUT FORMAT (STRICT):
   - You MUST return a full MongoDB Shell command.
   - Example: db.orders.aggregate([ {{ "$match": {{ "status": "completed" }} }} ])
   - NEVER return just the JSON array [].
   - NO EXPLANATIONS. NO MARKDOWN. NO CODE BLOCKS.
   - Return [] ONLY if the query is impossible.

6. USER INTENT & VISIBILITY (MANDATORY):
   - READ CAREFULLY: If the user asks for a specific field (like 'name', 'month', or 'quantity'), it MUST be in the final result.
   - USE $PROJECT: Always use a final `$project` stage to ensure the output is clean and contains the specific fields requested.
   - MONTH NAMES: If the user asks for "month wise", use the $switch or similar logic in $project to convert month numbers (1-12) to names (January-December) if possible, or at least label the field "Month".
   - CALCULATED FIELDS: If you calculate a sum, label it clearly (e.g., "Total Revenue", "Count", "QuantitySold").

7. EXAMPLE TEMPLATES:
   - Monthly Sales: db.orders.aggregate([ {{ "$match": {{ "status": "completed" }} }}, {{ "$group": {{ "_id": {{ "$month": "$orderDate" }}, "Revenue": {{ "$sum": "$amount" }} }} }}, {{ "$project": {{ "_id": 0, "Month": "$_id", "Revenue": 1 }} }}, {{ "$sort": {{ "Month": 1 }} }} ])
   - Top Products: db.products.aggregate([ {{ "$lookup": {{ "from": "orders", "localField": "_id", "foreignField": "productId", "as": "sales" }} }}, {{ "$unwind": "$sales" }}, {{ "$group": {{ "_id": "$name", "QuantitySold": {{ "$sum": "$sales.quantity" }} }} }}, {{ "$project": {{ "_id": 0, "ProductName": "$_id", "QuantitySold": 1 }} }}, {{ "$sort": {{ "QuantitySold": -1 }} }}, {{ "$limit": 5 }} ])
"""

    modelfile_content = f"""
FROM qwen2.5:3b
PARAMETER temperature 0.0
PARAMETER stop "USER QUESTION:"
PARAMETER stop "Question:"
SYSTEM "{system_prompt.replace('"', '\\"').replace('\n', ' ')}"
"""
    
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"🚀 Creating/Updating Ollama model '{model_name}'...")
    try:
        # Run ollama create command
        result = subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully created model '{model_name}'")
        else:
            print(f"❌ Error creating model: {result.stderr}")
    except Exception as e:
        print(f"❌ Failed to run ollama command: {e}")

if __name__ == "__main__":
    print("🔍 Analyzing database schema...")
    metadata = extract_full_metadata()
    
    print("📝 Generating schema reference file...")
    schema_md = generate_schema_markdown(metadata)
    with open("database_schema.md", "w", encoding="utf-8") as f:
        f.write(schema_md)
    
    create_ollama_model(schema_md)
    print("✨ Sync complete! The AI now has persistent knowledge of your database.")

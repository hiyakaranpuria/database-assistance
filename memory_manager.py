from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["ai_test_db"]
history_col = db["chat_history"]

def save_message(role, content, session_id="default"):
    history_col.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

def get_chat_history(session_id="default", limit=5):
    """Retrieves the last few messages for context."""
    cursor = history_col.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
    messages = list(cursor)[::-1] # Reverse to get chronological order
    
    # Format for the LLM prompt
    history_text = ""
    for msg in messages:
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"
    return history_text
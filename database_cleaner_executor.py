
import streamlit as st
from pymongo import MongoClient, UpdateOne, DeleteOne
from llm_integration import local_llm
import traceback

class DatabaseCleaningExecutor:
    def __init__(self, connection_string="mongodb://localhost:27017", db_name="ai_test_db"):
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]

    def get_collections(self):
        """Get list of collections from schema metadata"""
        import json
        import os
        
        info_path = "d:\\ai-data_assistance\\database_info.json"
        
        try:
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    data = json.load(f)
                    return list(data.get("collections", {}).keys())
        except Exception:
            pass
            
        # Fallback if file not found
        return self.db.list_collection_names()

    def get_sample_data(self, collection_name, limit=3):
        """Get top N rows from collection"""
        collection = self.db[collection_name]
        return list(collection.find().limit(limit))

    def _truncate_data(self, data, max_chars=50):
        """Helper to truncate long strings in sample data to reduce token count"""
        truncated = []
        for item in data:
            new_item = {}
            for k, v in item.items():
                if isinstance(v, str) and len(v) > max_chars:
                    new_item[k] = v[:max_chars] + "..."
                elif isinstance(v, dict):
                    # Simple recursive truncation for nested dicts
                    new_item[k] = str(v)[:max_chars]
                elif isinstance(v, list):
                    new_item[k] = str(v)[:max_chars]
                else:
                    new_item[k] = v
            truncated.append(new_item)
        return truncated

    def generate_plan(self, collection_name):
        """Step 1: Get cleaning code from LLM"""
        if not local_llm:
            return {"error": "Local LLM is not initialized."}

        st.info(f"🤖 AI Agent is scanning '{collection_name}' for quality issues...")
        
        # 1. Fetch Sample Data
        samples = self.get_sample_data(collection_name)
        if not samples:
            return {"error": "Collection is empty."}
        
        # 2. Optimize payload
        optimized_samples = self._truncate_data(samples)

        # 3. Ask LLM for Code
        cleaning_code = local_llm.generate_cleaning_code(optimized_samples, collection_name)
        
        return {
            "samples": samples, # Return full samples for UI
            "code": cleaning_code
        }

    def execute_cleaning(self, cleaning_code, collection_name):
        """Step 2: Execute the generated pipeline safely"""
        import json
        import re
        
        try:
            # 1. Clean up JSON string (remove Python comments if LLM added them)
            # This regex removes # comments
            cleaned_json = re.sub(r'(?m)^ *#.*\n?', '', cleaning_code)
            
            # 2. Parse Pipeline
            pipeline = json.loads(cleaned_json)
            
            if not isinstance(pipeline, list):
                return {"success": False, "error": "AI generated invalid format (not a list)."}
            
            # 3. Execute updateMany with pipeline
            result = self.db[collection_name].update_many({}, pipeline)
            
            summary = {
                "matched_docs": result.matched_count,
                "modified_docs": result.modified_count,
                "method": "MongoDB Aggregation Pipeline"
            }
            
            return {"success": True, "summary": summary}
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON Pipeline: {str(e)}", "raw": cleaning_code}
        except Exception as e:
            return {"success": False, "error": f"Execution Error: {str(e)}\n{traceback.format_exc()}"}

# Global Instance
cleaning_executor = DatabaseCleaningExecutor()
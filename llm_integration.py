#!/usr/bin/env python3
"""
Local LLM Integration for AI Database Analytics
Supports Ollama, Hugging Face, and multilingual query generation
"""

import requests
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from metadata_provider import extract_metadata

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"$date": obj.isoformat() + "Z"}
        return super().default(obj)

class OllamaLLM:
    """Ollama local LLM integration for natural language to MongoDB query conversion"""
    
    def __init__(self, model_name="db-assistant", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.conversation_history = []
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                # Relaxed check: if specific model not found, don't crash, just warn
                if self.model_name not in model_names and f"{self.model_name}:latest" not in model_names:
                    print(f"⚠️  Model {self.model_name} not found in Ollama list. Available: {model_names}")
                    print(f"💡 Defaulting to available model if possible, or run: ollama pull {self.model_name}")
            else:
                raise Exception(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            # Don't raise, just print warning so app doesn't crash if Ollama isn't up
            print(f"⚠️  Cannot connect to Ollama. Make sure it's running: {e}")

    def generate_cleaning_code(self, sample_data: List[Dict], collection_name: str) -> str:
        """Generate MongoDB Aggregation Pipeline for updateMany"""
        
        prompt = f"""You are a MongoDB Expert. Write an Aggregation Pipeline (JSON) to update '{collection_name}'.
        
        SAMPLE DATA (Truncated):
        {json.dumps(sample_data, default=str, indent=2)}

        TASK: 
        Create a pipeline for `updateMany` to clean the data.
        Use these EXACT templates (replace 'field' with actual field name):
        1. STRINGS: {{"$set": {{ "field": {{ "$trim": {{ "input": "$field" }} }} }} }}
        2. EMAILS: {{"$set": {{ "email": {{ "$toLower": "$email" }} }} }}
        3. NULLS: {{"$set": {{ "field": {{ "$ifNull": ["$field", "Unknown"] }} }} }}
        
        OUTPUT FORMAT:
        Return ONLY a JSON list of stages.
        Example: [ {{ "$set": {{ "email": {{ "$toLower": "$email" }} }} }}, {{ "$set": {{ "name": {{ "$trim": {{ "input": "$name" }} }} }} }} ]
        
        RULES:
        - JSON ONLY. NO MARKDOWN.
        - Must be a valid list of pipeline stages.
        - DO NOT modify fields ending in 'Id', 'id', or '_id' (like orderId, _id, userId).
        - DO NOT apply $toLower or $trim to dates or numbers.
        """

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "max_tokens": 800,
                        "stop": ["User:", "Confidence Score:"]
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result["response"]
                
                # Robust extraction: Look for JSON list [...]
                code_match = re.search(r'(\[.*\])', raw_response, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1).strip()
                else:
                    code = raw_response.strip()

                return code
            else:
                return f"# Error generating query: {response.text}"
                
        except Exception as e:
            return f"# Error generating cleaning pipeline: {str(e)}"


    def generate_query(self, user_question: str, schema_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate MongoDB query from natural language with Dynamic Schema Linking"""
        
        # 0. Conversational Guardrail (Handle High/Hello)
        # Matches: "hi", "Hi!", "hello.", "hey there"
        if re.match(r'^(hi|hello|hey|greetings|good morning|good evening)([!. ]|$)', user_question.strip(), re.IGNORECASE):
            # Return a pipeline that creates a single text response
            return '[ { "$limit": 1 }, { "$project": { "_id": 0, "Agent Message": { "$literal": "👋 Hello! I am your AI Data Assistant. Ask me about your Sales, Products, or Customers." } } } ]'

        if schema_info is None:
            schema_info = extract_metadata()
            
        # --- HYBRID ROUTER LOGIC ---
        # 1. Identify relevant collections based on keywords (Fast Path)
        relevant_schema = {}
        question_lower = user_question.lower()
        
        # Synonyms map to detect intent even if table name isn't used
        synonyms = {
            "orders": ["sales", "revenue", "income", "money", "selling", "bought", "purchase", "transaction"],
            "products": ["items", "stock", "inventory", "goods", "sku", "available", "product"],
            "customers": ["users", "clients", "people", "buyers", "accounts", "profiles", "customer"],
            "reviews": ["feedback", "rating", "comments", "stars", "review"],
        }
        
        # Check for direct table matches or synonyms
        for coll_name in schema_info.keys():
            singular = coll_name.rstrip('s')
            
            # Check explicit table mentions
            if coll_name in question_lower or singular in question_lower:
                relevant_schema[coll_name] = schema_info[coll_name]
            
            # Check synonyms
            if coll_name in synonyms:
                 if any(syn in question_lower for syn in synonyms[coll_name]):
                     relevant_schema[coll_name] = schema_info[coll_name]
                
        # 2. Strict Intent Check (The "STOP" Rule)
        if not relevant_schema:
            # No database intent detected -> STOP
            print("🛑 No database keywords found. Rejecting query.")
            return '[ { "$limit": 1 }, { "$project": { "_id": 0, "System": { "$literal": "NOT_IN_DATABASE" } } } ]'

        # 3. Add Related Tables (Smart Context)
        final_schema = relevant_schema.copy()
        
        relationships = {
            "products": ["orders", "categories", "reviews"],
            "orders": ["customers", "products", "payments"],
            "customers": ["orders"],
            "reviews": ["products"],
            "payments": ["orders"]
        }
        
        for table in list(relevant_schema.keys()):
            if table in relationships:
                for related in relationships[table]:
                    if related in schema_info and related not in final_schema:
                        final_schema[related] = schema_info[related]
                        
        target_schema = final_schema
        print(f"⚡ Router Selected: {list(final_schema.keys())}")
            
        # Simplified prompt: No longer sending the massive schema metadata
        # The 'db-assistant' model already has this knowledge baked in.
        prompt = self._build_prompt(user_question)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent results
                        "top_p": 0.9,
                        "max_tokens": 1000,
                        "stop": ["USER QUESTION:", "Question:"]
                    }
                },
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                query = self._extract_query(result["response"])
                
                # Store in conversation history
                self.conversation_history.append({
                    "question": user_question,
                    "query": query,
                    "timestamp": datetime.now()
                })
                
                return query
            else:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("LLM request timed out. Try a simpler query or check Ollama performance.")
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._fallback_query(user_question)

    def synthesize_answer(self, question: str, data: Any) -> str:
        """Convert database results into natural language answer (Stage 3)"""
        # Truncate large data for prompt
        if isinstance(data, list) and len(data) > 20:
            data_sample = data[:20]
        else:
            data_sample = data
            
        prompt = f"""You are a data summarizer.

Rules:
- Use simple language
- Do NOT add information
- Do NOT guess
- Explain ONLY what the data shows

User Question: {question}

Data:
{json.dumps(data_sample, default=str, indent=2)}
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                return result["response"].strip()
        except:
            # Fallback message
            return "Analysis available (see table)."
        return "Analysis complete."
    
    def _build_prompt(self, question: str) -> str:
        """Build a simplified prompt for the 'db-assistant' model"""
        
        current_year = datetime.now().year
        
        # Add conversation context if available
        context = ""
        if self.conversation_history:
            recent_context = self.conversation_history[-2:]  # Last 2 interactions
            context = "\nRECENT CONVERSATION:\n"
            for item in recent_context:
                context += f"Q: {item['question']}\nA: {item['query']}\n"
        
        prompt = f"""Task: Convert the User Question into a MongoDB Aggregation Pipeline.
Current Year: {current_year}
{context}
USER QUESTION: "{question}"
MongoDB Aggregation Pipeline:"""
        
        return prompt
        
        return prompt
    
    def _format_schema(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for the prompt"""
        schema_text = ""
        for collection, info in schema_info.items():
            fields = info.get('fields', {})
            schema_text += f"\nCollection '{collection}':\n"
            # Schema optimization: Types includes field names, so we don't need a separate Fields list.
            
            # Add field type information
            field_details = []
            for field, field_type in fields.items():
                field_details.append(f"{field}({field_type})")
            schema_text += f"  Types: {', '.join(field_details)}\n"
        
        return schema_text
    
    def _extract_query(self, response: str) -> str:
        """Extract JSON query from LLM response"""
        # Clean the response
        response = response.strip()
        
        # Try to find JSON array in response
        json_patterns = [
            r'\[.*?\]',  # Simple array
            r'\[\s*\{.*?\}\s*\]',  # Array with objects
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Validate JSON
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, try to extract from code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                json.loads(code_block_match.group(1))
                return code_block_match.group(1)
            except json.JSONDecodeError:
                pass
        
        # Last resort: look for any array-like structure
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            return array_match.group(0)
        
        # Fallback to simple query
        print(f"⚠️  Could not extract valid JSON from LLM response: {response[:200]}...")
        return self._fallback_query("")
    
    def _fallback_query(self, question: str) -> str:
        """Generate fallback query when LLM fails"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['total', 'sales', 'revenue']):
            return json.dumps([
                {"$match": {"status": "completed"}},
                {"$group": {"_id": None, "total": {"$sum": "$amount"}}},
                {"$project": {"_id": 0, "total": 1}}
            ])
        elif any(word in question_lower for word in ['customer', 'customers']):
            return json.dumps([
                {"$lookup": {"from": "customers", "localField": "customerId", "foreignField": "_id", "as": "customer"}},
                {"$unwind": "$customer"},
                {"$project": {"customerName": "$customer.name", "amount": 1}},
                {"$limit": 10}
            ])
        else:
            return json.dumps([{"$limit": 10}])

class MultilingualLLM:
    """Multilingual support for database queries"""
    
    def __init__(self):
        # Using db-assistant for all languages
        self.default_model = OllamaLLM("db-assistant")
        self.language_models = {
            'en': self.default_model,
            'es': self.default_model,
            'fr': self.default_model,
            'de': self.default_model,
            'it': self.default_model,
            'pt': self.default_model,
            'zh': self.default_model,
            'ja': self.default_model,
            'ko': self.default_model,
        }
        
        # Try to import language detection
        try:
            from langdetect import detect
            self.detect_language = detect
        except ImportError:
            print("💡 Install langdetect for better language detection: pip install langdetect")
            self.detect_language = self._simple_language_detection
    
    def _simple_language_detection(self, text: str) -> str:
        """Simple language detection without external libraries"""
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'  # Chinese
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'  # Japanese
        elif re.search(r'[\uac00-\ud7af]', text):
            return 'ko'  # Korean
        else:
            return 'en'  # Default to English
    
    def generate_query(self, user_question: str, schema_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate query with automatic language detection"""
        try:
            language = self.detect_language(user_question)
        except:
            language = 'en'
        
        # Use appropriate model for detected language
        model = self.language_models.get(language, self.language_models['en'])
        
        try:
            return model.generate_query(user_question, schema_info)
        except Exception as e:
            print(f"Multilingual LLM failed for {language}, trying English model: {e}")
            return self.language_models['en'].generate_query(user_question, schema_info)

# Global instances
try:
    # Try to initialize Ollama LLM
    local_llm = OllamaLLM()
    multilingual_llm = MultilingualLLM()
    LLM_AVAILABLE = True
    print("✅ Local LLM initialized successfully")
except Exception as e:
    print(f"⚠️  Local LLM not available: {e}")
    print("💡 To enable local LLM:")
    print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   2. Start Ollama: ollama serve")
    print("   3. Download model: ollama pull qwen2.5:3b")
    local_llm = None
    multilingual_llm = None
    LLM_AVAILABLE = False

def generate_llm_query(question: str, use_multilingual: bool = True) -> str:
    """Main function to generate queries using local LLM"""
    if not LLM_AVAILABLE:
        # Fallback to existing system
        from dynamic_query_generator import generate_mongo_query
        return generate_mongo_query(question)
    
    try:
        if use_multilingual and multilingual_llm:
            return multilingual_llm.generate_query(question)
        elif local_llm:
            return local_llm.generate_query(question)
        else:
            raise Exception("No LLM available")
    except Exception as e:
        print(f"LLM query generation failed: {e}")
        # Fallback to existing system
        from dynamic_query_generator import generate_mongo_query
        return generate_mongo_query(question)

def test_llm_integration():
    """Test the LLM integration with sample queries"""
    if not LLM_AVAILABLE:
        print("❌ LLM not available for testing")
        return
    
    test_questions = [
        "Show total sales for 2024",
        "Top 10 customers by spending", 
        "Which products sold the least?",
        "Monthly sales trends this year",
        "Customers who haven't ordered recently",
        # Multilingual examples
        "Muestra las ventas totales de este año",  # Spanish
        "显示今年的总销售额",  # Chinese
        "今年の総売上を表示",  # Japanese
    ]
    
    print("🤖 Testing Local LLM Integration")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            query = generate_llm_query(question)
            print(f"✅ Generated Query: {query[:100]}...")
            
            # Validate JSON
            json.loads(query)
            print("✅ Valid JSON")
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON generated")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_llm_integration()
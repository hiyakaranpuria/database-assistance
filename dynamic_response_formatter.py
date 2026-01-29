from metadata_provider import extract_metadata

class DynamicResponseFormatter:
    def __init__(self):
        self.metadata = extract_metadata()
    
    def format_response(self, question, raw_data, collection_hint=None):
        """Format response based on data structure and question context"""
        question_lower = question.lower()
        
        if isinstance(raw_data, str) and "Error" in raw_data:
            return f"Sorry, there was an issue: {raw_data}"
        
        if not isinstance(raw_data, list) or len(raw_data) == 0:
            return "No data found for your query."
        
        # Determine response type based on data structure
        first_item = raw_data[0]
        
        # Handle count results
        if isinstance(first_item, dict) and ('total' in first_item or 'count' in str(first_item).lower()):
            count_value = first_item.get('total', first_item.get('totalOrders', 0))
            entity = self._guess_entity_from_question(question)
            return f"There are **{count_value:,}** {entity} in the database."
        
        # Handle aggregation results (sum, avg, min, max)
        elif isinstance(first_item, dict) and len(first_item) == 2 and '_id' in first_item:
            for key, value in first_item.items():
                if key != '_id' and isinstance(value, (int, float)):
                    if 'total' in key.lower() or 'sum' in key.lower():
                        return f"The total amount is **${value:,.2f}**"
                    elif 'avg' in key.lower() or 'average' in key.lower():
                        return f"The average value is **${value:.2f}**"
                    elif 'max' in key.lower():
                        return f"The maximum value is **${value:,.2f}**"
                    elif 'min' in key.lower():
                        return f"The minimum value is **${value:,.2f}**"
        
        # Handle list results - format based on detected fields
        else:
            return self._format_list_data(raw_data, question)
    
    def _guess_entity_from_question(self, question):
        """Guess what entity the user is asking about"""
        question_lower = question.lower()
        
        if 'customer' in question_lower:
            return 'customers'
        elif 'order' in question_lower:
            return 'orders'
        elif 'product' in question_lower:
            return 'products'
        elif 'user' in question_lower:
            return 'users'
        elif 'payment' in question_lower:
            return 'payments'
        elif 'review' in question_lower:
            return 'reviews'
        else:
            return 'records'
    
    def _format_list_data(self, data, question):
        """Format list data based on available fields"""
        if not data:
            return "No data found."
        
        sample_item = data[0]
        if not isinstance(sample_item, dict):
            return f"Here's what I found: {data}"
        
        # Detect data type based on fields
        fields = list(sample_item.keys())
        
        response = "Here's what I found:\n\n"
        
        for i, item in enumerate(data[:10], 1):  # Limit to 10 items
            line = f"{i}. "
            
            # Format based on detected fields
            if 'name' in item:
                line += f"**{item['name']}**"
                if 'email' in item:
                    line += f" - {item['email']}"
                if 'city' in item:
                    line += f" ({item['city']})"
                if 'price' in item:
                    line += f" - ${item['price']}"
                if 'stock' in item:
                    line += f" (Stock: {item['stock']})"
                    
            elif 'amount' in item:
                line += f"Amount: ${item['amount']}"
                if 'status' in item:
                    line += f" - Status: {item['status']}"
                if 'orderDate' in item:
                    line += f" - Date: {str(item['orderDate'])[:10]}"
                    
            elif 'rating' in item:
                line += f"Rating: {item['rating']}/5"
                if 'comment' in item:
                    line += f" - \"{item['comment'][:50]}...\""
                    
            else:
                # Generic formatting for unknown structure
                important_fields = [k for k in fields if k != '_id'][:3]
                values = [f"{k}: {item.get(k, 'N/A')}" for k in important_fields]
                line += " | ".join(values)
            
            response += line + "\n"
        
        if len(data) > 10:
            response += f"\n... and {len(data) - 10} more records"
            
        return response

# Global instance
formatter = DynamicResponseFormatter()

def format_final_answer(question, raw_data):
    """Dynamic response formatting that adapts to any data structure"""
    return formatter.format_response(question, raw_data)
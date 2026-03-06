import os
from dataclasses import dataclass
from typing import Dict, List, Optional

# Centralized env vars
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_MONGO_DB = os.getenv("MONGO_DB", "ai_test_db")

@dataclass
class DatabaseConfig:
    """Configuration for different database types and domains"""
    connection_string: str
    database_name: str
    domain: str  # e.g., 'restaurant', 'ecommerce', 'healthcare', 'finance'
    
    # Field mappings for common concepts
    amount_fields: List[str]  # Fields that represent money/amounts
    date_fields: List[str]    # Fields that represent dates
    status_fields: List[str]  # Fields that represent status
    name_fields: List[str]    # Fields that represent names/titles
    
    # Collection mappings
    primary_transaction_collection: str  # Main transactional data
    customer_collection: Optional[str] = None
    product_collection: Optional[str] = None
    
    @classmethod
    def database_config(cls):
        return cls(
            connection_string=_MONGO_URI,
            database_name=_MONGO_DB,
            domain="database",
            amount_fields=["amount", "total", "price", "cost"],
            date_fields=["orderDate", "createdAt", "paymentDate"],
            status_fields=["status"],
            name_fields=["name", "title"],
            primary_transaction_collection="orders",
            customer_collection="customers",
            product_collection="products"
        )
    
    @classmethod
    def ecommerce_config(cls):
        return cls(
            connection_string=_MONGO_URI,
            database_name=os.getenv("MONGO_DB", "ecommerce_db"),
            domain="ecommerce",
            amount_fields=["price", "total", "subtotal", "amount"],
            date_fields=["created_at", "updated_at", "order_date"],
            status_fields=["status", "order_status", "payment_status"],
            name_fields=["name", "title", "product_name"],
            primary_transaction_collection="orders",
            customer_collection="users",
            product_collection="products"
        )
    
    @classmethod
    def finance_config(cls):
        return cls(
            connection_string=_MONGO_URI,
            database_name=os.getenv("MONGO_DB", "finance_db"),
            domain="finance",
            amount_fields=["amount", "balance", "transaction_amount", "value"],
            date_fields=["transaction_date", "created_at", "settlement_date"],
            status_fields=["status", "transaction_status"],
            name_fields=["account_name", "description", "reference"],
            primary_transaction_collection="transactions",
            customer_collection="accounts",
            product_collection="products"
        )
    
    @classmethod
    def from_env(cls):
        """Load configuration from environment variables"""
        domain = os.getenv("DB_DOMAIN", "database")
        
        if domain == "database":
            return cls.database_config()
        elif domain == "restaurant":
            return cls.database_config()  # Use database config for restaurant too
        elif domain == "ecommerce":
            return cls.ecommerce_config()
        elif domain == "finance":
            return cls.finance_config()
        else:
            # Generic configuration
            return cls(
                connection_string=os.getenv("MONGO_CONNECTION", _MONGO_URI),
                database_name=os.getenv("DB_NAME", _MONGO_DB),
                domain=domain,
                amount_fields=["amount", "price", "total", "value"],
                date_fields=["date", "created_at", "updated_at"],
                status_fields=["status"],
                name_fields=["name", "title"],
                primary_transaction_collection=os.getenv("PRIMARY_COLLECTION", "transactions")
            )

# Global configuration
config = DatabaseConfig.from_env()
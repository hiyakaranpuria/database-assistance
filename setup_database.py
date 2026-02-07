#!/usr/bin/env python3
"""
Setup script to configure the AI Data Assistant for different database types
"""

import os
import sys
from database_config import DatabaseConfig

def setup_restaurant_demo():
    """Setup for restaurant/food service business"""
    print("🍽️  Setting up Restaurant Demo...")
    
    # Set environment variables
    os.environ["DB_DOMAIN"] = "restaurant"
    os.environ["DB_NAME"] = "ai_test_db"
    os.environ["PRIMARY_COLLECTION"] = "orders"
    
    # Create .env file
    with open('.env', 'w') as f:
        f.write("DB_DOMAIN=restaurant\n")
        f.write("DB_NAME=ai_test_db\n")
        f.write("PRIMARY_COLLECTION=orders\n")
    

    
    print("✅ Restaurant configuration saved!")
    print("Run: python seed_mongo.py && streamlit run app_dynamic.py")

def setup_ecommerce_demo():
    """Setup for e-commerce business"""
    print("🛒 Setting up E-commerce Demo...")
    
    with open('.env', 'w') as f:
        f.write("DB_DOMAIN=ecommerce\n")
        f.write("DB_NAME=ecommerce_db\n")
        f.write("PRIMARY_COLLECTION=orders\n")
    

    print("✅ E-commerce configuration saved!")
    print("You'll need to create seed data for e-commerce schema")

def setup_finance_demo():
    """Setup for financial services"""
    print("💰 Setting up Finance Demo...")
    
    with open('.env', 'w') as f:
        f.write("DB_DOMAIN=finance\n")
        f.write("DB_NAME=finance_db\n")
        f.write("PRIMARY_COLLECTION=transactions\n")

    
    print("✅ Finance configuration saved!")
    print("You'll need to create seed data for finance schema")

def setup_custom():
    """Setup custom configuration"""
    print("⚙️  Custom Setup...")
    
    domain = input("Enter your domain (e.g., healthcare, logistics, retail): ")
    db_name = input("Enter database name: ")
    primary_collection = input("Enter primary collection name: ")
    connection = input("Enter MongoDB connection string (or press Enter for localhost): ")
    
    if not connection:
        connection = "mongodb://localhost:27017"
    
    with open('.env', 'w') as f:
        f.write(f"DB_DOMAIN={domain}\n")
        f.write(f"DB_NAME={db_name}\n")
        f.write(f"PRIMARY_COLLECTION={primary_collection}\n")
        f.write(f"MONGO_CONNECTION={connection}\n")
    

    
    print("✅ Custom configuration saved!")
    print("Make sure your database has data and run: streamlit run app_dynamic.py")

def main():
    print("🚀 AI Data Assistant Setup")
    print("=" * 40)
    print("1. Restaurant/Food Service")
    print("2. E-commerce")
    print("3. Finance/Banking")
    print("4. Custom Setup")
    print("5. Exit")
    
    choice = input("\nSelect setup type (1-5): ")
    
    if choice == "1":
        setup_restaurant_demo()
    elif choice == "2":
        setup_ecommerce_demo()
    elif choice == "3":
        setup_finance_demo()
    elif choice == "4":
        setup_custom()
    elif choice == "5":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Database Import Script
Imports the exported database backup
"""

import os
import subprocess
import json
import tarfile
from pymongo import MongoClient

def import_database():
    print("Starting database import...")
    
    # Extract backup if compressed file exists
    if os.path.exists("database_backup.tar.gz"):
        print("Extracting database backup...")
        with tarfile.open("database_backup.tar.gz", "r:gz") as tar:
            tar.extractall(".")
        print("Backup extracted!")
    
    # Try mongorestore first
    if os.path.exists("database_backup"):
        try:
            print("Importing using mongorestore...")
            cmd = ["mongorestore", "database_backup/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Database imported successfully using mongorestore!")
                return True
            else:
                print(f"Mongorestore failed: {result.stderr}")
                
        except FileNotFoundError:
            print("mongorestore not found. Trying Python import...")
        except Exception as e:
            print(f"Error with mongorestore: {e}")
    
    # Fallback to Python import
    return import_using_python()

def import_using_python():
    print("Importing using Python...")
    
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["ai_test_db"]
        
        backup_dir = "database_backup/ai_test_db"
        if not os.path.exists(backup_dir):
            print(f"Backup directory not found: {backup_dir}")
            return False
        
        for filename in os.listdir(backup_dir):
            if filename.endswith('.json'):
                collection_name = filename.replace('.json', '')
                print(f"Importing {collection_name}...")
                
                with open(os.path.join(backup_dir, filename), 'r') as f:
                    documents = json.load(f)
                
                if documents:
                    # Clear existing collection
                    db[collection_name].drop()
                    # Insert documents
                    db[collection_name].insert_many(documents)
                    print(f"   Imported {len(documents):,} documents")
        
        print("Database imported successfully using Python!")
        return True
        
    except Exception as e:
        print(f"Error during Python import: {e}")
        return False

if __name__ == "__main__":
    success = import_database()
    if success:
        print("\nDatabase import completed!")
        print("You can now run: streamlit run app_dynamic.py")
    else:
        print("\nDatabase import failed!")
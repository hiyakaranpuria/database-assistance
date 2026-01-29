#!/usr/bin/env python3
"""
Database Export Utility
Exports the ai_test_db database to a file for GitHub sharing
"""

import os
import subprocess
import json
from pymongo import MongoClient
from datetime import datetime
import tarfile
import shutil

class DatabaseExporter:
    def __init__(self, db_name="ai_test_db", connection_string="mongodb://localhost:27017"):
        self.db_name = db_name
        self.connection_string = connection_string
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        
    def check_database_exists(self):
        """Check if database exists and has data"""
        try:
            collections = self.db.list_collection_names()
            if not collections:
                print("❌ Database is empty or doesn't exist!")
                return False
            
            total_docs = 0
            for collection in collections:
                count = self.db[collection].count_documents({})
                total_docs += count
                print(f"📊 {collection}: {count:,} documents")
            
            print(f"📈 Total documents: {total_docs:,}")
            return total_docs > 0
            
        except Exception as e:
            print(f"❌ Error checking database: {e}")
            return False
    
    def export_using_mongodump(self):
        """Export database using mongodump command"""
        print("🚀 Exporting database using mongodump...")
        
        # Create backup directory
        backup_dir = "./database_backup"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        
        try:
            # Run mongodump command
            cmd = ["mongodump", "--db", self.db_name, "--out", backup_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Database exported successfully using mongodump!")
                return True
            else:
                print(f"❌ Mongodump failed: {result.stderr}")
                return False
                
        except FileNotFoundError:
            print("⚠️  mongodump not found. Trying Python export method...")
            return False
        except Exception as e:
            print(f"❌ Error running mongodump: {e}")
            return False
    
    def export_using_python(self):
        """Export database using Python (fallback method)"""
        print("🐍 Exporting database using Python...")
        
        backup_dir = f"./database_backup/{self.db_name}"
        os.makedirs(backup_dir, exist_ok=True)
        
        try:
            collections = self.db.list_collection_names()
            
            for collection_name in collections:
                print(f"📦 Exporting {collection_name}...")
                
                collection = self.db[collection_name]
                documents = list(collection.find())
                
                # Save as JSON file
                json_file = os.path.join(backup_dir, f"{collection_name}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(documents, f, default=str, indent=2)
                
                print(f"   ✅ Saved {len(documents):,} documents to {collection_name}.json")
            
            print("✅ Database exported successfully using Python!")
            return True
            
        except Exception as e:
            print(f"❌ Error during Python export: {e}")
            return False
    
    def create_compressed_backup(self):
        """Create compressed tar.gz file"""
        print("📦 Creating compressed backup...")
        
        backup_file = "database_backup.tar.gz"
        
        try:
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add("database_backup", arcname="database_backup")
            
            # Get file size
            file_size = os.path.getsize(backup_file)
            size_mb = file_size / (1024 * 1024)
            
            print(f"✅ Compressed backup created: {backup_file}")
            print(f"📏 File size: {size_mb:.1f} MB")
            
            if size_mb > 100:
                print("⚠️  Warning: File is larger than 100MB (GitHub limit)")
                print("   Consider reducing sample data size")
            else:
                print("✅ File size is good for GitHub!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating compressed backup: {e}")
            return False
    
    def create_import_script(self):
        """Create script to import the database"""
        print("📝 Creating import script...")
        
        import_script = """#!/usr/bin/env python3
\"\"\"
Database Import Script
Imports the exported database backup
\"\"\"

import os
import subprocess
import json
import tarfile
from pymongo import MongoClient

def import_database():
    print("🚀 Starting database import...")
    
    # Extract backup if compressed file exists
    if os.path.exists("database_backup.tar.gz"):
        print("📦 Extracting database backup...")
        with tarfile.open("database_backup.tar.gz", "r:gz") as tar:
            tar.extractall(".")
        print("✅ Backup extracted!")
    
    # Try mongorestore first
    if os.path.exists("database_backup"):
        try:
            print("🔄 Importing using mongorestore...")
            cmd = ["mongorestore", "database_backup/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Database imported successfully using mongorestore!")
                return True
            else:
                print(f"⚠️  Mongorestore failed: {result.stderr}")
                
        except FileNotFoundError:
            print("⚠️  mongorestore not found. Trying Python import...")
        except Exception as e:
            print(f"⚠️  Error with mongorestore: {e}")
    
    # Fallback to Python import
    return import_using_python()

def import_using_python():
    print("🐍 Importing using Python...")
    
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["ai_test_db"]
        
        backup_dir = "database_backup/ai_test_db"
        if not os.path.exists(backup_dir):
            print(f"❌ Backup directory not found: {backup_dir}")
            return False
        
        for filename in os.listdir(backup_dir):
            if filename.endswith('.json'):
                collection_name = filename.replace('.json', '')
                print(f"📥 Importing {collection_name}...")
                
                with open(os.path.join(backup_dir, filename), 'r') as f:
                    documents = json.load(f)
                
                if documents:
                    # Clear existing collection
                    db[collection_name].drop()
                    # Insert documents
                    db[collection_name].insert_many(documents)
                    print(f"   ✅ Imported {len(documents):,} documents")
        
        print("✅ Database imported successfully using Python!")
        return True
        
    except Exception as e:
        print(f"❌ Error during Python import: {e}")
        return False

if __name__ == "__main__":
    success = import_database()
    if success:
        print("\\n🎉 Database import completed!")
        print("You can now run: streamlit run app_dynamic.py")
    else:
        print("\\n❌ Database import failed!")
"""
        
        with open("import_database.py", "w") as f:
            f.write(import_script)
        
        # Make it executable on Unix systems
        try:
            os.chmod("import_database.py", 0o755)
        except:
            pass
        
        print("✅ Import script created: import_database.py")
    
    def create_database_info(self):
        """Create database information file"""
        print("📋 Creating database info file...")
        
        try:
            collections = self.db.list_collection_names()
            db_info = {
                "database_name": self.db_name,
                "export_date": datetime.now().isoformat(),
                "collections": {}
            }
            
            for collection_name in collections:
                collection = self.db[collection_name]
                count = collection.count_documents({})
                sample_doc = collection.find_one()
                
                db_info["collections"][collection_name] = {
                    "document_count": count,
                    "sample_fields": list(sample_doc.keys()) if sample_doc else []
                }
            
            with open("database_info.json", "w") as f:
                json.dump(db_info, f, indent=2, default=str)
            
            print("✅ Database info saved: database_info.json")
            
        except Exception as e:
            print(f"❌ Error creating database info: {e}")
    
    def export_database(self):
        """Main export function"""
        print("🚀 AI Database Export Utility")
        print("=" * 40)
        
        # Check if database exists
        if not self.check_database_exists():
            return False
        
        print(f"\\n📤 Exporting database: {self.db_name}")
        
        # Try mongodump first, fallback to Python
        success = self.export_using_mongodump()
        if not success:
            success = self.export_using_python()
        
        if not success:
            print("❌ Database export failed!")
            return False
        
        # Create compressed backup
        if not self.create_compressed_backup():
            return False
        
        # Create helper files
        self.create_import_script()
        self.create_database_info()
        
        # Clean up uncompressed backup
        if os.path.exists("database_backup"):
            shutil.rmtree("database_backup")
            print("🧹 Cleaned up temporary files")
        
        print("\\n🎉 Database export completed successfully!")
        print("📁 Files created:")
        print("   • database_backup.tar.gz (main backup)")
        print("   • import_database.py (import script)")
        print("   • database_info.json (database details)")
        print("\\n✅ Ready to push to GitHub!")
        
        return True

def main():
    exporter = DatabaseExporter()
    exporter.export_database()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
AI Database Analytics - Easy Setup Script
Automates the complete installation and setup process
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and show progress"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_prerequisites():
    """Check if prerequisites are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version compatible")
    
    # Check pip
    try:
        subprocess.run(["pip", "--version"], capture_output=True, check=True)
        print("✅ pip is available")
    except:
        print("❌ pip not found")
        return False
    
    # Check MongoDB
    try:
        subprocess.run(["mongod", "--version"], capture_output=True, check=True)
        print("✅ MongoDB is installed")
    except:
        print("⚠️  MongoDB not found - please install MongoDB first")
        print("   Download from: https://www.mongodb.com/try/download/community")
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Try minimal requirements first
    if os.path.exists("requirements-minimal.txt"):
        success = run_command("pip install -r requirements-minimal.txt", "Installing core dependencies")
    else:
        success = run_command("pip install -r requirements.txt", "Installing all dependencies")
    
    return success

def setup_database():
    """Import the sample database"""
    print("\n🗄️  Setting up database...")
    
    # Check if MongoDB is running
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB is running")
    except:
        print("⚠️  MongoDB not running. Starting MongoDB...")
        print("   Please start MongoDB manually: mongod")
        print("   Then run: python import_database.py")
        return False
    
    # Import database
    if os.path.exists("import_database.py"):
        success = run_command("python import_database.py", "Importing sample database")
        return success
    else:
        print("❌ import_database.py not found")
        return False

def verify_installation():
    """Verify the installation"""
    print("\n🔍 Verifying installation...")
    
    if os.path.exists("verify_installation.py"):
        return run_command("python verify_installation.py", "Running verification")
    else:
        print("⚠️  Verification script not found, skipping...")
        return True

def main():
    print("🚀 AI Database Analytics - Easy Setup")
    print("=" * 40)
    print("This script will:")
    print("1. Check prerequisites")
    print("2. Install Python dependencies")
    print("3. Import sample database")
    print("4. Verify installation")
    print("5. Provide next steps")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed!")
        print("Please install missing requirements and try again.")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        return False
    
    # Step 3: Setup database
    if not setup_database():
        print("\n⚠️  Database setup incomplete!")
        print("You can import it manually later: python import_database.py")
    
    # Step 4: Verify installation
    verify_installation()
    
    # Success message
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\n🚀 Next Steps:")
    print("1. Start the application:")
    print("   streamlit run app_dynamic.py")
    print("\n2. Open your browser:")
    print("   http://localhost:8501")
    print("\n3. Try sample queries:")
    print("   - 'Show total sales for 2024'")
    print("   - 'Compare 2023 vs 2024 sales'")
    print("   - 'Top 10 customers by spending'")
    
    print("\n📚 Documentation:")
    print("   - README.md for detailed instructions")
    print("   - database_info.json for schema details")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        print("Please check the error and try manual installation")
#!/usr/bin/env python3
"""
Installation Verification Script
Checks if all required dependencies are properly installed
"""

import sys
import importlib
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name} ({version})")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False

def check_mongodb_connection():
    """Check if MongoDB is accessible"""
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB - Connected successfully")
        return True
    except Exception as e:
        print(f"❌ MongoDB - Connection failed: {e}")
        return False

def main():
    print("🔍 AI Database Analytics - Installation Verification")
    print("=" * 55)
    
    all_good = True
    
    # Check Python version
    print("\n📋 Python Version:")
    if not check_python_version():
        all_good = False
    
    # Core packages
    print("\n📦 Core Dependencies:")
    core_packages = [
        ("streamlit", "streamlit"),
        ("pymongo", "pymongo"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("plotly", "plotly"),
        ("scipy", "scipy"),
        ("python-dotenv", "dotenv"),
        ("faker", "faker")
    ]
    
    for package, import_name in core_packages:
        if not check_package(package, import_name):
            all_good = False
    
    # Optional packages
    print("\n🔧 Optional Dependencies:")
    optional_packages = [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("nltk", "nltk"),
        ("requests", "requests")
    ]
    
    optional_missing = 0
    for package, import_name in optional_packages:
        if not check_package(package, import_name):
            optional_missing += 1
    
    # Check MongoDB
    print("\n🗄️  Database Connection:")
    if not check_mongodb_connection():
        print("   💡 Make sure MongoDB is running: mongod")
        all_good = False
    
    # Summary
    print("\n" + "=" * 55)
    if all_good:
        print("🎉 All core dependencies are installed correctly!")
        print("✅ Ready to run: streamlit run app_dynamic.py")
    else:
        print("⚠️  Some issues found. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    if optional_missing > 0:
        print(f"📝 Note: {optional_missing} optional packages missing (not required for basic functionality)")
    
    print("\n🚀 Next Steps:")
    print("1. Import database: python import_database.py")
    print("2. Run application: streamlit run app_dynamic.py")
    print("3. Open browser: http://localhost:8501")

if __name__ == "__main__":
    main()
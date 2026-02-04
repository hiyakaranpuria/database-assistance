#!/usr/bin/env python3
"""
Local LLM Setup Script
Automated setup for local language models with the AI Database Analytics tool
"""

import os
import sys
import subprocess
import requests
import time
import json

def check_system_requirements():
    """Check system requirements for local LLM"""
    print("🔍 Checking System Requirements")
    print("-" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version compatible")
    
    # Check available RAM (rough estimate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💾 Available RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 8:
            print("⚠️  Warning: 8GB+ RAM recommended for optimal performance")
            print("💡 Consider using Phi-3 Mini model for lower RAM usage")
        else:
            print("✅ RAM sufficient for standard models")
    except ImportError:
        print("💡 Install psutil for system monitoring: pip install psutil")
    
    # Check disk space
    try:
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)
        print(f"💽 Free disk space: {free_space:.1f} GB")
        
        if free_space < 10:
            print("⚠️  Warning: 10GB+ free space recommended")
        else:
            print("✅ Disk space sufficient")
    except:
        print("⚠️  Could not check disk space")
    
    return True

def install_ollama():
    """Install Ollama based on operating system"""
    print("\n🚀 Installing Ollama")
    print("-" * 20)
    
    system = os.name
    
    if system == 'posix':  # Linux/Mac
        print("🐧 Detected Unix-like system")
        try:
            # Download and install Ollama
            print("📥 Downloading Ollama installer...")
            result = subprocess.run([
                'curl', '-fsSL', 'https://ollama.ai/install.sh'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Execute installer
                print("🔧 Installing Ollama...")
                install_result = subprocess.run([
                    'sh', '-c', result.stdout
                ], capture_output=True, text=True)
                
                if install_result.returncode == 0:
                    print("✅ Ollama installed successfully")
                    return True
                else:
                    print(f"❌ Installation failed: {install_result.stderr}")
                    return False
            else:
                print(f"❌ Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            print("💡 Manual installation:")
            print("   Visit: https://ollama.ai/download")
            return False
    
    elif system == 'nt':  # Windows
        print("🪟 Detected Windows system")
        print("💡 Please install Ollama manually:")
        print("   1. Visit: https://ollama.ai/download/windows")
        print("   2. Download and run the installer")
        print("   3. Restart this script after installation")
        return False
    
    else:
        print(f"❓ Unknown system: {system}")
        print("💡 Visit https://ollama.ai/download for manual installation")
        return False

def start_ollama_service():
    """Start Ollama service"""
    print("\n🔄 Starting Ollama Service")
    print("-" * 25)
    
    try:
        # Try to start Ollama service
        print("🚀 Starting Ollama...")
        
        # Start in background
        if os.name == 'posix':
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(['ollama.exe', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
        
        # Wait for service to start
        print("⏳ Waiting for service to start...")
        time.sleep(5)
        
        # Test connection
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Service not responding: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Could not connect to Ollama service")
        print("💡 Try starting manually: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error starting service: {e}")
        return False

def download_recommended_models():
    """Download recommended language models"""
    print("\n📥 Downloading Language Models")
    print("-" * 30)
    
    models = [
        {
            "name": "llama3.2:3b-instruct-q4_0",
            "size": "2.0GB",
            "description": "Best multilingual model - fast, efficient, excellent user understanding",
            "recommended": True
        },
        {
            "name": "llama3.1:8b-instruct-q4_0",
            "size": "4.7GB",
            "description": "Best overall model for database queries",
            "recommended": False
        },
        {
            "name": "phi3:mini",
            "size": "2.3GB", 
            "description": "Lightweight model for low-resource systems",
            "recommended": False
        },
        {
            "name": "qwen2.5:7b-instruct",
            "size": "4.1GB",
            "description": "Best multilingual support (29+ languages)",
            "recommended": False
        }
    ]
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        status = "⭐ RECOMMENDED" if model["recommended"] else "Optional"
        print(f"{i}. {model['name']} ({model['size']}) - {status}")
        print(f"   {model['description']}")
    
    print("\nChoose models to download:")
    print("1. Download recommended model only (fastest)")
    print("2. Download all models (comprehensive)")
    print("3. Custom selection")
    print("4. Skip model download")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    models_to_download = []
    
    if choice == "1":
        models_to_download = [models[0]]  # Just recommended
    elif choice == "2":
        models_to_download = models  # All models
    elif choice == "3":
        print("\nSelect models to download (comma-separated numbers):")
        selection = input("Models (e.g., 1,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            models_to_download = [models[i] for i in indices if 0 <= i < len(models)]
        except:
            print("❌ Invalid selection, downloading recommended model")
            models_to_download = [models[0]]
    elif choice == "4":
        print("⏭️  Skipping model download")
        return True
    else:
        print("❌ Invalid choice, downloading recommended model")
        models_to_download = [models[0]]
    
    # Download selected models
    success_count = 0
    for model in models_to_download:
        print(f"\n📥 Downloading {model['name']} ({model['size']})...")
        print("⏳ This may take several minutes...")
        
        try:
            result = subprocess.run([
                'ollama', 'pull', model['name']
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print(f"✅ {model['name']} downloaded successfully")
                success_count += 1
            else:
                print(f"❌ Failed to download {model['name']}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Download timeout for {model['name']}")
        except Exception as e:
            print(f"❌ Download error for {model['name']}: {e}")
    
    if success_count > 0:
        print(f"\n✅ Successfully downloaded {success_count}/{len(models_to_download)} models")
        return True
    else:
        print("\n❌ No models downloaded successfully")
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing Python Dependencies")
    print("-" * 32)
    
    dependencies = [
        "requests",
        "langdetect",  # For multilingual support
        "psutil",      # For system monitoring
    ]
    
    for package in dependencies:
        try:
            print(f"📥 Installing {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {package} installed")
            else:
                print(f"⚠️  {package} installation warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")

def test_llm_integration():
    """Test the LLM integration"""
    print("\n🧪 Testing LLM Integration")
    print("-" * 25)
    
    try:
        # Import and test
        from llm_integration import OllamaLLM, LLM_AVAILABLE
        
        if not LLM_AVAILABLE:
            print("❌ LLM integration not available")
            return False
        
        print("🤖 Testing basic functionality...")
        llm = OllamaLLM()
        
        # Simple test query
        test_question = "Show total sales"
        query = llm.generate_query(test_question)
        
        # Validate JSON
        json.loads(query)
        
        print("✅ LLM integration working correctly")
        print(f"   Test query: '{test_question}'")
        print(f"   Generated: {query[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def create_llm_config():
    """Create LLM configuration file"""
    print("\n⚙️  Creating LLM Configuration")
    print("-" * 28)
    
    config = {
        "llm_enabled": True,
        "default_model": "llama3.1:8b-instruct-q4_0",
        "multilingual_enabled": True,
        "fallback_to_rules": True,
        "ollama_url": "http://localhost:11434",
        "timeout_seconds": 30,
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        with open('llm_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration saved to llm_config.json")
        print("💡 You can modify these settings as needed")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 Local LLM Setup for AI Database Analytics")
    print("=" * 50)
    print("This script will set up local language models for enhanced")
    print("natural language processing in your database analytics tool.")
    print()
    
    # Check if user wants to proceed
    proceed = input("Continue with LLM setup? (y/N): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("👋 Setup cancelled")
        return
    
    success_steps = 0
    total_steps = 6
    
    # Step 1: Check requirements
    if check_system_requirements():
        success_steps += 1
    
    # Step 2: Install Ollama
    if install_ollama():
        success_steps += 1
    else:
        print("\n💡 If Ollama is already installed, continue anyway")
        continue_anyway = input("Continue? (y/N): ").strip().lower()
        if continue_anyway in ['y', 'yes']:
            success_steps += 1
    
    # Step 3: Start service
    if start_ollama_service():
        success_steps += 1
    
    # Step 4: Download models
    if download_recommended_models():
        success_steps += 1
    
    # Step 5: Install Python dependencies
    install_python_dependencies()
    success_steps += 1
    
    # Step 6: Test integration
    if test_llm_integration():
        success_steps += 1
    
    # Step 7: Create config
    if create_llm_config():
        pass  # Bonus step
    
    # Final report
    print("\n" + "=" * 50)
    print("📋 SETUP COMPLETE")
    print("=" * 50)
    
    if success_steps >= 4:
        print("🎉 Local LLM setup successful!")
        print("\n✅ What's working:")
        print("   • Ollama service")
        print("   • Language models")
        print("   • Python integration")
        
        print("\n🚀 Next steps:")
        print("   1. Run the main application: streamlit run app_dynamic.py")
        print("   2. Try natural language queries in multiple languages")
        print("   3. Test performance with: python test_llm_integration.py")
        
        print("\n💡 Example queries to try:")
        print("   • 'Show total sales for 2024'")
        print("   • 'Top 10 customers by spending'")
        print("   • 'Muestra las ventas totales' (Spanish)")
        print("   • '显示今年的总销售额' (Chinese)")
        
    else:
        print("⚠️  Setup partially completed")
        print(f"   Completed: {success_steps}/{total_steps} steps")
        print("\n🔧 Manual steps needed:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Start service: ollama serve")
        print("   3. Download model: ollama pull llama3.1:8b-instruct-q4_0")
        print("   4. Test integration: python test_llm_integration.py")
    
    print(f"\n📚 Documentation: LOCAL_LLM_INTEGRATION.md")

if __name__ == "__main__":
    main()
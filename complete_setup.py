#!/usr/bin/env python3
"""
MongoDB Assistant - Complete Setup Wizard
Handles: Identity verification, OTP, MongoDB config, Ollama setup
"""

import subprocess
import sys
import os
import time
import requests
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configuration
API_URL = os.getenv('API_URL', 'https://mongodb-assistant-backend.onrender.com')

# Colors
class C:
    G = '\033[92m'  # Green
    R = '\033[91m'  # Red
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    C = '\033[96m'  # Cyan
    W = '\033[97m'  # White
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{C.C}{'═'*70}{C.END}")
    print(f"{C.BOLD}{C.W}  {text}{C.END}")
    print(f"{C.C}{'═'*70}{C.END}\n")

def print_step(step, total, desc):
    print(f"\n{C.B}┌{'─'*68}┐{C.END}")
    print(f"{C.B}│{C.END} {C.BOLD}STEP {step}/{total}: {desc}{C.END}")
    print(f"{C.B}└{'─'*68}┘{C.END}\n")

def ok(text): print(f"{C.G}✅ {text}{C.END}")
def err(text): print(f"{C.R}❌ {text}{C.END}")
def warn(text): print(f"{C.Y}⚠️  {text}{C.END}")
def info(text): print(f"{C.C}ℹ️  {text}{C.END}")

# Step 1: Verify Secret Key
def verify_key():
    print_step(1, 5, "Secret Key Verification")
    print(f"{C.W}Enter your secret key from dashboard{C.END}")
    print(f"{C.C}Format: sk_live_xxxxxxxxxxxxxxxx{C.END}\n")
    
    for attempt in range(3):
        key = input(f"{C.BOLD}🔑 Secret Key: {C.END}").strip()
        if not key:
            err("Key cannot be empty")
            continue
        
        print(f"\n{C.C}🔍 Verifying...{C.END}")
        try:
            r = requests.post(f"{API_URL}/api/verification/verify-key",
                            json={"identityKey": key}, timeout=15)
            if r.status_code == 200:
                email = r.json().get('email', 'your email')
                ok("Key verified!")
                info(f"Email: {email}")
                return key, email
            else:
                err(f"Invalid key: {r.json().get('message')}")
        except:
            err("Cannot connect to server")
        
        if attempt < 2:
            warn(f"Attempts remaining: {2-attempt}")
    
    err("Max attempts reached")
    sys.exit(1)

# Step 2: OTP Verification
def verify_otp(key, email):
    print_step(2, 5, "Email OTP Verification")
    print(f"{C.W}Sending code to: {C.C}{email}{C.END}\n")
    
    print(f"{C.C}📤 Sending...{C.END}")
    try:
        r = requests.post(f"{API_URL}/api/verification/send-otp",
                        json={"identityKey": key}, timeout=30)
        if r.status_code != 200:
            err("Failed to send OTP")
            return False
        ok("Code sent! Check your email\n")
    except:
        err("Cannot send OTP")
        return False
    
    for attempt in range(3):
        otp = input(f"{C.BOLD}📧 Enter 6-digit OTP: {C.END}").strip()
        if len(otp) != 6 or not otp.isdigit():
            err("Must be 6 digits")
            continue
        
        print(f"{C.C}🔐 Verifying...{C.END}")
        try:
            r = requests.post(f"{API_URL}/api/verification/verify-otp",
                            json={"identityKey": key, "otp": otp}, timeout=15)
            if r.status_code == 200:
                ok("Email verified!")
                return True
            else:
                err("Invalid OTP")
        except:
            err("Verification failed")
        
        if attempt < 2:
            warn(f"Attempts remaining: {2-attempt}")
    
    return False

# Step 3: MongoDB Configuration
def config_mongo():
    print_step(3, 5, "MongoDB Configuration")
    print(f"{C.W}Choose connection type:{C.END}\n")
    print(f"  {C.C}1{C.END}. MongoDB Atlas (paste connection string)")
    print(f"  {C.C}2{C.END}. Local MongoDB (mongodb://localhost:27017)")
    print(f"  {C.C}3{C.END}. Docker Host (mongodb://host.docker.internal:27017)\n")
    
    choice = input(f"{C.BOLD}Select (1-3) [1]: {C.END}").strip() or "1"
    
    if choice == "1":
        print()
        uri = input(f"{C.BOLD}📎 MongoDB URI: {C.END}").strip()
    elif choice == "2":
        uri = "mongodb://localhost:27017"
    elif choice == "3":
        uri = "mongodb://host.docker.internal:27017"
    else:
        uri = "mongodb://localhost:27017"
    
    db_name = input(f"{C.BOLD}📁 Database name [test]: {C.END}").strip() or "test"
    
    print(f"\n{C.C}🔍 Testing connection...{C.END}")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=10000)
        client.admin.command('ping')
        db = client[db_name]
        colls = db.list_collection_names()
        client.close()
        
        ok(f"Connected to: {db_name}")
        info(f"Found {len(colls)} collections")
        
        # Save config to .env file
        with open('/app/.env', 'w') as f:
            f.write(f'MONGO_URI={uri}\n')
            f.write(f'MONGO_DB={db_name}\n')
        
        # Set in environment
        os.environ['MONGO_URI'] = uri
        os.environ['MONGO_DB'] = db_name
        
        ok("Configuration saved!")
        return True
    except:
        err("Connection failed")
        retry = input(f"\n{C.Y}Try again? (y/n): {C.END}").lower()
        if retry == 'y':
            return config_mongo()
        warn("Continuing without database")
        return False

# Step 4: Ollama Setup
def setup_ollama():
    print_step(4, 5, "AI Model Setup (Ollama)")
    
    # Check if installed
    try:
        subprocess.run(['ollama', '--version'], capture_output=True, timeout=5)
        ok("Ollama is installed")
    except:
        warn("Ollama not installed")
        install = input(f"{C.BOLD}Install Ollama? (y/n) [y]: {C.END}").lower()
        if install != 'n':
            print(f"{C.C}📥 Installing...{C.END}")
            try:
                subprocess.run('curl -fsSL https://ollama.com/install.sh | sh',
                             shell=True, timeout=300)
                ok("Ollama installed!")
            except:
                err("Installation failed")
                return False
        else:
            info("Skipping AI features")
            return False
    
    # Start service
    print(f"\n{C.C}🚀 Starting Ollama...{C.END}")
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(3)
    
    # Check/download model
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'qwen2.5:3b' in result.stdout:
            ok("Model qwen2.5:3b ready")
        else:
            download = input(f"{C.BOLD}Download model (~2GB)? (y/n) [y]: {C.END}").lower()
            if download != 'n':
                print(f"{C.C}📥 Downloading (5-10 min)...{C.END}")
                subprocess.run(['ollama', 'pull', 'qwen2.5:3b'])
                ok("Model downloaded!")
    except:
        warn("Model check failed")
    
    # Load Modelfile
    if os.path.exists('/app/Modelfile'):
        print(f"\n{C.C}🔧 Loading Modelfile...{C.END}")
        try:
            subprocess.run(['ollama', 'create', 'db-assistant', '-f', '/app/Modelfile'],
                         timeout=120)
            ok("Custom model 'db-assistant' ready!")
        except:
            warn("Modelfile loading failed")
    
    return True

# Step 5: Launch App
def launch_app():
    print_step(5, 5, "Launching Application")
    
    print(f"\n{C.G}╔══════════════════════════════════════════════════════════╗{C.END}")
    print(f"{C.G}║  {C.BOLD}🎉 SETUP COMPLETE! MongoDB Assistant is Ready!{C.END}{C.G}  ║{C.END}")
    print(f"{C.G}║                                                          ║{C.END}")
    print(f"{C.G}║  📍 Access at: {C.C}http://localhost:8501{C.END}{C.G}                  ║{C.END}")
    print(f"{C.G}║  ⏹️  Press Ctrl+C to stop                                ║{C.END}")
    print(f"{C.G}╚══════════════════════════════════════════════════════════╝{C.END}\n")
    
    print(f"{C.C}🚀 Starting Streamlit...{C.END}\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run',
                       '/app/app_dynamic.py', '--server.port=8501',
                       '--server.address=0.0.0.0', '--server.headless=true'])
    except KeyboardInterrupt:
        print(f"\n\n{C.Y}👋 Goodbye!{C.END}")

# Main
def main():
    print_header("🎯 MongoDB Assistant - Setup Wizard")
    
    print(f"{C.W}This wizard will guide you through:{C.END}\n")
    print(f"  {C.C}1.{C.END} Verify secret key")
    print(f"  {C.C}2.{C.END} Email OTP verification")
    print(f"  {C.C}3.{C.END} Configure MongoDB")
    print(f"  {C.C}4.{C.END} Setup AI model (optional)")
    print(f"  {C.C}5.{C.END} Launch application\n")
    
    input(f"{C.BOLD}Press Enter to continue...{C.END}")
    
    key, email = verify_key()
    if not verify_otp(key, email):
        err("OTP verification failed")
        sys.exit(1)
    
    config_mongo()
    setup_ollama()
    launch_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{C.Y}👋 Setup cancelled{C.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{C.R}❌ Error: {e}{C.END}")
        sys.exit(1)

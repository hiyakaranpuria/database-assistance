#!/usr/bin/env python3
import os
import sys
import time
import json
import getpass
import requests
import subprocess
from pymongo import MongoClient
import platform
import io

# Force UTF-8 encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Hardcoded Production Backend URL
BACKEND_URL = "https://mongoassist-backend.onrender.com"

# Local config file to persist setup state across container restarts
# Mount a Docker volume at /app/data to persist this file
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CONFIG_FILE = os.path.join(CONFIG_DIR, ".mongoassist_config.json")

# Persist Ollama models in the volume so they survive container restarts
OLLAMA_DATA_DIR = os.path.join(CONFIG_DIR, "ollama")
OLLAMA_MODELS_DIR = os.path.join(OLLAMA_DATA_DIR, "models")

def load_config():
    """Load saved config from previous setup, if it exists."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return None

def save_config(identity_key, mongo_uri, db_name):
    """Save setup config locally so the wizard can be skipped on next run."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config = {
        "identity_key": identity_key,
        "mongo_uri": mongo_uri,
        "db_name": db_name,
        "setup_completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def verify_saved_db(config):
    """Verify that the saved database connection is still reachable."""
    print("  Checking database connection... ", end="", flush=True)
    try:
        client = MongoClient(config["mongo_uri"], serverSelectionTimeoutMS=3000)
        client.admin.command('ping')
        print("✅")
        return True
    except Exception:
        print("❌ Database unreachable.")
        return False


def print_header(title):
    print("\n" + "━" * 60)
    print(title)
    print("━" * 60 + "\n")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# --- Backend Reachability Check
def check_backend():
    global BACKEND_URL
    
    # Render free tier cold-starts can take 30-50 seconds.
    # Retry a few times with increasing timeouts instead of failing instantly.
    max_attempts = 4
    timeouts = [5, 15, 20, 25]  # total wait: up to ~65 seconds
    
    for attempt in range(max_attempts):
        timeout = timeouts[attempt]
        
        try:
            res = requests.get(f"{BACKEND_URL}/api/health", timeout=timeout)
            if res.status_code == 200:
                return True
        except requests.RequestException:
            pass

        # Try localhost fallback (for Linux host networking)
        fallback_url = "http://localhost:5000"
        try:
            res = requests.get(f"{fallback_url}/api/health", timeout=3)
            if res.status_code == 200:
                BACKEND_URL = fallback_url
                return True
        except requests.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            print(f"  ⏳ Backend waking up... (attempt {attempt + 1}/{max_attempts})")
        
    print(f"❌ Cannot reach verification server at {BACKEND_URL} or localhost.")
    print("Please ensure your backend is running or pass a valid BACKEND_URL.")
    return False

# --- Key Verification (returns identity key only, OTP is always required separately)
def verify_identity_key():
    print_header("STEP 1 — Identity Verification")
    
    while True:
        key = input("Enter your identity key (sk_live_...): ").strip()
        if not key:
            continue
            
        print("  Verifying key... ", end="", flush=True)
            
        try:
            res = requests.post(f"{BACKEND_URL}/api/verify/key", json={"identityKey": key}, timeout=5)
            data = res.json()
            
            if res.status_code == 200 and data.get("success"):
                print("✅ Key valid!")
                print(f"  Account: {data.get('maskedEmail')}")
                return key
            else:
                print(f"❌ Invalid key: {data.get('message', 'Key not found.')}")
                print("Get your key at https://mongoassist.com/signup")
                
        except requests.RequestException as e:
            print("❌ Error reaching backend.")
            print(e)
            
# --- OTP Verification (always required)
def verify_otp(identity_key):
    print_header("STEP 2 — Email Verification")
    
    # Send OTP
    try:
        print("  Sending OTP to your email... ", end="", flush=True)
        res = requests.post(f"{BACKEND_URL}/api/verify/send-otp", json={"identityKey": identity_key}, timeout=30)
        data = res.json()
        if res.status_code == 200 and data.get("success"):
            print(f"✅ OTP sent to {data.get('maskedEmail')}")
        else:
            print("❌ Failed to send OTP:", data.get("message"))
            sys.exit(1)
    except requests.RequestException as e:
        print(f"❌ Error reaching backend to send OTP: {e}")
        sys.exit(1)

    # Verify Loop
    retries = 3
    while retries > 0:
        otp = input("\nEnter 6-digit OTP (or 'R' to resend): ").strip()
        if otp.upper() == 'R':
            verify_otp(identity_key)
            return
            
        print("  Verifying OTP... ", end="", flush=True)
        try:
            res = requests.post(f"{BACKEND_URL}/api/verify/verify-otp", json={"identityKey": identity_key, "otp": otp}, timeout=5)
            data = res.json()
            if res.status_code == 200 and data.get("success"):
                print("✅ Verified!")
                return
            elif data.get("expired"):
                print(f"❌ {data.get('message')}")
                retries = 3
            else:
                print(f"❌ {data.get('message')}")
                retries -= 1
                if retries > 0:
                    print(f"  ({retries} attempts remaining)")
        except requests.RequestException:
            print("❌ Error reaching backend.")
            sys.exit(1)
            
    print("❌ Too many failed attempts. Exiting.")
    sys.exit(1)

# --- Database Connection Setup
def setup_database():
    print_header("STEP 3 — Database Connection")
    print("How do you want to connect your database?\n")
    print("  [1] 🖥️  Local MongoDB (running on this machine)")
    print("  [2] 🌐  Remote / Atlas (connection string)\n")
    
    while True:
        choice = input("Select [1/2]: ").strip()
        if choice in ['1', '2']:
            break
            
    uri = ""
    db_name = ""
    
    if choice == '1':
        print("\nDetecting OS for local connection...")
        system = platform.system().lower()
        if system == "linux" and os.path.exists("/.dockerenv"):
            uri = "mongodb://127.0.0.1:27017"
            print("  → Linux detected. Using localhost direct connection.")
        else:
            uri = "mongodb://host.docker.internal:27017"
            print("  → Windows/Mac detected. Using host.docker.internal.")
            
        print(f"\nWe'll connect via: {uri}")
        print("Testing connection... ", end="", flush=True)
        
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            print("✅ Connected!")
        except Exception as e:
            print(f"❌ Connection failed.\nIs your local MongoDB running on port 27017?\n({str(e)})")
            sys.exit(1)
            
        db_name = input("\nEnter your database name: ").strip()
        
    else:
        uri = input("\nPaste your MongoDB connection string (e.g. mongodb+srv://...): ").strip()
        print("Testing connection... ", end="", flush=True)
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            print("✅ Connected!\n")
            
            dbs = client.list_database_names()
            dbs = [db for db in dbs if db not in ['admin', 'local', 'config']]
            
            if not dbs:
                print("⚠️ No user databases found on this cluster.")
                db_name = input("Enter a database name to create/use: ").strip()
            else:
                print("Available databases:")
                for i, db in enumerate(dbs, 1):
                    print(f"  [{i}] {db}")
                    
                idx = input(f"\nSelect database [1-{len(dbs)}]: ").strip()
                try:
                    db_name = dbs[int(idx)-1]
                    print(f"  → Using: {db_name}")
                except (ValueError, IndexError):
                    print("Invalid selection. Exiting.")
                    sys.exit(1)
                    
        except Exception as e:
            print(f"❌ Connection failed.\n({str(e)})")
            sys.exit(1)
            
    os.environ["MONGO_URI"] = uri
    os.environ["MONGO_DB"] = db_name
    return uri, db_name

# --- AI Model Setup (with persistence — skips download if already done)
def setup_ai_model(is_first_setup=True):
    print_header("AI Model Setup" if is_first_setup else "Starting AI Model")
    
    # Point Ollama model storage to the persistent volume
    os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)
    os.environ["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR
    
    # 1. Check/Install Ollama
    print("  Checking Ollama... ", end="", flush=True)
    ollama_installed = subprocess.run(["which", "ollama"], capture_output=True).returncode == 0
    
    if not ollama_installed:
        print("Installing...")
        try:
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("  ✅ Ollama installed.")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama.")
            sys.exit(1)
    else:
        print("✅ Already installed")
        
    # 2. Start Ollama Server
    print("  Starting Ollama service... ", flush=True)
    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS_DIR},
    )
    time.sleep(3)
    
    # 3. Check if model already exists (skip download if it does)
    model_exists = False
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if "qwen2.5:3b" in result.stdout:
            model_exists = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    
    if model_exists:
        print("  ✅ Model qwen2.5:3b already downloaded — skipping pull.")
    else:
        print("  Pulling qwen2.5:3b (this may take a while)...")
        subprocess.run(["ollama", "pull", "qwen2.5:3b"])
        print("  ✅ Model downloaded.")
    
    # 4. Generate Modelfile based on DB Schema
    print("\n  Discovering your schema & Regenerating Modelfile...")
    try:
        subprocess.run([sys.executable, "generate_modelfile.py"], check=True)
    except subprocess.CalledProcessError:
        print("  ⚠️ Schema discovery script failed. Using default Modelfile.")
        
    # 5. Create Model
    print("  Creating db-assistant AI model...")
    subprocess.run(["ollama", "create", "db-assistant", "-f", "Modelfile"], check=True)
    print("  ✅ db-assistant ready!")
    
    return ollama_process

# --- Launch Streamlit
def launch_streamlit():
    print_header("🚀 Launching")
    print("  Starting MongoDB AI Assistant...\n")
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  Open in your browser:                      │")
    print("  │  → http://localhost:8501                     │")
    print("  └─────────────────────────────────────────────┘\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app_dynamic.py", "--server.port=8501", "--server.address=0.0.0.0"])
    except KeyboardInterrupt:
        print("\nShutting down...")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    clear_screen()
    print("╔" + "═" * 60 + "╗")
    print("║            🍃  MongoAssist — Setup Wizard                  ║")
    print("╚" + "═" * 60 + "╝")

    # ── Check for saved config from a previous run ──
    saved = load_config()
    
    if saved:
        # ─── RETURNING USER FLOW ───
        # Saved config exists → validate DB, require OTP, then launch (no re-download)
        print("\n  ℹ️  Found saved setup from a previous run.")
        print(f"     Database: {saved.get('db_name', '?')}")
        print(f"     Set up at: {saved.get('setup_completed_at', '?')}\n")

        choice = input("  Continue with saved setup? [Y/n/reset]: ").strip().lower()
        
        if choice == 'reset':
            print("  🗑️  Clearing saved config. Running full setup...\n")
            try:
                os.remove(CONFIG_FILE)
            except OSError:
                pass
            saved = None  # Fall through to full setup below
            
        elif choice in ('', 'y', 'yes'):
            # Validate database is still reachable
            print("\n  Validating saved configuration...")
            if not verify_saved_db(saved):
                print("\n  ⚠️  Database unreachable. Running full setup...\n")
                saved = None  # Fall through to full setup
            else:
                # Database OK → now require OTP for security
                print("\n  📡 Connecting to verification server (may take ~30s on free tier)...")
                if not check_backend():
                    print("  ⚠️  Backend unreachable — cannot verify OTP. Exiting.")
                    sys.exit(1)
                    
                # Use saved identity key to send & verify OTP
                print(f"\n  🔐 OTP verification required to continue.\n")
                verify_otp(saved["identity_key"])
                
                # OTP passed — restore env vars and launch (skip DB setup & model download)
                os.environ["MONGO_URI"] = saved["mongo_uri"]
                os.environ["MONGO_DB"] = saved["db_name"]
                
                ollama_proc = setup_ai_model(is_first_setup=False)
                try:
                    launch_streamlit()
                finally:
                    if ollama_proc:
                        ollama_proc.terminate()
                return
        else:
            saved = None  # 'n' — fall through to full setup

    # ─── FIRST-TIME SETUP FLOW ───
    if not check_backend():
        print()
        print("Note: In localized development without backend, continuing...")
      
    # Step 1: Verify identity key
    key = verify_identity_key()
    
    # Step 2: OTP verification (always required)
    verify_otp(key)
        
    # Step 3: Database connection
    uri, db = setup_database()

    # Save config for next time
    save_config(key, uri, db)
    print(f"\n  💾 Setup saved! Next time you'll only need to verify your OTP.")

    # Step 4: AI model setup (first time — will download)
    ollama_proc = setup_ai_model(is_first_setup=True)
    
    # Step 5: Launch
    try:
        launch_streamlit()
    finally:
        if ollama_proc:
            ollama_proc.terminate()

if __name__ == "__main__":
    main()

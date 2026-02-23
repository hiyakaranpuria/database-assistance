@echo off
echo ============================================================
echo MongoDB Chat Assistant - Quick Start
echo ============================================================
echo.

echo Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python not found. Please install Python first.
    pause
    exit /b 1
)
echo [OK] Python installed

REM Check MongoDB
mongosh --eval "db.version()" >nul 2>&1
if errorlevel 1 (
    echo [X] MongoDB not running. Please start MongoDB first.
    echo     Run: mongod
    pause
    exit /b 1
)
echo [OK] MongoDB running

REM Check Ollama
ollama list >nul 2>&1
if errorlevel 1 (
    echo [X] Ollama not running. Please start Ollama first.
    echo     Run: ollama serve
    pause
    exit /b 1
)
echo [OK] Ollama running

echo.
echo Starting chat assistant...
echo.

python simple_chat_flow.py

pause

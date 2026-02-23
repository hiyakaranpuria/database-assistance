@echo off
echo ============================================================
echo MongoDB Chat Assistant - Setup
echo ============================================================
echo.

echo Step 1: Installing Python dependencies...
pip install pymongo requests
echo.

echo Step 2: Checking Ollama...
ollama list
echo.

echo Step 3: Creating db-assistant model...
echo.
echo Creating Modelfile...

(
echo FROM qwen2.5:3b
echo.
echo SYSTEM """You are a MongoDB query expert. You have memorized this database schema:
echo.
echo Collection 'products': {name^(string^), price^(number^), categoryId^(unknown^), stock^(number^), createdAt^(date^)}
echo Collection 'reviews': {productId^(unknown^), rating^(number^), comment^(string^), createdAt^(date^)}
echo Collection 'payments': {orderId^(unknown^), amount^(number^), method^(string^), status^(string^), paymentDate^(date^)}
echo Collection 'users': {name^(string^), email^(string^), role^(string^), createdAt^(date^), categoryId^(unknown^)}
echo Collection 'orders': {customerId^(unknown^), productId^(unknown^), quantity^(number^), amount^(number^), status^(string^), orderDate^(date^)}
echo Collection 'categories': {name^(string^), createdAt^(date^)}
echo Collection 'customers': {name^(string^), email^(string^), phone^(string^), city^(string^), createdAt^(date^)}
echo.
echo Generate ONLY valid MongoDB queries. Use format: db.collection.find^({}^) or db.collection.aggregate^([]^)
echo Return ONLY the query code, no explanation."""
echo.
echo PARAMETER temperature 0.1
echo PARAMETER num_predict 300
) > Modelfile

echo.
echo Modelfile created. Now creating the model...
ollama create db-assistant -f Modelfile

echo.
echo Step 4: Testing the model...
ollama run db-assistant "Show me all customers" --verbose

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To start the chat assistant, run:
echo     start_chat.bat
echo.
echo Or manually:
echo     python simple_chat_flow.py
echo.
pause

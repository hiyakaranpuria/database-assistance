# 🚀 AI Database Analytics Tool

Complete database analytics tool with interactive visualizations and natural language querying. Includes sample restaurant data with 17,000+ records for immediate testing.

## ✨ Features

- 📊 **Interactive Analytics Dashboard** - Multiple analysis views with charts
- 💬 **Natural Language Queries** - Ask questions in plain English
- 📈 **Year-over-Year Comparisons** - Sales growth analysis with metrics
- 📅 **Monthly Trends** - Seasonal pattern analysis with interactive charts
- 🛍️ **Product Performance** - Top/bottom performers with visualizations
- 👥 **Customer Insights** - Spending patterns and segmentation analysis
- 🗺️ **Geographic Analysis** - Sales distribution by city with pie charts
- 🧹 **Data Cleaning Tools** - Automated data quality analysis and cleaning
- 🔧 **Custom Query Interface** - Direct MongoDB query execution

## 🗃️ Sample Data Included

- **1,000 customers** across 4 cities (Mumbai, Delhi, Jaipur, Udaipur)
- **264 products** in 5 categories (Electronics, Clothing, Furniture, Books, Groceries)
- **5,000 orders** spanning 2 years (2023-2024)
- **3,500 payments** with different methods (card, UPI, netbanking, COD)
- **1,200 product reviews** with ratings and comments
- **200 system users** with different roles

## 🚀 Quick Setup (3 Steps)

### Prerequisites
- Python 3.8+
- MongoDB installed and running

### Installation Options

#### Option 1: Full Installation (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/yourusername/ai-database-analytics
cd ai-database-analytics

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Import sample database
python import_database.py

# 4. Run application
streamlit run app_dynamic.py
```

#### Option 2: Minimal Installation (Core Features Only)
```bash
# For basic functionality only
pip install -r requirements-minimal.txt
```

#### Option 3: Development Installation (All Features)
```bash
# For development with advanced features
pip install -r requirements-dev.txt
```

### Open Browser
Navigate to: **http://localhost:8501**

## 📊 Try These Sample Queries

### Sales Analysis
- "Show total sales for 2024"
- "Compare 2023 vs 2024 sales"
- "Show monthly sales trends"
- "What was our best performing month?"

### Product Analysis
- "Which products perform worst?"
- "Show top 10 products by revenue"
- "Which category generates most sales?"

### Customer Analysis
- "Top 10 customers by spending"
- "How many customers do we have?"
- "Show customer distribution by city"

### Geographic Analysis
- "Sales breakdown by city"
- "Which city generates most revenue?"

## 🎯 Analytics Dashboard

Use the sidebar to navigate between different analysis views:

### 1. **Chat Interface**
Natural language querying with automatic chart generation

### 2. **Sales Comparison** 
Year-over-year analysis with growth metrics and trend charts

### 3. **Monthly Trends**
Seasonal pattern analysis with interactive line and bar charts

### 4. **Product Analysis**
Performance metrics with top/bottom performers visualization

### 5. **Customer Insights**
Spending behavior analysis with segmentation and distribution charts

### 6. **Geographic Analysis**
Location-based sales analysis with pie charts and city metrics

### 7. **Custom Query**
Direct MongoDB aggregation pipeline execution with auto-visualization

## 🧹 Data Cleaning Features

The tool includes automated data cleaning capabilities:

- **Duplicate Detection** - Finds and removes duplicate records
- **Email Validation** - Standardizes email formats
- **Phone Standardization** - Normalizes phone number formats
- **Data Quality Reports** - Comprehensive analysis of data issues
- **Safe Cleaning** - All operations include backup and rollback options

### Run Data Cleaning
```bash
python data_cleaner.py
```

## 🔧 Technical Architecture

### Core Components
- **Dynamic Query Generator** - Converts natural language to MongoDB queries
- **Metadata Provider** - Automatic database schema discovery
- **Response Formatter** - Converts raw data to human-readable responses
- **Memory Manager** - Chat history and context management
- **Analytics Engine** - Statistical analysis and visualization generation

### Visualization Stack
- **Plotly** - Interactive charts and graphs
- **Pandas** - Data manipulation and analysis
- **Streamlit** - Web interface and dashboard
- **MongoDB** - Database and aggregation engine

## 📁 Project Structure

```
ai-database-analytics/
├── app_dynamic.py              # Main Streamlit application
├── requirements.txt            # Python dependencies
├── database_backup.tar.gz      # Sample database (compressed)
├── import_database.py          # Database import utility
├── database_info.json          # Database schema information
├── metadata_provider.py        # Schema discovery
├── dynamic_query_generator.py  # Query generation
├── dynamic_query_executor.py   # Query execution
├── dynamic_response_formatter.py # Response formatting
├── memory_manager.py           # Chat history
├── data_cleaner.py            # Data quality analysis
├── database_cleaner_executor.py # Data cleaning execution
└── mongodb_cleaning_queries.js # MongoDB cleaning scripts
```

## 📦 Dependencies Overview

### Requirements Files
- **`requirements.txt`** - Complete installation with all features
- **`requirements-minimal.txt`** - Core functionality only (faster install)
- **`requirements-dev.txt`** - Development setup with advanced features

### Core Dependencies
- **Streamlit** - Web dashboard framework
- **PyMongo** - MongoDB database connectivity
- **Pandas** - Data manipulation and analysis
- **Plotly** - Interactive data visualization
- **NumPy** - Numerical computing
- **SciPy** - Statistical analysis

### Optional Dependencies
- **Matplotlib/Seaborn** - Additional visualization options
- **Scikit-learn** - Machine learning capabilities
- **NLTK** - Natural language processing
- **Faker** - Sample data generation
- **Jupyter** - Notebook support for analysis

## 🛠️ Troubleshooting

### MongoDB Issues
```bash
# Start MongoDB
mongod

# Check if MongoDB is running
mongo --eval "db.adminCommand('ismaster')"
```

### Port Issues
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Run on different port
streamlit run app_dynamic.py --server.port 8502
```

### Import Issues
```bash
# Manual database import
tar -xzf database_backup.tar.gz
mongorestore database_backup/
```

### Package Issues
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 📈 Sample Analytics Results

The dashboard provides insights like:
- **Sales Growth**: 28% increase from 2023 to 2024
- **Top Product**: Electronics category dominates with 40% of sales
- **Best Customer**: Top 10% customers generate 60% of revenue
- **Geographic Leader**: Mumbai accounts for 35% of total sales
- **Seasonal Trends**: December shows 45% higher sales than average

## 🎓 Educational Use

Perfect for:
- **Database Analytics Learning** - Real-world data analysis scenarios
- **MongoDB Practice** - Complex aggregation pipeline examples
- **Data Visualization** - Interactive chart creation with Plotly
- **Python Development** - Full-stack application development
- **Business Intelligence** - KPI analysis and reporting

## 📧 Support

For questions or issues:
- Check the troubleshooting section above
- Review the database_info.json for schema details
- Examine the sample queries for inspiration

## 🏆 Features Showcase

This project demonstrates:
- ✅ **Full-stack development** with Python and MongoDB
- ✅ **Interactive data visualization** with Plotly and Streamlit
- ✅ **Natural language processing** for query generation
- ✅ **Database design** with realistic relational data
- ✅ **Data cleaning** and quality assurance
- ✅ **Business intelligence** dashboard creation
- ✅ **Containerization** ready (Docker support available)
- ✅ **Professional documentation** and code organization

---

**Ready to explore your data? Start with the quick setup above and begin analyzing!** 🚀
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from metadata_provider import extract_metadata
from dynamic_query_generator import generate_mongo_query
from dynamic_query_executor import execute_mongo_query
from dynamic_response_formatter import format_final_answer
from memory_manager import save_message, get_chat_history
from database_config import config
from pymongo import MongoClient

# Initialize MongoDB connection for direct analytics
@st.cache_resource
def get_db_connection():
    client = MongoClient("mongodb://localhost:27017")
    return client["ai_test_db"]

class DataAnalytics:
    def __init__(self):
        self.db = get_db_connection()
    
    def get_yearly_sales_comparison(self):
        """Compare sales between years"""
        pipeline = [
            {
                "$match": {
                    "status": "completed",
                    "orderDate": {"$exists": True}
                }
            },
            {
                "$group": {
                    "_id": {"$year": "$orderDate"},
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "avgOrderValue": {"$avg": "$amount"}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results).rename(columns={'_id': 'year'})
    
    def get_monthly_trends(self, year=2024):
        """Get monthly sales trends for a specific year"""
        pipeline = [
            {
                "$match": {
                    "status": "completed",
                    "orderDate": {
                        "$gte": datetime(year, 1, 1),
                        "$lt": datetime(year + 1, 1, 1)
                    }
                }
            },
            {
                "$group": {
                    "_id": {"$month": "$orderDate"},
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]
        
        results = list(self.db.orders.aggregate(pipeline))
        df = pd.DataFrame(results).rename(columns={'_id': 'month'})
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['monthName'] = df['month'].apply(lambda x: month_names[x-1] if x <= 12 else str(x))
        
        return df
    
    def get_top_products(self, limit=10, year=None):
        """Get top selling products"""
        match_stage = {"status": "completed"}
        
        if year:
            match_stage["orderDate"] = {
                "$gte": datetime(year, 1, 1),
                "$lt": datetime(year + 1, 1, 1)
            }
        
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": "$productId",
                    "totalSales": {"$sum": "$amount"},
                    "quantitySold": {"$sum": "$quantity"},
                    "orderCount": {"$sum": 1}
                }
            },
            {
                "$lookup": {
                    "from": "products",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "product"
                }
            },
            {"$unwind": "$product"},
            {
                "$project": {
                    "productName": "$product.name",
                    "totalSales": 1,
                    "quantitySold": 1,
                    "orderCount": 1
                }
            },
            {"$sort": {"totalSales": -1}},
            {"$limit": limit}
        ]
        
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)
    
    def get_customer_analysis(self):
        """Analyze customer behavior"""
        pipeline = [
            {"$match": {"status": "completed"}},
            {
                "$group": {
                    "_id": "$customerId",
                    "totalSpent": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "avgOrderValue": {"$avg": "$amount"}
                }
            },
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "customer"
                }
            },
            {"$unwind": "$customer"},
            {
                "$project": {
                    "customerName": "$customer.name",
                    "customerCity": "$customer.city",
                    "totalSpent": 1,
                    "orderCount": 1,
                    "avgOrderValue": 1
                }
            },
            {"$sort": {"totalSpent": -1}}
        ]
        
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)
    
    def get_city_wise_sales(self):
        """Get sales by city"""
        pipeline = [
            {"$match": {"status": "completed"}},
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "customerId",
                    "foreignField": "_id",
                    "as": "customer"
                }
            },
            {"$unwind": "$customer"},
            {
                "$group": {
                    "_id": "$customer.city",
                    "totalSales": {"$sum": "$amount"},
                    "orderCount": {"$sum": 1},
                    "customerCount": {"$addToSet": "$customerId"}
                }
            },
            {
                "$project": {
                    "city": "$_id",
                    "totalSales": 1,
                    "orderCount": 1,
                    "customerCount": {"$size": "$customerCount"}
                }
            },
            {"$sort": {"totalSales": -1}}
        ]
        
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)

# Initialize analytics
analytics = DataAnalytics()

st.set_page_config(page_title="AI Data Analytics", layout="wide")
st.title("📊 AI Data Analytics Dashboard")

# Sidebar for navigation
st.sidebar.header("📈 Analytics Options")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Chat Interface", "Sales Comparison", "Monthly Trends", "Product Analysis", 
     "Customer Insights", "Geographic Analysis", "Custom Query"]
)

if analysis_type == "Chat Interface":
    st.subheader("💬 Chat with your data using natural language")
    
    # Show database info
    with st.sidebar:
        st.header("Database Info")
        metadata = extract_metadata()
        st.write(f"**Domain:** {config.domain}")
        st.write(f"**Database:** {config.database_name}")
        st.write("**Collections:**")
        for collection, info in metadata.items():
            field_count = len(info.get('fields', {}))
            st.write(f"- {collection} ({field_count} fields)")

    # Initialize Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input
    if user_input := st.chat_input("Ask anything about your data..."):
        # Display user message
        st.chat_message("user").markdown(user_input)
        save_message("user", user_input)

        with st.spinner("Analyzing your data..."):
            try:
                # Generate query dynamically
                query_str = generate_mongo_query(user_input)
                
                # Execute query with context
                raw_results = execute_mongo_query(query_str, user_input)
                
                # Format answer dynamically
                friendly_answer = format_final_answer(user_input, raw_results)
                
            except Exception as e:
                friendly_answer = f"Sorry, I encountered an error: {str(e)}"
                query_str = "Error generating query"
            
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(friendly_answer)
            
            # Show technical details in expander
            with st.expander("🔍 Technical Details"):
                st.write("**Generated Query:**")
                st.code(query_str, language="json")
                st.write("**Raw Results:**")
                st.json(raw_results)
                
        save_message("assistant", friendly_answer)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": friendly_answer})

elif analysis_type == "Sales Comparison":
    st.subheader("📊 Year-over-Year Sales Comparison")
    
    # Get yearly data
    yearly_data = analytics.get_yearly_sales_comparison()
    
    if not yearly_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales comparison chart
            fig = px.bar(yearly_data, x='year', y='totalSales', 
                        title='Total Sales by Year',
                        labels={'totalSales': 'Total Sales ($)', 'year': 'Year'})
            fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Order count comparison
            fig2 = px.line(yearly_data, x='year', y='orderCount', 
                          title='Order Count by Year', markers=True)
            fig2.update_traces(texttemplate='%{y}', textposition='top center')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Growth calculation
        if len(yearly_data) >= 2:
            current_year = yearly_data.iloc[-1]
            previous_year = yearly_data.iloc[-2]
            
            sales_growth = ((current_year['totalSales'] - previous_year['totalSales']) / previous_year['totalSales']) * 100
            order_growth = ((current_year['orderCount'] - previous_year['orderCount']) / previous_year['orderCount']) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sales Growth", f"{sales_growth:.1f}%", 
                         f"${current_year['totalSales'] - previous_year['totalSales']:,.0f}")
            with col2:
                st.metric("Order Growth", f"{order_growth:.1f}%", 
                         f"{current_year['orderCount'] - previous_year['orderCount']:,}")
            with col3:
                avg_growth = (current_year['avgOrderValue'] - previous_year['avgOrderValue']) / previous_year['avgOrderValue'] * 100
                st.metric("Avg Order Value Growth", f"{avg_growth:.1f}%", 
                         f"${current_year['avgOrderValue'] - previous_year['avgOrderValue']:.2f}")
        
        # Data table
        st.subheader("📋 Detailed Yearly Data")
        yearly_data['totalSales'] = yearly_data['totalSales'].apply(lambda x: f"${x:,.2f}")
        yearly_data['avgOrderValue'] = yearly_data['avgOrderValue'].apply(lambda x: f"${x:.2f}")
        st.dataframe(yearly_data, use_container_width=True)
    else:
        st.warning("No sales data found for comparison.")

elif analysis_type == "Monthly Trends":
    st.subheader("📈 Monthly Sales Trends")
    
    # Year selector
    available_years = analytics.get_yearly_sales_comparison()['year'].tolist()
    selected_year = st.selectbox("Select Year", available_years, index=-1 if available_years else 0)
    
    if selected_year:
        monthly_data = analytics.get_monthly_trends(selected_year)
        
        if not monthly_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly sales trend
                fig = px.line(monthly_data, x='monthName', y='totalSales', 
                             title=f'Monthly Sales Trend - {selected_year}',
                             markers=True)
                fig.update_traces(texttemplate='$%{y:,.0f}', textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Monthly order count
                fig2 = px.bar(monthly_data, x='monthName', y='orderCount',
                             title=f'Monthly Order Count - {selected_year}')
                fig2.update_traces(texttemplate='%{y}', textposition='outside')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Monthly statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Month (Sales)", 
                         monthly_data.loc[monthly_data['totalSales'].idxmax(), 'monthName'],
                         f"${monthly_data['totalSales'].max():,.0f}")
            with col2:
                st.metric("Worst Month (Sales)", 
                         monthly_data.loc[monthly_data['totalSales'].idxmin(), 'monthName'],
                         f"${monthly_data['totalSales'].min():,.0f}")
            with col3:
                st.metric("Average Monthly Sales", 
                         f"${monthly_data['totalSales'].mean():,.0f}")
            with col4:
                st.metric("Total Orders", f"{monthly_data['orderCount'].sum():,}")
            
            # Data table
            st.subheader("📋 Monthly Breakdown")
            display_data = monthly_data.copy()
            display_data['totalSales'] = display_data['totalSales'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_data, use_container_width=True)

elif analysis_type == "Product Analysis":
    st.subheader("🛍️ Product Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        product_limit = st.slider("Number of products to show", 5, 50, 10)
    with col2:
        year_filter = st.selectbox("Filter by year", ["All Years"] + analytics.get_yearly_sales_comparison()['year'].tolist())
    
    year = None if year_filter == "All Years" else year_filter
    top_products = analytics.get_top_products(product_limit, year)
    
    if not top_products.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top products by sales
            fig = px.bar(top_products.head(10), x='totalSales', y='productName', 
                        orientation='h', title='Top Products by Sales Revenue')
            fig.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top products by quantity
            fig2 = px.bar(top_products.head(10), x='quantitySold', y='productName',
                         orientation='h', title='Top Products by Quantity Sold')
            fig2.update_traces(texttemplate='%{x}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Product performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Selling Product", 
                     top_products.iloc[0]['productName'],
                     f"${top_products.iloc[0]['totalSales']:,.0f}")
        with col2:
            avg_sales = top_products['totalSales'].mean()
            st.metric("Average Product Sales", f"${avg_sales:,.0f}")
        with col3:
            total_products = len(top_products)
            st.metric("Products Analyzed", f"{total_products}")
        
        # Detailed product table
        st.subheader("📋 Detailed Product Performance")
        display_products = top_products.copy()
        display_products['totalSales'] = display_products['totalSales'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_products, use_container_width=True)

elif analysis_type == "Customer Insights":
    st.subheader("👥 Customer Analysis")
    
    customer_data = analytics.get_customer_analysis()
    
    if not customer_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top customers by spending
            top_customers = customer_data.head(10)
            fig = px.bar(top_customers, x='totalSpent', y='customerName',
                        orientation='h', title='Top 10 Customers by Spending')
            fig.update_traces(texttemplate='$%{x:,.0f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer spending distribution
            fig2 = px.histogram(customer_data, x='totalSpent', nbins=20,
                               title='Customer Spending Distribution')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Customer metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(customer_data):,}")
        with col2:
            avg_spending = customer_data['totalSpent'].mean()
            st.metric("Avg Customer Value", f"${avg_spending:,.2f}")
        with col3:
            top_customer_spending = customer_data['totalSpent'].max()
            st.metric("Top Customer Spending", f"${top_customer_spending:,.2f}")
        with col4:
            avg_orders = customer_data['orderCount'].mean()
            st.metric("Avg Orders per Customer", f"{avg_orders:.1f}")
        
        # Customer segmentation
        st.subheader("🎯 Customer Segmentation")
        
        # Create spending segments
        customer_data['segment'] = pd.cut(customer_data['totalSpent'], 
                                        bins=3, labels=['Low Value', 'Medium Value', 'High Value'])
        
        segment_summary = customer_data.groupby('segment').agg({
            'totalSpent': ['count', 'mean', 'sum'],
            'orderCount': 'mean'
        }).round(2)
        
        st.dataframe(segment_summary, use_container_width=True)

elif analysis_type == "Geographic Analysis":
    st.subheader("🗺️ Geographic Sales Analysis")
    
    city_data = analytics.get_city_wise_sales()
    
    if not city_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales by city
            fig = px.pie(city_data, values='totalSales', names='city',
                        title='Sales Distribution by City')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Customer count by city
            fig2 = px.bar(city_data, x='city', y='customerCount',
                         title='Customer Count by City')
            fig2.update_traces(texttemplate='%{y}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Geographic metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            top_city = city_data.iloc[0]
            st.metric("Top City by Sales", 
                     top_city['city'],
                     f"${top_city['totalSales']:,.0f}")
        with col2:
            total_cities = len(city_data)
            st.metric("Cities Served", f"{total_cities}")
        with col3:
            avg_city_sales = city_data['totalSales'].mean()
            st.metric("Avg Sales per City", f"${avg_city_sales:,.0f}")
        
        # City performance table
        st.subheader("📋 City Performance Details")
        display_cities = city_data.copy()
        display_cities['totalSales'] = display_cities['totalSales'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_cities, use_container_width=True)

elif analysis_type == "Custom Query":
    st.subheader("🔧 Custom MongoDB Query")
    
    st.write("Enter your custom MongoDB aggregation pipeline:")
    
    # Sample queries
    sample_queries = {
        "Daily Sales Last 30 Days": '''[
    {
        "$match": {
            "status": "completed",
            "orderDate": {
                "$gte": new Date(new Date().setDate(new Date().getDate() - 30))
            }
        }
    },
    {
        "$group": {
            "_id": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$orderDate"
                }
            },
            "dailySales": {"$sum": "$amount"},
            "orderCount": {"$sum": 1}
        }
    },
    {"$sort": {"_id": 1}}
]''',
        "Product Category Analysis": '''[
    {
        "$lookup": {
            "from": "products",
            "localField": "productId",
            "foreignField": "_id",
            "as": "product"
        }
    },
    {"$unwind": "$product"},
    {
        "$lookup": {
            "from": "categories",
            "localField": "product.categoryId",
            "foreignField": "_id",
            "as": "category"
        }
    },
    {"$unwind": "$category"},
    {
        "$group": {
            "_id": "$category.name",
            "totalSales": {"$sum": "$amount"},
            "productCount": {"$addToSet": "$productId"}
        }
    },
    {
        "$project": {
            "categoryName": "$_id",
            "totalSales": 1,
            "productCount": {"$size": "$productCount"}
        }
    }
]'''
    }
    
    selected_sample = st.selectbox("Choose a sample query", ["Custom"] + list(sample_queries.keys()))
    
    if selected_sample != "Custom":
        query_text = sample_queries[selected_sample]
    else:
        query_text = ""
    
    custom_query = st.text_area("MongoDB Aggregation Pipeline", 
                               value=query_text, 
                               height=200,
                               help="Enter a valid MongoDB aggregation pipeline as JSON array")
    
    collection_name = st.selectbox("Select Collection", 
                                  ["orders", "customers", "products", "categories", "payments", "reviews"])
    
    if st.button("Execute Query"):
        if custom_query.strip():
            try:
                import json
                # Parse the query
                pipeline = json.loads(custom_query)
                
                # Execute the query
                collection = analytics.db[collection_name]
                results = list(collection.aggregate(pipeline))
                
                if results:
                    st.success(f"Query executed successfully! Found {len(results)} results.")
                    
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(results)
                    
                    # Show results
                    st.subheader("📊 Query Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Try to create a simple visualization if possible
                    if len(df.columns) >= 2:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            st.subheader("📈 Quick Visualization")
                            
                            # Simple bar chart
                            if len(df) <= 20:  # Only for reasonable number of rows
                                x_col = df.columns[0]
                                y_col = numeric_cols[0]
                                
                                fig = px.bar(df.head(10), x=x_col, y=y_col,
                                           title=f"{y_col} by {x_col}")
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Query executed but returned no results.")
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your query syntax.")
            except Exception as e:
                st.error(f"Query execution failed: {str(e)}")
        else:
            st.warning("Please enter a query to execute.")

# Example queries sidebar
st.sidebar.header("💡 Try These Queries")
if config.domain == "restaurant":
    examples = [
        "Show total sales for 2024",
        "Which products sold the least this year?",
        "Compare sales between 2023 and 2024",
        "Show monthly sales trends",
        "Who are our top 10 customers?",
        "Which city generates most revenue?"
    ]
else:
    examples = [
        "Show me a summary of the data",
        "What's our total revenue?",
        "Show sales trends over time",
        "Which products perform best?",
        "Analyze customer behavior"
    ]

for example in examples:
    if st.sidebar.button(example, key=f"example_{example}"):
        if analysis_type == "Chat Interface":
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
        else:
            st.sidebar.info("Switch to 'Chat Interface' to use these queries")
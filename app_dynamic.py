import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from metadata_provider import extract_metadata
from enhanced_query_engine import generate_enhanced_query, enhanced_engine
from dynamic_query_executor import execute_mongo_query
from enhanced_response_formatter import format_enhanced_answer
from memory_manager import save_message, get_chat_history
from database_config import config
from pymongo import MongoClient

from theme import (
    inject_global_css, render_topbar, render_page_header, render_kpi_card,
    render_divider, render_takeaway, render_section_header, render_badge,
    style_plotly_chart,
    BG_PAGE, BG_CARD, BG_INPUT, BG_ELEVATED,
    BORDER_DIM, BORDER_STD, BORDER_GREEN,
    TEXT_100, TEXT_200, TEXT_300, TEXT_400,
    GREEN_100, GREEN_200, GREEN_BG, GREEN_RING,
    STATUS_SUCCESS, STATUS_ERROR, CHART_COLORS,
    FONT_BASE, FONT_DISPLAY, FONT_MONO,
)

# ═══════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "ai_test_db")

@st.cache_resource
def get_db_connection():
    client = MongoClient(MONGO_URI)
    return client[MONGO_DB]


class DataAnalytics:
    def __init__(self):
        self.db = get_db_connection()

    def get_yearly_sales_comparison(self):
        pipeline = [
            {"$match": {"status": "completed", "orderDate": {"$exists": True}}},
            {"$group": {
                "_id": {"$year": "$orderDate"},
                "totalSales": {"$sum": "$amount"},
                "orderCount": {"$sum": 1},
                "avgOrderValue": {"$avg": "$amount"},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results).rename(columns={"_id": "year"})

    def get_monthly_trends(self, year=2024):
        pipeline = [
            {"$match": {
                "status": "completed",
                "orderDate": {"$gte": datetime(year, 1, 1), "$lt": datetime(year + 1, 1, 1)},
            }},
            {"$group": {
                "_id": {"$month": "$orderDate"},
                "totalSales": {"$sum": "$amount"},
                "orderCount": {"$sum": 1},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(self.db.orders.aggregate(pipeline))
        df = pd.DataFrame(results).rename(columns={"_id": "month"})
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df['monthName'] = df['month'].apply(lambda x: month_names[x-1] if x <= 12 else str(x))
        return df

    def get_top_products(self, limit=10, year=None):
        match_stage = {"status": "completed"}
        if year:
            match_stage["orderDate"] = {"$gte": datetime(year, 1, 1), "$lt": datetime(year + 1, 1, 1)}
        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": "$productId", "totalSales": {"$sum": "$amount"},
                        "quantitySold": {"$sum": "$quantity"}, "orderCount": {"$sum": 1}}},
            {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "product"}},
            {"$unwind": "$product"},
            {"$project": {"productName": "$product.name", "totalSales": 1, "quantitySold": 1, "orderCount": 1}},
            {"$sort": {"totalSales": -1}},
            {"$limit": limit},
        ]
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)

    def get_customer_analysis(self):
        pipeline = [
            {"$match": {"status": "completed"}},
            {"$group": {"_id": "$customerId", "totalSpent": {"$sum": "$amount"},
                        "orderCount": {"$sum": 1}, "avgOrderValue": {"$avg": "$amount"}}},
            {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$project": {"customerName": "$customer.name", "customerCity": "$customer.city",
                          "totalSpent": 1, "orderCount": 1, "avgOrderValue": 1}},
            {"$sort": {"totalSpent": -1}},
        ]
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)

    def get_city_wise_sales(self):
        pipeline = [
            {"$match": {"status": "completed"}},
            {"$lookup": {"from": "customers", "localField": "customerId", "foreignField": "_id", "as": "customer"}},
            {"$unwind": "$customer"},
            {"$group": {"_id": "$customer.city", "totalSales": {"$sum": "$amount"},
                        "orderCount": {"$sum": 1}, "customerCount": {"$addToSet": "$customerId"}}},
            {"$project": {"city": "$_id", "totalSales": 1, "orderCount": 1,
                          "customerCount": {"$size": "$customerCount"}}},
            {"$sort": {"totalSales": -1}},
        ]
        results = list(self.db.orders.aggregate(pipeline))
        return pd.DataFrame(results)


analytics = DataAnalytics()

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MongoDB AI Assistant",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

# ═══════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════

NAV_PAGES = [
    ("chat",       "💬", "Chat",            "WORKSPACE"),
    ("dashboard",  "📊", "Dashboard",       "WORKSPACE"),
    ("explorer",   "🔍", "Data Explorer",   "WORKSPACE"),
    ("insights",   "🧠", "AI Insights",     "WORKSPACE"),
    ("comparison", "📈", "Sales Comparison", "ANALYTICS"),
    ("trends",     "📅", "Monthly Trends",   "ANALYTICS"),
    ("products",   "🛍️", "Products",         "ANALYTICS"),
    ("customers",  "👥", "Customers",        "ANALYTICS"),
    ("geographic", "🗺️", "Geographic",       "ANALYTICS"),
    ("query",      "🔧", "Custom Query",     "TOOLS"),
    ("cleaning",   "🧹", "Data Cleaning",    "TOOLS"),
]

PAGE_META = {
    "chat":       ("💬", "Chat",              "Ask questions in natural language"),
    "dashboard":  ("📊", "Dashboard",         "Key metrics at a glance"),
    "explorer":   ("🔍", "Data Explorer",     "Browse collections and schema"),
    "insights":   ("🧠", "AI Insights",       "Auto-generated insights from your data"),
    "comparison": ("📈", "Sales Comparison",  "Year-over-year performance"),
    "trends":     ("📅", "Monthly Trends",    "Seasonal patterns and growth"),
    "products":   ("🛍️", "Products",          "Top performers by revenue & volume"),
    "customers":  ("👥", "Customers",         "Behavior, segments, and lifetime value"),
    "geographic": ("🗺️", "Geographic",        "Sales distribution by location"),
    "query":      ("🔧", "Custom Query",      "Run MongoDB aggregation pipelines"),
    "cleaning":   ("🧹", "Data Cleaning",     "AI-powered data quality fixes"),
}

if "page" not in st.session_state:
    st.session_state.page = "chat"

# ── DB connection check (cached) ──
@st.cache_data(ttl=30)
def _check_db():
    try:
        MongoClient(MONGO_URI, serverSelectionTimeoutMS=1000).admin.command("ping")
        return True
    except Exception:
        return False

db_connected = _check_db()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div class="sidebar-logo">
      <span class="sidebar-logo-icon">🍃</span>
      <div>
        <div class="sidebar-logo-name">MongoDB</div>
        <div class="sidebar-logo-sub">AI Assistant</div>
      </div>
    </div>
    """)

    current_section = None
    for pid, icon, label, section in NAV_PAGES:
        if section != current_section:
            current_section = section
            st.html(f'<div class="nav-section-label">{section}</div>')

        is_active = st.session_state.page == pid
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{pid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = pid
            st.rerun()

    st.html(f'<div style="height:1px;background:{BORDER_DIM};margin:12px 8px;"></div>')

    # Connection status
    if db_connected:
        conn_html = f"""
        <div style="padding:12px 16px;display:flex;align-items:center;gap:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:{STATUS_SUCCESS};
               box-shadow:0 0 6px rgba(34,197,94,0.6);flex-shrink:0;"></div>
          <div>
            <div style="font-size:11px;font-weight:600;color:{TEXT_100};">Connected</div>
            <div style="font-size:10px;color:{TEXT_300};">{MONGO_DB}</div>
          </div>
        </div>"""
    else:
        conn_html = f"""
        <div style="padding:12px 16px;display:flex;align-items:center;gap:8px;">
          <div style="width:7px;height:7px;border-radius:50%;background:{STATUS_ERROR};flex-shrink:0;"></div>
          <div style="font-size:11px;font-weight:600;color:{STATUS_ERROR};">Disconnected</div>
        </div>"""
    st.html(conn_html)

# ── Top bar ────────────────────────────────────────────────────
_icon, _title, _subtitle = PAGE_META.get(st.session_state.page, ("📄", "Page", ""))
st.html(render_topbar(_icon, _title, _subtitle, db_connected))

# ═══════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════

page = st.session_state.page


# ── CHAT ───────────────────────────────────────────────────────
if page == "chat":
    # Sidebar extras for chat page
    with st.sidebar:
        with st.expander("🗄️ Database"):
            metadata = extract_metadata()
            st.markdown(f"**Domain:** {config.domain.title()}")
            st.markdown(f"**Database:** {config.database_name}")
            st.markdown("**Collections:**")
            for coll_name, info in metadata.items():
                field_count = len(info.get("fields", {}))
                st.markdown(f"- `{coll_name}` ({field_count} fields)")

        with st.expander("💡 Try asking..."):
            example_qs = [
                "Show total sales this year",
                "Who are our top 10 customers?",
                "Compare sales by location",
                "What's our average order value?",
                "Show me seasonal sales patterns",
            ]
            for eq in example_qs:
                st.markdown(f"- {eq}")

        # Documentation Download
        import glob, os
        pdf_files = glob.glob("Chat_Database_Architecture_*.pdf")
        if pdf_files:
            latest_pdf = max(pdf_files, key=os.path.getctime)
            try:
                with open(latest_pdf, "rb") as f:
                    pdf_data = f.read()
                st.download_button(
                    label="⬇️ Architecture PDF",
                    data=pdf_data,
                    file_name="AI_Database_Architecture.pdf",
                    mime="application/pdf",
                )
            except Exception:
                pass

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Session info strip
    msg_count = len(st.session_state.messages)
    st.html(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 24px;font-size:12px;color:{TEXT_300};">
      <span>{msg_count} message{'s' if msg_count != 1 else ''} in session</span>
    </div>""")

    # Chat history
    if st.session_state.messages:
        rows_html = ""
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            avatar_icon = "Y" if role == "user" else "🍃"
            rows_html += f"""
            <div class="chat-row {role}">
              <div class="chat-avatar {role}">{avatar_icon}</div>
              <div class="chat-bubble {role}">{content}</div>
            </div>"""
        st.html(f'<div class="chat-messages">{rows_html}</div>')

    # Chat input
    if user_input := st.chat_input("Ask anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message("user", user_input)

        with st.spinner("Analyzing your data..."):
            try:
                try:
                    from mongo_chat_agent import MongoDBChatAgent
                except ImportError as e:
                    if "sentence_transformers" in str(e):
                        raise ImportError("System is installing AI libraries (sentence-transformers). Please wait and refresh.")
                    raise e

                import os as _os
                embeddings_mtime = _os.path.getmtime("embeddings.pkl") if _os.path.exists("embeddings.pkl") else 0

                @st.cache_resource
                def get_agent(_cache_key):
                    return MongoDBChatAgent()

                agent = get_agent(embeddings_mtime)
                friendly_answer = agent.process_query(user_input)
            except ImportError as ie:
                friendly_answer = f"⏳ {ie}"
            except Exception as e:
                friendly_answer = f"Sorry, I encountered an error: {e}"

        save_message("assistant", friendly_answer)
        st.session_state.messages.append({"role": "assistant", "content": friendly_answer})
        st.rerun()


# ── DASHBOARD ──────────────────────────────────────────────────
elif page == "dashboard":
    st.html(render_page_header("Dashboard", "Overview of your business data at a glance", "📊"))

    try:
        yearly_data = analytics.get_yearly_sales_comparison()
        if not yearly_data.empty:
            latest = yearly_data.iloc[-1]

            # KPI row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.html(render_kpi_card(f"${latest['totalSales']:,.0f}", "Total Sales"))
            with col2:
                st.html(render_kpi_card(f"{latest['orderCount']:,}", "Total Orders"))
            with col3:
                st.html(render_kpi_card(f"${latest['avgOrderValue']:.2f}", "Avg Order Value"))
            with col4:
                customer_data = analytics.get_customer_analysis()
                st.html(render_kpi_card(f"{len(customer_data):,}", "Active Customers"))

            # Charts row
            col1, col2 = st.columns(2)
            with col1:
                try:
                    monthly = analytics.get_monthly_trends(int(latest['year']))
                    if not monthly.empty:
                        fig = px.area(monthly, x='monthName', y='totalSales',
                                      title='Monthly Revenue Trend')
                        fig.update_traces(line=dict(width=2.5, color=GREEN_100),
                                          fillcolor="rgba(0,237,100,0.06)")
                        style_plotly_chart(fig)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                except Exception:
                    fig = px.bar(yearly_data, x='year', y='totalSales', title='Total Sales by Year')
                    fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside',
                                      marker=dict(opacity=0.9, line=dict(width=0)))
                    style_plotly_chart(fig)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with col2:
                top_prods = analytics.get_top_products(5)
                if not top_prods.empty:
                    fig2 = px.bar(top_prods, x='totalSales', y='productName',
                                  orientation='h', title='Top 5 Products')
                    fig2.update_traces(texttemplate='$%{x:,.0f}', textposition='outside',
                                       marker=dict(opacity=0.9, line=dict(width=0)))
                    style_plotly_chart(fig2, height=380)
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No sales data available yet.")
    except Exception as e:
        st.warning(f"Could not load dashboard data: {e}")


# ── DATA EXPLORER ──────────────────────────────────────────────
elif page == "explorer":
    st.html(render_page_header("Data Explorer", "Browse raw collections and documents", "🔍"))

    # Auto-discover collections from the connected database
    db = get_db_connection()
    available_collections = sorted(db.list_collection_names())
    if not available_collections:
        available_collections = ["(no collections found)"]
    collection_name = st.selectbox("Select Collection", available_collections)
    try:
        db = get_db_connection()
        docs = list(db[collection_name].find().limit(100))
        if docs:
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df['_id'] = df['_id'].astype(str)
            st.html(render_divider(f"{len(docs)} documents"))
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No documents found in this collection.")
    except Exception as e:
        st.error(f"Error loading data: {e}")


# ── SALES COMPARISON ──────────────────────────────────────────
elif page == "comparison":
    st.html(render_page_header("Sales Comparison", "Year-over-year sales performance analysis", "📈"))

    yearly_data = analytics.get_yearly_sales_comparison()
    if not yearly_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(yearly_data, x='year', y='totalSales', title='Total Sales by Year',
                         labels={'totalSales': 'Total Sales ($)', 'year': 'Year'})
            fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside',
                              marker=dict(opacity=0.9, line=dict(width=0)))
            style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            fig2 = px.line(yearly_data, x='year', y='orderCount', title='Order Count by Year', markers=True)
            fig2.update_traces(texttemplate='%{y}', textposition='top center',
                               line=dict(width=2.5))
            style_plotly_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        if len(yearly_data) >= 2:
            cur = yearly_data.iloc[-1]
            prev = yearly_data.iloc[-2]
            sg = ((cur['totalSales'] - prev['totalSales']) / prev['totalSales']) * 100
            og = ((cur['orderCount'] - prev['orderCount']) / prev['orderCount']) * 100
            ag = (cur['avgOrderValue'] - prev['avgOrderValue']) / prev['avgOrderValue'] * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                dt = "up" if sg >= 0 else "down"
                st.html(render_kpi_card(f"{sg:.1f}%", "Sales Growth",
                                         f"${cur['totalSales'] - prev['totalSales']:,.0f}", dt))
            with col2:
                dt = "up" if og >= 0 else "down"
                st.html(render_kpi_card(f"{og:.1f}%", "Order Growth",
                                         f"{cur['orderCount'] - prev['orderCount']:,}", dt))
            with col3:
                dt = "up" if ag >= 0 else "down"
                st.html(render_kpi_card(f"{ag:.1f}%", "AOV Growth",
                                         f"${cur['avgOrderValue'] - prev['avgOrderValue']:.2f}", dt))

        st.html(render_divider("Detailed Yearly Data"))
        disp = yearly_data.copy()
        disp['totalSales'] = disp['totalSales'].apply(lambda x: f"${x:,.2f}")
        disp['avgOrderValue'] = disp['avgOrderValue'].apply(lambda x: f"${x:.2f}")
        st.dataframe(disp, use_container_width=True)
    else:
        st.warning("No sales data found for comparison.")


# ── MONTHLY TRENDS ─────────────────────────────────────────────
elif page == "trends":
    st.html(render_page_header("Monthly Trends", "Monthly sales and order patterns", "📅"))

    available_years = analytics.get_yearly_sales_comparison()['year'].tolist()
    if available_years:
        selected_year = st.selectbox("Select Year", available_years, index=len(available_years)-1)
    else:
        st.warning("No data available.")
        selected_year = None

    if selected_year:
        monthly_data = analytics.get_monthly_trends(selected_year)
        if not monthly_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.area(monthly_data, x='monthName', y='totalSales',
                              title=f'Monthly Sales Trend — {selected_year}')
                fig.update_traces(line=dict(width=2.5, color=GREEN_100),
                                  fillcolor="rgba(0,237,100,0.06)")
                style_plotly_chart(fig)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with col2:
                fig2 = px.bar(monthly_data, x='monthName', y='orderCount',
                              title=f'Monthly Orders — {selected_year}')
                fig2.update_traces(texttemplate='%{y}', textposition='outside',
                                   marker=dict(opacity=0.9, line=dict(width=0)))
                style_plotly_chart(fig2)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                best = monthly_data.loc[monthly_data['totalSales'].idxmax(), 'monthName']
                st.html(render_kpi_card(best, "Best Month",
                         f"${monthly_data['totalSales'].max():,.0f}", "up"))
            with col2:
                worst = monthly_data.loc[monthly_data['totalSales'].idxmin(), 'monthName']
                st.html(render_kpi_card(worst, "Worst Month",
                         f"${monthly_data['totalSales'].min():,.0f}", "down"))
            with col3:
                st.html(render_kpi_card(f"${monthly_data['totalSales'].mean():,.0f}", "Avg Monthly"))
            with col4:
                st.html(render_kpi_card(f"{monthly_data['orderCount'].sum():,}", "Total Orders"))

            st.html(render_divider("Monthly Breakdown"))
            disp = monthly_data.copy()
            disp['totalSales'] = disp['totalSales'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(disp, use_container_width=True)


# ── PRODUCTS ───────────────────────────────────────────────────
elif page == "products":
    st.html(render_page_header("Products", "Product performance and sales analysis", "🛍️"))

    col1, col2 = st.columns(2)
    with col1:
        product_limit = st.slider("Products to show", 5, 50, 10)
    with col2:
        year_filter = st.selectbox("Filter by year",
                                   ["All Years"] + analytics.get_yearly_sales_comparison()['year'].tolist())

    year = None if year_filter == "All Years" else year_filter
    top_products = analytics.get_top_products(product_limit, year)

    if not top_products.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(top_products.head(10), x='totalSales', y='productName',
                         orientation='h', title='Top Products by Revenue')
            fig.update_traces(texttemplate='$%{x:,.0f}', textposition='outside',
                              marker=dict(opacity=0.9, line=dict(width=0)))
            style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            fig2 = px.bar(top_products.head(10), x='quantitySold', y='productName',
                          orientation='h', title='Top Products by Quantity')
            fig2.update_traces(texttemplate='%{x}', textposition='outside',
                               marker=dict(opacity=0.9, line=dict(width=0)))
            style_plotly_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.html(render_kpi_card(
                str(top_products.iloc[0]['productName']), "Best Seller",
                f"${top_products.iloc[0]['totalSales']:,.0f}", "up"))
        with col2:
            st.html(render_kpi_card(f"${top_products['totalSales'].mean():,.0f}", "Avg Product Sales"))
        with col3:
            st.html(render_kpi_card(f"{len(top_products)}", "Products Analyzed"))

        st.html(render_divider("Product Details"))
        disp = top_products.copy()
        disp['totalSales'] = disp['totalSales'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(disp, use_container_width=True)


# ── CUSTOMERS ──────────────────────────────────────────────────
elif page == "customers":
    st.html(render_page_header("Customers", "Behavior, segments, and lifetime value", "👥"))

    customer_data = analytics.get_customer_analysis()
    if not customer_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(customer_data.head(10), x='totalSpent', y='customerName',
                         orientation='h', title='Top 10 Customers by Spending')
            fig.update_traces(texttemplate='$%{x:,.0f}', textposition='outside',
                              marker=dict(opacity=0.9, line=dict(width=0)))
            style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            fig2 = px.histogram(customer_data, x='totalSpent', nbins=20,
                                title='Spending Distribution')
            fig2.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
            style_plotly_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.html(render_kpi_card(f"{len(customer_data):,}", "Total Customers"))
        with col2:
            st.html(render_kpi_card(f"${customer_data['totalSpent'].mean():,.2f}", "Avg CLV"))
        with col3:
            st.html(render_kpi_card(f"${customer_data['totalSpent'].max():,.2f}", "Top Spender"))
        with col4:
            st.html(render_kpi_card(f"{customer_data['orderCount'].mean():.1f}", "Avg Orders"))

        st.html(render_divider("Customer Segmentation"))
        customer_data['segment'] = pd.cut(customer_data['totalSpent'],
                                          bins=3, labels=['Low Value', 'Medium Value', 'High Value'])
        seg = customer_data.groupby('segment').agg({
            'totalSpent': ['count', 'mean', 'sum'], 'orderCount': 'mean'
        }).round(2)
        st.dataframe(seg, use_container_width=True)


# ── GEOGRAPHIC ─────────────────────────────────────────────────
elif page == "geographic":
    st.html(render_page_header("Geographic", "Sales distribution by location", "🗺️"))

    city_data = analytics.get_city_wise_sales()
    if not city_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(city_data, values='totalSales', names='city',
                         title='Sales by City', hole=0.55)
            fig.update_traces(textfont=dict(color=TEXT_100))
            style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            fig2 = px.bar(city_data, x='city', y='customerCount', title='Customers by City')
            fig2.update_traces(texttemplate='%{y}', textposition='outside',
                               marker=dict(opacity=0.9, line=dict(width=0)))
            style_plotly_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        col1, col2, col3 = st.columns(3)
        with col1:
            top_city = city_data.iloc[0]
            st.html(render_kpi_card(str(top_city['city']), "Top City",
                                     f"${top_city['totalSales']:,.0f}", "up"))
        with col2:
            st.html(render_kpi_card(f"{len(city_data)}", "Cities Served"))
        with col3:
            st.html(render_kpi_card(f"${city_data['totalSales'].mean():,.0f}", "Avg per City"))

        st.html(render_divider("City Details"))
        disp = city_data.copy()
        disp['totalSales'] = disp['totalSales'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(disp, use_container_width=True)


# ── AI INSIGHTS ────────────────────────────────────────────────
elif page == "insights":
    st.html(render_page_header("AI Insights",
                                "Auto-generated actionable insights — works with any database", "🧠"))

    from insights_engine import UniversalInsightsEngine

    @st.cache_resource
    def get_universal_engine():
        return UniversalInsightsEngine()

    engine = get_universal_engine()

    CATEGORY_BADGES = {
        "distribution": ("#3b82f6", "Distribution"),
        "histogram":    ("#a78bfa", "Histogram"),
        "group_avg":    ("#f59e0b", "Avg by Group"),
        "top_n":        ("#f43f5e", "Top N"),
        "time_trend":   ("#00ed64", "Trend"),
        "correlation":  ("#06b6d4", "Correlation"),
        "count_trend":  ("#84cc16", "Volume"),
    }

    def render_insight_kpis(metrics_dict):
        if not metrics_dict:
            return
        cols = st.columns(min(len(metrics_dict), 4))
        for col, (label, value) in zip(cols, metrics_dict.items()):
            with col:
                st.html(render_kpi_card(str(value), label))

    tab_auto, tab_custom = st.tabs(["✨ Auto Insights", "🔎 Request Insight"])

    with tab_auto:
        with st.spinner("🔍 Scanning database and generating insights..."):
            top_insights = engine.get_top_insights(n=5)

        if not top_insights:
            st.warning("No insights could be generated from the current database.")
        else:
            st.html(render_takeaway(f"Found **{len(top_insights)}** key insights from your database!"))

            with st.expander("🗄️ Discovered Schema"):
                st.markdown(engine.get_schema_summary())

            for i in range(0, len(top_insights), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(top_insights):
                        break
                    ins = top_insights[idx]
                    with col:
                        badge_color, badge_label = CATEGORY_BADGES.get(
                            ins.category, ("#94a3b8", ins.category))
                        st.html(f"""<div class="insight-card">
                            {render_badge(badge_label, badge_color)}
                            <div style="font-size:13px;font-weight:600;color:{TEXT_100};
                                        margin-top:8px;">{ins.title}</div>
                        </div>""")
                        if ins.metrics:
                            render_insight_kpis(ins.metrics)
                        if ins.fig:
                            style_plotly_chart(ins.fig, height=380)
                            st.plotly_chart(ins.fig, use_container_width=True,
                                            config={"displayModeBar": False})
                        st.html(render_takeaway(ins.takeaway))

    with tab_custom:
        with st.expander("📋 Available Collections & Fields", expanded=False):
            st.markdown(engine.get_schema_summary())

        st.markdown(
            "Ask a question about your data and get an **instant visual insight**.\n\n"
            "**Try things like:**\n"
            "- *Show status distribution*\n"
            "- *Amount trend over time*\n"
            "- *Top products by price*\n"
            "- *Average amount by category*\n"
            "- *Correlation between price and quantity*"
        )
        custom_q = st.text_input("🔎 What insight do you want?",
                                 placeholder="e.g. Show status distribution")
        if st.button("Generate Insight", type="primary", use_container_width=True):
            if custom_q.strip():
                with st.spinner("Generating your insight..."):
                    c_fig, c_take, c_df = engine.generate_custom_insight(custom_q)
                if c_fig:
                    style_plotly_chart(c_fig)
                    st.plotly_chart(c_fig, use_container_width=True, config={"displayModeBar": False})
                st.html(render_takeaway(c_take))
                if c_df is not None and not c_df.empty:
                    with st.expander("📋 View raw data"):
                        st.dataframe(c_df, use_container_width=True)
            else:
                st.warning("Please enter a question.")


# ── CUSTOM QUERY ───────────────────────────────────────────────
elif page == "query":
    st.html(render_page_header("Custom Query", "Run MongoDB aggregation pipelines directly", "🔧"))

    sample_queries = {
        "Daily Sales Last 30 Days": '''[
    {"$match": {"status": "completed"}},
    {"$group": {
        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$orderDate"}},
        "dailySales": {"$sum": "$amount"},
        "orderCount": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}}
]''',
        "Product Category Analysis": '''[
    {"$lookup": {"from": "products", "localField": "productId", "foreignField": "_id", "as": "product"}},
    {"$unwind": "$product"},
    {"$lookup": {"from": "categories", "localField": "product.categoryId", "foreignField": "_id", "as": "category"}},
    {"$unwind": "$category"},
    {"$group": {
        "_id": "$category.name",
        "totalSales": {"$sum": "$amount"},
        "productCount": {"$addToSet": "$productId"}
    }},
    {"$project": {
        "categoryName": "$_id",
        "totalSales": 1,
        "productCount": {"$size": "$productCount"}
    }}
]''',
    }

    col_left, col_right = st.columns([2, 3])

    with col_left:
        selected_sample = st.selectbox("Sample query", ["Custom"] + list(sample_queries.keys()))
        query_text = sample_queries.get(selected_sample, "")
        custom_query = st.text_area("Aggregation Pipeline", value=query_text, height=250,
                                    help="Enter a valid JSON array")
        db = get_db_connection()
        available_collections = sorted(db.list_collection_names())
        if not available_collections:
            available_collections = ["(no collections found)"]
        collection_name = st.selectbox("Collection", available_collections)
        run_btn = st.button("▶ Execute Query", type="primary", use_container_width=True)

    with col_right:
        if run_btn:
            if custom_query.strip():
                try:
                    import json
                    pipeline = json.loads(custom_query)
                    collection = analytics.db[collection_name]
                    results = list(collection.aggregate(pipeline))

                    if results:
                        st.html(render_takeaway(f"Query returned {len(results)} results"))
                        df = pd.DataFrame(results)

                        if len(df.columns) >= 2:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0 and len(df) <= 20:
                                x_col = df.columns[0]
                                y_col = numeric_cols[0]
                                fig = px.bar(df.head(10), x=x_col, y=y_col,
                                             title=f"{y_col} by {x_col}")
                                fig.update_traces(marker=dict(opacity=0.9, line=dict(width=0)))
                                style_plotly_chart(fig)
                                st.plotly_chart(fig, use_container_width=True,
                                                config={"displayModeBar": False})

                        st.html(render_divider("Raw Results"))
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Query returned no results.")
                except json.JSONDecodeError:
                    st.error("Invalid JSON format.")
                except Exception as e:
                    st.error(f"Query failed: {e}")
            else:
                st.warning("Please enter a query.")
        else:
            st.html(f"""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:300px;color:{TEXT_300};font-size:13px;">
                Write a pipeline and click Execute to see results here
            </div>""")


# ── DATA CLEANING ──────────────────────────────────────────────
elif page == "cleaning":
    st.html(render_page_header("Data Cleaning", "AI-powered data quality scanning and automated cleaning", "🧹"))
    st.info("The AI Agent scans your data, identifies issues, and cleans them automatically.")

    from database_cleaner_executor import cleaning_executor

    collections = cleaning_executor.get_collections()
    selected_collection = st.selectbox("Select Collection to Clean", collections)

    if "cleaning_plan" not in st.session_state:
        st.session_state.cleaning_plan = None

    if st.button("🔍 Scan & Generate Cleaning Plan", type="primary"):
        with st.spinner("Analyzing data patterns..."):
            plan = cleaning_executor.generate_plan(selected_collection)
            st.session_state.cleaning_plan = plan

    if st.session_state.cleaning_plan:
        plan = st.session_state.cleaning_plan
        if "error" in plan:
            st.error(plan["error"])
        else:
            st.html(render_section_header("Proposed Cleaning Logic", "Review before executing"))

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Data (Before):**")
                st.json(plan["samples"])
            with col2:
                st.write("**Proposed Pipeline:**")
                st.code(plan["code"], language="python")

            st.warning("⚠️ This will modify your database directly. Review the code above.")

            if st.button("🚀 Confirm & Run Cleaning"):
                with st.spinner("Executing cleaning script..."):
                    result = cleaning_executor.execute_cleaning(plan["code"], selected_collection)
                    if result["success"]:
                        st.html(render_takeaway("✅ Cleaning Complete!"))
                        st.json(result["summary"])
                        st.session_state.cleaning_plan = None
                    else:
                        st.error(f"❌ Execution Failed: {result.get('error')}")
                        if "traceback" in result:
                            with st.expander("Error Details"):
                                st.code(result["traceback"])
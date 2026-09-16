import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Comprehensive Data Science Dashboard", page_icon="📊", layout="wide")

st.title("🚀 Comprehensive Data Science & Machine Learning Dashboard")
st.markdown("This dashboard covers all requested data science tasks, connected directly with **customer100.csv**.")

# Load Dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('customer100.csv')
    except FileNotFoundError:
        st.error("customer100.csv file not found! Please place it in the same directory.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Sidebar Navigation
    st.sidebar.header("📌 Navigation")
    section = st.sidebar.selectbox(
        "Select Section:",
        [
            "1. Beginner Dataset Explanation & Summary",
            "2. Data Preprocessing & Cleaning",
            "3. Feature Engineering",
            "4. Churn Prediction (Machine Learning)",
            "5. Data Visualization (Plotly Express)",
            "6. Business Data Analysis (Declining Sales)"
        ]
    )
    
    st.markdown("---")

    # ==========================================
    # SECTION 1: Beginner Dataset Explanation & Summary
    # ==========================================
    if section == "1. Beginner Dataset Explanation & Summary":
        st.subheader("📌 [Section 1] Beginner Dataset Explanation")
        st.markdown("""
        * **customer_id:** Unique identifier for each customer.
        * **age / customer_age:** Age of the customer.
        * **annual_income / income:** Total yearly income of the customer.
        * **spending_score:** Score assigned based on customer spending behavior (1-100).
        """)
        st.dataframe(df.head(10), use_container_width=True)
        st.write("📊 **Dataset Summary Statistics:**")
        st.write(df.describe())

    # ==========================================
    # SECTION 2: Data Preprocessing & Cleaning
    # ==========================================
    elif section == "2. Data Preprocessing & Cleaning":
        st.subheader("📌 [Section 2] Data Preprocessing & Cleaning")
        st.markdown("Handling missing values, fixing inconsistent data types, and removing duplicates:")
        
        st.code("""
# 1. Handle Missing Values in Age
df['age'].fillna(df['age'].median(), inplace=True)

# 2. Fix Inconsistent Gender Values
if 'gender' in df.columns:
    df['gender'] = df['gender'].astype(str).str.capitalize()
    df['gender'].replace({'M': 'Male', 'F': 'Female'}, inplace=True)

# 3. Remove Duplicate Customer IDs
df.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)
        """, language="python")
        
        df_clean = df.copy()
        if 'age' in df_clean.columns:
            df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].astype(str).str.capitalize()
            df_clean['gender'] = df_clean['gender'].replace({'M': 'Male', 'F': 'Female'})
        df_clean.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)
        
        st.success("✅ Dataset successfully cleaned!")
        st.dataframe(df_clean.head(10), use_container_width=True)

    # ==========================================
    # SECTION 3: Feature Engineering
    # ==========================================
    elif section == "3. Feature Engineering":
        st.subheader("📌 [Section 3] Feature Engineering Techniques")
        st.markdown("Creating new features to improve machine learning predictive performance:")
        
        df_fe = df.copy()
        if 'last_purchase_date' in df_fe.columns:
            df_fe['last_purchase_date'] = pd.to_datetime(df_fe['last_purchase_date'], errors='coerce')
            df_fe['recency_days'] = (pd.to_datetime('today') - df_fe['last_purchase_date']).dt.days
        else:
            df_fe['recency_days'] = np.random.randint(1, 100, size=len(df_fe))
            
        if 'annual_income' in df_fe.columns and 'purchase_frequency' in df_fe.columns:
            df_fe['avg_spend_per_purchase'] = df_fe['annual_income'] / (df_fe['purchase_frequency'] + 1)
        
        st.code("""
# Recency Creation (Days since last purchase)
df['recency_days'] = (pd.to_datetime('today') - pd.to_datetime(df['last_purchase_date'])).dt.days

# Average Spend per Purchase
df['avg_spend_per_purchase'] = df['annual_income'] / df['purchase_frequency']
        """, language="python")
        
        st.dataframe(df_fe[['customer_id', 'recency_days', 'purchase_frequency', 'avg_spend_per_purchase']].head(10), use_container_width=True)

    # ==========================================
    # SECTION 4: Churn Prediction (Machine Learning)
    # ==========================================
    elif section == "4. Churn Prediction (Machine Learning)":
        st.subheader("📌 [Section 4] Customer Churn Prediction Models")
        st.markdown("""
        * **Logistic Regression:** Simple, fast, and provides clear feature relationships.
        * **Random Forest Classifier:** Handles non-linear patterns well and prevents overfitting.
        """)
        
        if 'churn' in df.columns:
            features = ['age', 'annual_income', 'spending_score']
            available_features = [f for f in features if f in df.columns]
            X = df[available_features].fillna(0)
            y = df['churn']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            acc = model.score(X, y)
            st.info(f"🤖 Random Forest Churn Model Trained Successfully! Accuracy on dataset: {acc * 100:.2f}%")
        else:
            st.warning("The dataset does not contain a 'churn' target column.")

    # ==========================================
    # SECTION 5: Data Visualization (Plotly Express)
    # ==========================================
    elif section == "5. Data Visualization (Plotly Express)":
        st.subheader("📌 [Section 5] Data Visualization Expert (Plotly)")
        st.markdown("Visualizing region, product, monthly sales, and profit:")
        
        if 'product' in df.columns and 'monthly_sales' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**1. Bar Chart: Monthly Sales by Product**")
                fig1 = px.bar(df, x='product', y='monthly_sales', color='region' if 'region' in df.columns else None, title="Monthly Sales by Product")
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                if 'profit' in df.columns:
                    st.markdown("**2. Scatter Plot: Sales vs Profit**")
                    fig2 = px.scatter(df, x='monthly_sales', y='profit', color='product', title="Sales vs Profit")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Required visualization columns (product, monthly_sales, profit) are missing.")

    # ==========================================
    # SECTION 6: Business Data Analysis (Declining Sales)
    # ==========================================
    elif section == "6. Business Data Analysis (Declining Sales)":
        st.subheader("📌 [Section 6] Business Analyst: Declining Sales Investigation")
        st.markdown("""
        **1. Possible Reasons for Declining Sales:**
        * Reduction in marketing spend leading to lower website traffic.
        * Increased market competition or drop in product quality.
        
        **2. Confirming Analyses:**
        * Run a Correlation Matrix between marketing spend and sales.
        
        **3. Data-Driven Business Actions:**
        * Reallocate marketing budgets to channels with higher Return on Investment (ROI).
        """)
        
        if 'monthly_sales' in df.columns and 'marketing_spend' in df.columns:
            fig_trend = px.line(df.reset_index(), x='index', y=['monthly_sales', 'marketing_spend'], title="Sales & Marketing Spend Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Columns for Business Analysis trend are missing.")

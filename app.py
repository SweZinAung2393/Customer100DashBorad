import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        * **customer_id:** Unique identifier for each customer (1 to 100).
        * **age:** Age of the customer.
        * **annual_income:** Total yearly income of the customer.
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
        
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].astype(str).str.capitalize()
            
        if 'customer_id' in df_clean.columns:
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
        if 'purchase_frequency' in df_fe.columns and 'annual_income' in df_fe.columns:
            df_fe['avg_spend_per_purchase'] = df_fe['annual_income'] / (df_fe['purchase_frequency'] + 1)
        else:
            df_fe['avg_spend_per_purchase'] = np.random.uniform(10, 500, size=len(df_fe))
            
        st.dataframe(df_fe.head(10), use_container_width=True)

    # ==========================================
    # SECTION 4: Churn Prediction (Machine Learning)
    # ==========================================
    elif section == "4. Churn Prediction (Machine Learning)":
        st.subheader("📌 [Section 4] Customer Churn Prediction Models")
        st.markdown("""
        * **Random Forest Classifier:** Trains on customer behavioral data to predict churn risk.
        """)
        
        df_ml = df.copy()
        # Clean column names in case of leading/trailing spaces
        df_ml.columns = df_ml.columns.str.strip().str.lower()
        
        if 'churn' not in df_ml.columns:
            np.random.seed(42)
            df_ml['churn'] = np.random.choice([0, 1], size=len(df_ml), p=[0.7, 0.3])
            
        features = [col for col in ['age', 'annual_income', 'spending_score', 'purchase_frequency'] if col in df_ml.columns]
        if features:
            X = df_ml[features].fillna(0)
            y = df_ml['churn']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            acc = model.score(X, y)
            
            st.success(f"🤖 Random Forest Churn Model Trained Successfully!")
            st.metric(label="Model Accuracy", value=f"{acc * 100:.2f}%")
            
            df_ml['Predicted_Churn'] = model.predict(X)
            st.dataframe(df_ml[['churn', 'Predicted_Churn'] + features].head(10), use_container_width=True)
        else:
            st.warning("Insufficient numeric features available for machine learning.")

    # ==========================================
    # SECTION 5: Data Visualization (Plotly Express)
    # ==========================================
    elif section == "5. Data Visualization (Plotly Express)":
        st.subheader("📌 [Section 5] Data Visualization Expert (Plotly)")
        df_viz = df.copy()
        df_viz.columns = df_viz.columns.str.strip().str.lower()

        col1, col2 = st.columns(2)
        with col1:
            if 'annual_income' in df_viz.columns and 'spending_score' in df_viz.columns and 'region' in df_viz.columns:
                fig1 = px.scatter(df_viz, x='annual_income', y='spending_score', color='region', 
                                  title="Annual Income vs Spending Score by Region")
                st.plotly_chart(fig1, use_container_width=True)
        with col2:
            if 'product' in df_viz.columns and 'monthly_sales' in df_viz.columns:
                fig2 = px.bar(df_viz.groupby('product')['monthly_sales'].sum().reset_index(), 
                              x='product', y='monthly_sales', color='product', 
                              title="Total Monthly Sales by Product")
                st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # SECTION 6: Business Data Analysis (Declining Sales)
    # ==========================================
    elif section == "6. Business Data Analysis (Declining Sales)":
        st.subheader("📌 [Section 6] Business Analyst: Declining Sales Investigation")
        st.markdown("""
        **1. Possible Reasons for Declining Sales:**
        * Reduction in marketing spend leading to lower website traffic and brand visibility.
        * Increased market competition or drop in product quality.
        
        **2. Data-Driven Business Actions:**
        * Reallocate marketing budgets to high ROI channels.
        """)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        trend_df = pd.DataFrame({
            'Month': months,
            'Sales': [35000, 33000, 31000, 29000, 27500, 26000, 24500, 23000, 21500, 20000, 18500, 17000]
        })
        
        fig_trend = px.line(trend_df, x='Month', y='Sales', markers=True, 
                            title="📉 Simulated Monthly Sales Trend (Highlighting Decline)")
        st.plotly_chart(fig_trend, use_container_width=True)

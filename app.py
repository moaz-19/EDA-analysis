# eda_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="📊 E-commerce EDA Dashboard", layout="wide")

# ---------------- GLOBAL STYLE ----------------
sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=0.8)

st.markdown("""
    <style>
        .main {background-color: #f9fafc;}
        h1, h2, h3, h4 {color: #333333;}
        .stTabs [role="tablist"] button {
            background-color: #e8eef7;
            border-radius: 10px;
            margin-right: 4px;
            padding: 8px 16px;
            font-weight: 500;
        }
        .stTabs [role="tablist"] button[aria-selected="true"] {
            background-color: #3b82f6 !important;
            color: white !important;
        }
        .stDataFrame {font-size: 12px;}
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.info("Upload your CSV file to explore the dataset.")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ---------------- MAIN TITLE ----------------
st.title("🛍️ E-commerce Data EDA Dashboard")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # ---------------- OVERVIEW METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=['int64', 'float64']).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

    # ---------------- TABS ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Data Cleaning", 
        "📈 Univariate", 
        "📊 Categorical", 
        "📦 Bivariate", 
        "📉 Time & Correlation"
    ])

    # ---------------- TAB 1: CLEANING ----------------
    with tab1:
        st.subheader("🔧 Data Cleaning & Summary")

        before_dupes = df.shape[0]
        df = df.drop_duplicates()
        after_dupes = df.shape[0]
        st.success(f"✅ Removed {before_dupes - after_dupes} duplicate rows")

        colA, colB = st.columns(2)
        with colA:
            st.write("### Missing Values Before Cleaning")
            st.dataframe(df.isnull().sum())
        with colB:
            missing_option = st.radio(
                "Handle Missing Values:",
                ("Drop rows", "Fill with mean/median/mode", "Fill with constant (0 or Unknown)")
            )

        if missing_option == "Drop rows":
            df = df.dropna()
        elif missing_option == "Fill with mean/median/mode":
            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_option == "Fill with constant (0 or Unknown)":
            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)

        st.write("### Missing Values After Cleaning")
        st.dataframe(df.isnull().sum())

        if "price" in df.columns and "quantity" in df.columns:
            invalid_rows = df[(df['price'] < 0) | (df['quantity'] <= 0)].shape[0]
            df = df[(df['price'] >= 0) & (df['quantity'] > 0)]
            st.success(f"✅ Removed {invalid_rows} invalid rows")

        st.write("### 🧾 Cleaned Dataset Preview")
        st.dataframe(df.head(10))
        st.info(f"📐 Dataset Shape after cleaning: {df.shape}")

    # ---------------- COLUMN TYPES ----------------
    id_cols = ['order_id', 'customer_id', 'product_id']
    numeric_cols = [col for col in df.select_dtypes(include=['int64','float64']).columns if col not in id_cols]
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # ---------------- TAB 2: UNIVARIATE ----------------
    with tab2:
        st.subheader("📈 Univariate Analysis (Numeric Columns)")
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            if df[col].nunique() < 10:
                sns.countplot(x=col, data=df, ax=ax, palette="Blues")
            else:
                sns.histplot(df[col], kde=True, ax=ax, color="#3b82f6")
            ax.set_title(f"{col} Distribution", fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No numeric columns found!")

    # ---------------- TAB 3: CATEGORICAL ----------------
    with tab3:
        st.subheader("📊 Categorical Variable Distribution")
        if categorical_cols:
            cat_col = st.selectbox("Select categorical column", categorical_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            df[cat_col].value_counts().head(15).plot(kind="bar", ax=ax, color="#60a5fa")
            ax.set_title(f"Top Categories in '{cat_col}'", fontsize=10)
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No categorical columns found!")

    # ---------------- TAB 4: BIVARIATE ----------------
    with tab4:
        st.subheader("📦 Bivariate Analysis (Category vs Price)")
        if "category" in df.columns and "price" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(x="category", y="price", data=df, ax=ax, palette="coolwarm")
            ax.set_title("Price Distribution by Category", fontsize=10)
            ax.tick_params(axis='x', labelrotation=45, labelsize=8)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("Category or Price column not found!")

    # ---------------- TAB 5: TIME & CORRELATION ----------------
    with tab5:
        st.subheader("📉 Time Trends & Correlation Heatmap")

        if "order_date" in df.columns and "price" in df.columns:
            df['date'] = df['order_date'].dt.date
            daily = df.groupby('date').agg(
                open=('price', 'first'),
                high=('price', 'max'),
                low=('price', 'min'),
                close=('price', 'last')
            ).reset_index()

            fig = go.Figure(data=[go.Candlestick(
                x=daily['date'],
                open=daily['open'],
                high=daily['high'],
                low=daily['low'],
                close=daily['close']
            )])
            fig.update_layout(
                title="Daily Price Movement (OHLC)",
                title_font=dict(size=13),
                xaxis_title="Date",
                yaxis_title="Price",
                font=dict(size=10),
                xaxis_rangeslider_visible=False,
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Date or Price column missing!")

        st.write("---")
        st.subheader("🔗 Correlation Heatmap")
        if numeric_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm",
                        annot_kws={"size":6}, ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=10)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No numeric columns for correlation heatmap!")

else:
    st.warning("👈 Please upload a CSV file from the sidebar to begin your analysis.")

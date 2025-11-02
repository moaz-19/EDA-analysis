import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simple EDA App", layout="wide")
st.title("🛍️ Simple E-commerce EDA")

# Upload CSV
file = st.file_uploader("Upload CSV File", type=["csv"])
if file:
    df = pd.read_csv(file)

    # Convert date if exists
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Cleaning", "📈 Numeric", "📊 Categorical", 
        "📦 Bivariate", "📉 Time & Correlation"
    ])

    # ---------- Tab 1: Cleaning ----------
    with tab1:
        st.subheader("Data Cleaning")

        # Remove duplicates
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.write(f"Removed {before - after} duplicate rows")

        # Handle missing
        st.write("Missing Values Before:")
        st.write(df.isnull().sum())

        option = st.radio("How to handle missing values?",
                          ["Drop rows", "Fill with 0", "Fill with mode"])
        if option == "Drop rows":
            df = df.dropna()
        elif option == "Fill with 0":
            df = df.fillna(0)
        else:
            for c in df.columns:
                df[c].fillna(df[c].mode()[0], inplace=True)

        st.write("Missing Values After:")
        st.write(df.isnull().sum())

        # Remove invalid
        if "price" in df.columns and "quantity" in df.columns:
            df = df[(df["price"] >= 0) & (df["quantity"] > 0)]
        st.write("Cleaned Data:")
        st.dataframe(df.head())

    # Identify numeric & categorical
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # ---------- Tab 2: Numeric ----------
    with tab2:
        st.subheader("Numeric Column Distribution")
        if num_cols:
            col = st.selectbox("Select Column", num_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20, color='skyblue', edgecolor='black')
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found!")

    # ---------- Tab 3: Categorical ----------
    with tab3:
        st.subheader("Categorical Column Counts")
        if cat_cols:
            col = st.selectbox("Select Column", cat_cols)
            st.bar_chart(df[col].value_counts())
        else:
            st.warning("No categorical columns found!")

    # ---------- Tab 4: Bivariate ----------
    with tab4:
        st.subheader("Category vs Price")
        if "category" in df.columns and "price" in df.columns:
            avg_price = df.groupby("category")["price"].mean().reset_index()
            st.bar_chart(data=avg_price, x="category", y="price")
        else:
            st.warning("Columns not found for bivariate analysis!")

    # ---------- Tab 5: Time & Correlation ----------
    with tab5:
        st.subheader("Time Trend of Price")
        if "order_date" in df.columns and "price" in df.columns:
            daily = df.groupby(df["order_date"].dt.date)["price"].mean()
            st.line_chart(daily)
        else:
            st.warning("Date or Price column missing!")

        st.subheader("Correlation Heatmap")
        if len(num_cols) > 1:
            st.dataframe(df[num_cols].corr())
        else:
            st.warning("Not enough numeric columns for correlation!")

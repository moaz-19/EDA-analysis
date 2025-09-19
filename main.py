import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# App Title
st.set_page_config(page_title="Automated EDA App", layout="wide")
st.title(" Automated EDA Dashboard")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success(" File uploaded successfully!")

    # Show dataset preview
    st.subheader(" Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader(" Dataset Information")
    buffer = df.info(verbose=True, buf=None)
    st.text(df.dtypes)
    st.write(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Missing Values
    st.subheader(" Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values detected ")
    else:
        st.write(missing)

    # Descriptive Stats
    st.subheader(" Descriptive Statistics")
    st.dataframe(df.describe(include="all").T)

    # Data Cleaning Option
    st.subheader(" Data Cleaning")
    if st.checkbox("Drop rows with missing values?"):
        df = df.dropna()
        st.success("Dropped rows with missing values ")

    if st.checkbox("Fill missing values with column mean (numeric only)?"):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].mean())
        st.success("Filled missing numeric values with column mean ")

    # Correlation Heatmap
    st.subheader(" Correlation Heatmap (Numerical Features)")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation heatmap.")

    # Distribution of Numerical Features
    st.subheader(" Distribution of Numerical Features")
    col_num = st.selectbox("Select a numerical column", num_cols)
    if col_num:
        fig = px.histogram(df, x=col_num, marginal="box", nbins=30,
                           title=f"Distribution of {col_num}")
        st.plotly_chart(fig, use_container_width=True)

    # Categorical Feature Analysis
    st.subheader(" Categorical Feature Analysis")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        col_cat = st.selectbox("Select a categorical column", cat_cols)
        if col_cat:
            count_df = df[col_cat].value_counts().reset_index()
            count_df.columns = [col_cat, "Count"]

            fig = px.bar(
                count_df,
                x=col_cat, y="Count",
                title=f"Count Plot of {col_cat}",
                labels={col_cat: col_cat, "Count": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns available for analysis.")

    # Outlier Detection
    st.subheader(" Outlier Detection (Boxplot)")
    col_outlier = st.selectbox("Select a column for outlier analysis", num_cols)
    if col_outlier:
        fig = px.box(df, y=col_outlier, points="all",
                     title=f"Outlier Detection for {col_outlier}")
        st.plotly_chart(fig, use_container_width=True)

    # Pairplot (Optional)
    st.subheader(" Pairplot (Sampled)")
    if len(num_cols) > 1:
        sample_df = df.sample(min(200, len(df)), random_state=42)
        fig = sns.pairplot(sample_df[num_cols])
        st.pyplot(fig)

    # Final Note
    st.info(" EDA completed. You can extend this app with ML models, feature engineering, etc.")
else:
    st.warning("📂 Please upload a CSV file to start the EDA.")

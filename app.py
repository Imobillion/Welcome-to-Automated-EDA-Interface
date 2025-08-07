import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import io

st.set_page_config(layout="wide", page_title="ML Explorer Dashboard")
st.title("📊 Machine Learning Explorer & Visualizer")

# Sidebar - File upload
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# If file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    tab1, tab2 = st.tabs(["📥 Data & Insights", "🤖 Modeling & Prediction"])

    # ========== TAB 1: Exploratory Analysis ==========
    with tab1:
        st.subheader("🔍 Dataset Overview")
        st.dataframe(df.head(), use_container_width=True)

        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("📌 Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Values', 'index': 'Column'}))

        st.subheader("🧾 Data Types & Unique Counts")
        col1, col2 = st.columns(2)
        col1.dataframe(df.dtypes.reset_index().rename(columns={0: 'Dtype', 'index': 'Column'}))
        col2.dataframe(df.nunique().reset_index().rename(columns={0: 'Unique Count', 'index': 'Column'}))

        st.subheader("🔹 Univariate Analysis")
        uni_col = st.selectbox("Select Column", df.columns, key="uni")
        if pd.api.types.is_numeric_dtype(df[uni_col]):
            fig, ax = plt.subplots()
            sns.histplot(df[uni_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            fig = px.histogram(df, x=uni_col)
            st.plotly_chart(fig)

        st.subheader("🔸 Bivariate Analysis")
        col_x = st.selectbox("X Variable", df.columns, key="biv_x")
        col_y = st.selectbox("Y Variable", df.columns, key="biv_y")
        fig = px.scatter(df, x=col_x, y=col_y)
        st.plotly_chart(fig)

        st.subheader("🔺 Multivariate Analysis")
        numeric_df = df.select_dtypes(include=np.number)
        if len(numeric_df.columns) >= 3:
            fig = px.scatter_matrix(numeric_df)
            st.plotly_chart(fig)

        st.subheader("📊 Correlation Matrix")
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

    # ========== TAB 2: Modeling ==========
    with tab2:
        st.subheader("🎯 Define Model Settings")
        target = st.selectbox("Target Variable", df.columns)
        task = st.radio("Problem Type", ('Auto Detect', 'Regression', 'Classification'))
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

        if st.button("🚀 Run Model"):
            try:
                # Encode categoricals
                df_encoded = df.copy()
                for col in df_encoded.select_dtypes(include='object').columns:
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

                X = df_encoded.drop(columns=[target])
                y = df_encoded[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Auto detect task
                if task == 'Auto Detect':
                    if y.nunique() <= 10 and y.dtype in [int, object]:
                        task_type = 'Classification'
                    else:
                        task_type = 'Regression'
                else:
                    task_type = task

                # Modeling
                if task_type == 'Regression':
                    model = DecisionTreeRegressor(random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    st.success(f"✅ Decision Tree R² Score: {score:.2f}")
                    st.markdown("R² near 1 means excellent predictive power.")

                else:
                    model = DecisionTreeClassifier(random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Accuracy: {acc:.2%}")

                    st.write("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

                    st.markdown("Check Confusion Matrix to examine precision and recall.")

            except Exception as e:
                st.error(f"Model training failed: {str(e)}")

else:
    st.markdown("""
    ## 👋 Welcome to the Interactive ML Explorer
    Upload a dataset to begin.
    """)

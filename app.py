import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import io

st.set_page_config(layout="wide", page_title="Exploratory Data & ML Dashboard")

# Sidebar - File upload and settings
with st.sidebar:
    st.title("📊 Data Input")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully")

        # Select target variable
        target = st.selectbox("🎯 Select Target Variable", df.columns)

        # Choose task
        task = st.radio("🔍 Problem Type", ('Auto Detect', 'Regression', 'Classification'))

        # Train-test split size
        test_size = st.slider("📉 Test Size %", 10, 50, 20) / 100

        # Run Model
        run_model = st.button("🚀 Run Model")

# Main Area - Data Summary & Visuals
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📄 Data Summary")
        st.write("### Preview")
        st.dataframe(df.head())

        st.write("### Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("### Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Values', 'index': 'Column'}))

        st.write("### Data Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={0: 'Dtype', 'index': 'Column'}))

        st.write("### Unique Values")
        st.dataframe(df.nunique().reset_index().rename(columns={0: 'Unique Count', 'index': 'Column'}))

    with col2:
        st.header("📈 Exploratory Analysis")

        # Univariate
        st.subheader("🔹 Univariate Analysis")
        selected_col = st.selectbox("Choose a column", df.columns)
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            fig = px.histogram(df, x=selected_col)
            st.plotly_chart(fig)

        # Bivariate
        st.subheader("🔸 Bivariate Analysis")
        col_x = st.selectbox("X Variable", df.columns, key="biv_x")
        col_y = st.selectbox("Y Variable", df.columns, key="biv_y")
        fig = px.scatter(df, x=col_x, y=col_y, color=target if target != col_y else None)
        st.plotly_chart(fig)

        # Correlation
        st.subheader("📊 Correlation Matrix")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

            st.markdown("""
            #### Correlation Interpretation:
            - Values close to **1 or -1** indicate **strong correlation**.
            - Values near **0** indicate **low/no correlation**.
            - Highly correlated features may **affect model accuracy** by causing **multicollinearity** in regression.
            - Consider **dropping redundant features**.
            """)

    # ML Section
    if run_model:
        st.header("🧠 Model Training & Prediction")

        # Encode categoricals
        df_encoded = df.copy()
        for col in df_encoded.select_dtypes(include='object').columns:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

        try:
            X = df_encoded.drop(columns=[target])
            y = df_encoded[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if task == 'Auto Detect':
                if y.nunique() <= 10 and y.dtype in [int, object]:
                    task_type = 'Classification'
                else:
                    task_type = 'Regression'
            else:
                task_type = task

            if task_type == 'Regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                st.success(f"✅ Regression R² Score: {score:.2f}")

                st.markdown("""
                #### 📌 Insight:
                - R² score near **1.0** = excellent fit.
                - Score below **0.5** suggests model doesn't explain variability well.
                - Try adding/removing variables or transforming data.
                """)
            else:
                model = LogisticRegression(max_iter=2000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ Classification Accuracy: {acc:.2%}")

                st.write("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.markdown("""
                #### 📌 Insight:
                - Accuracy is the proportion of correct predictions.
                - Check **confusion matrix** for false positives/negatives.
                - Improve using better features or advanced classifiers.
                """)

        except Exception as e:
            st.error(f"Model training failed: {str(e)}")

else:
    st.markdown("""
    ## 👋 Welcome to the Interactive ML & Data Dashboard
    Upload your dataset to begin exploring!
    """)

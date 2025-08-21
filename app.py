import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shap
import joblib
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report

# ------------------- UI CONFIG -------------------
st.set_page_config(page_title="AI-Powered Auto EDA & ML", layout="wide")
st.title("📊 AI-Powered Auto EDA & Prediction Dashboard")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ------------------- LOAD DATA -------------------
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Overview", "📈 EDA", "📝 Insights", "🤖 Modeling", "🔎 Explainability"
    ])

    # ------------------- TAB 1 -------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    # ------------------- TAB 2 -------------------
    with tab2:
        st.subheader("Univariate Analysis")
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.write("🔹 **Categorical Variables**")
        for col in categorical_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        st.write("🔹 **Numerical Variables**")
        for col in numerical_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.subheader("Bivariate Analysis")
        col_x = st.selectbox("X Variable", df.columns, key="biv_x")
        col_y = st.selectbox("Y Variable", df.columns, key="biv_y")
        fig = px.scatter(df, x=col_x, y=col_y, color=col_x)
        st.plotly_chart(fig)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation analysis.")

        # PCA
        if len(numerical_cols) > 2:
            st.subheader("PCA Dimensionality Reduction")
            X_scaled = StandardScaler().fit_transform(df[numerical_cols].dropna())
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            pcs_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
            fig = px.scatter(pcs_df, x="PC1", y="PC2", title="PCA Result")
            st.plotly_chart(fig)

    # ------------------- TAB 3 -------------------
    with tab3:
        st.subheader("Automated Findings & Recommendations")
        findings = []

        # Missing values
        missing_info = df.isnull().mean() * 100
        if (missing_info > 0).any():
            cols_missing = missing_info[missing_info > 0].index.tolist()
            findings.append(f"⚠️ Missing values in {cols_missing} → Use imputation (mean/median/advanced).")

        # Zero variance
        zero_var = [col for col in df.columns if df[col].nunique() <= 1]
        if zero_var:
            findings.append(f"⚠️ Zero variance in {zero_var} → Remove these.")

        # Skewness
        for col in df.select_dtypes(include=np.number).columns:
            skewness = df[col].skew()
            if skewness > 1 or skewness < -1:
                findings.append(f"⚠️ {col} is skewed (skew={skewness:.2f}) → Apply log/BoxCox transform.")

        if not findings:
            findings.append("✅ Data looks clean. Proceed with modeling.")

        for f in findings:
            st.write(f)

        # Download findings
        st.download_button("⬇️ Download Findings", "\n".join(findings), "findings.txt")

    # ------------------- TAB 4 -------------------
    with tab4:
        st.subheader("Train a Model")

        target = st.selectbox("🎯 Select Target Variable", df.columns)
        test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

        if st.button("🚀 Run Model"):
            try:
                df_encoded = df.copy()
                for col in df_encoded.select_dtypes(include="object").columns:
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

                X = df_encoded.drop(columns=[target])
                y = df_encoded[target]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                if y.nunique() <= 10:
                    model = DecisionTreeClassifier(random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Accuracy: {acc:.2%}")

                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

                    st.text(classification_report(y_test, y_pred))

                else:
                    model = DecisionTreeRegressor(random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    st.success(f"✅ R² Score: {score:.2f}")

                # Save model
                joblib.dump(model, "trained_model.pkl")
                st.download_button("⬇️ Download Model", open("trained_model.pkl", "rb"), "trained_model.pkl")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ------------------- TAB 5 -------------------
    with tab5:
        st.subheader("Model Explainability")

        try:
            model = joblib.load("trained_model.pkl")
            if "X_train" in locals():
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_train)

                st.write("### Feature Importance (SHAP)")
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                st.pyplot(bbox_inches="tight")

        except:
            st.warning("⚠️ Train a model first to see explainability results.")

else:
    st.info("👆 Upload a dataset to start the analysis.")

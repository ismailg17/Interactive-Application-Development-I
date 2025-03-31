import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    confusion_matrix, roc_curve, auc,
    classification_report
)
from sklearn.preprocessing import LabelBinarizer

st.set_page_config(page_title="ML Model Trainer", layout="wide")
st.title("Machine Learning Model Trainer App")

# ------------------ DATA LOADING ------------------

@st.cache_data
def load_seaborn_dataset(name):
    return sns.load_dataset(name)

st.sidebar.header(" Dataset Selection")
dataset_name = st.sidebar.selectbox("Choose Seaborn Dataset", sns.get_dataset_names())
df = load_seaborn_dataset(dataset_name)

uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

st.subheader(" Dataset Preview")
st.dataframe(df.head(10))

# ------------------ MODEL CONFIGURATION ------------------

st.sidebar.header("Model Configuration")
target_column = st.sidebar.selectbox("Target Variable", df.columns)
feature_columns = st.sidebar.multiselect(
    "Select Features", [col for col in df.columns if col != target_column]
)

model_type = st.sidebar.selectbox(" Choose Model Type", ["Linear Regression", "Random Forest Classifier"])
test_size = st.sidebar.slider(" Test Size (for test split)", 0.1, 0.5, 0.2)

if model_type == "Random Forest Classifier":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

# ------------------ TRAINING FORM ------------------

with st.form("train_form"):
    st.subheader(" Ready to Train?")
    submitted = st.form_submit_button("🎯 Fit Model")

if submitted:
    if len(feature_columns) == 0:
        st.warning("Please select at least one feature.")
    else:
        # Clean data
        df_clean = df[feature_columns + [target_column]].dropna()
        X = pd.get_dummies(df_clean[feature_columns], drop_first=True)
        y = df_clean[target_column]

        # Detect task type
        is_classification = y.dtype == 'object' or y.nunique() < 20

        if model_type == "Random Forest Classifier" and not is_classification:
            st.error("❌ Target looks continuous. Use Linear Regression.")
        elif model_type == "Linear Regression" and is_classification:
            st.error("❌ Target looks categorical. Use Random Forest Classifier.")
        else:
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            # Store in session state
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test

            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader(" Regression Results")
                st.write("R² Score:", round(r2_score(y_test, preds), 3))
                st.write("MSE:", round(mean_squared_error(y_test, preds), 3))

                fig, ax = plt.subplots()
                sns.histplot(y_test - preds, kde=True, ax=ax)
                ax.set_title("Residual Distribution")
                st.pyplot(fig)

                st.subheader("Feature Importance (Coefficients)")
                st.bar_chart(pd.Series(model.coef_, index=X.columns))

            elif model_type == "Random Forest Classifier":
                model = RandomForestClassifier(n_estimators=n_estimators)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader(" Classification Results")
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                # ROC Curve (if binary)
                if len(np.unique(y)) == 2:
                    probs = model.predict_proba(X_test)[:, 1]
                    y_test_binary = LabelBinarizer().fit_transform(y_test).ravel()
                    fpr, tpr, thresholds = roc_curve(y_test_binary, probs)
                    roc_auc = auc(fpr, tpr)

                    st.subheader(" ROC Curve")
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0, 1], [0, 1], linestyle='--')
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("ROC Curve")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.info("ROC Curve available only for binary classification.")

                st.subheader("Feature Importance")
                st.bar_chart(pd.Series(model.feature_importances_, index=X.columns))

            # Save model in session
            st.session_state["model"] = model

            st.success("Model trained successfully!")

            st.subheader("Download Your Model")
            joblib.dump(model, "trained_model.pkl")
            with open("trained_model.pkl", "rb") as f:
                st.download_button("Download .pkl", f, file_name="trained_model.pkl")


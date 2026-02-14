import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
)

from model.logistic_regression import run_logistic_regression as load_lr
from model.decision_tree_classifier import run_decision_tree_classifier as load_dt
from model.knn_classifier import run_knn as load_knn
from model.naive_bayes_classifier import run_naive_bayes_classifier as load_nb
from model.em_random_forest_classifier import run_random_forest as load_rf
from model.em_xgboost_classifier import run_xgboost as load_xgb

# page configs
st.set_page_config(page_title="Machine Learning Assignment-2", layout="wide")
st.title("Breast Cancer Classification")

# download test data
st.subheader("Download Test Dataset")
with open("data/test_data.csv", "rb") as f:
    st.download_button(
        label="Download Test Data",
        data=f,
        file_name="test_data.csv",
        mime="text/csv"
    )

model_dict = {
    "Logistic Regression": load_lr,
    "Decision Tree Classifier": load_dt,
    "K-Nearest Neighbor Classifier": load_knn,
    "Naive Bayes Classifier": load_nb,
    "Ensemble Model - Random Forest": load_rf,
    "Ensemble Model - XGBoost": load_xgb
}

# upload 
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type="csv")

# select model
model_name = st.selectbox("Select Machine Learning Model ", model_dict.keys())

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    if df["diagnosis"].dtype == object:
        y_test = df["diagnosis"].map({"M": 1, "B": 0})
    else:
        y_test = df["diagnosis"]
    X_test = df.drop(columns=["diagnosis"])

    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Load selected model
    model = model_dict[model_name]()
    # st.write("Loaded model type:", type(model))
    y_pred = model.predict(X_test_scaled)

    #metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics["AUC"] = roc_auc_score(y_test, y_prob)

    st.subheader("Evaluation Metrics on Uploaded Test Data")
    metrics_df = pd.DataFrame(metrics, index=["Score"])

    styled_metrics = (
        metrics_df.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("text-align", "center")]}
            ]
        )
    )

    st.dataframe(styled_metrics, width='stretch')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix")
    _, col_cm, _ = st.columns([1, 3, 1])

    with col_cm:
        fig, ax = plt.subplots(figsize=(3, 3))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            cbar=False,
            square=True,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig, width='content')
        plt.close(fig)







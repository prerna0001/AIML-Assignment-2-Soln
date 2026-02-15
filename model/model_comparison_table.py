import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

from model.data_prep_py import get_preprocessed_data
from model.logistic_regression import run_logistic_regression
from model.decision_tree_classifier import run_decision_tree_classifier
from model.knn_classifier import run_knn
from model.naive_bayes_classifier import run_naive_bayes_classifier
from model.em_random_forest_classifier import run_random_forest
from model.em_xgboost_classifier import run_xgboost

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["AUC"] = roc_auc_score(y_test, y_prob)
    else:
        metrics["AUC"] = None

    return metrics


def build_comparison_table():
    
    # comparison table for all the  models
    
    _, X_test, _, y_test = get_preprocessed_data()

    results = []

    models = [
        ("Logistic Regression", run_logistic_regression()),
        ("Decision Tree Classifier", run_decision_tree_classifier()),
        ("K-Nearest Neighbor Classifier", run_knn()),
        ("Naive Bayes Classifier", run_naive_bayes_classifier()),
        ("Ensemble Model - Random Forest", run_random_forest()),
        ("Ensemble Model - XGBoost", run_xgboost())
    ]

    for model_name, model in models:
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "Model": model_name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "AUC": round(metrics["AUC"], 4) if metrics["AUC"] is not None else "NA",
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1"], 4),
            "MCC": round(metrics["MCC"], 4)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = build_comparison_table()
    print(df)

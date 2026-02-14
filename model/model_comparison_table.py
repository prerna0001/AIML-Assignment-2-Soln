import pandas as pd

from model.logistic_regression import run_logistic_regression
from model.decision_tree_classifier import run_decision_tree_classifier
from model.knn_classifier import run_knn
from model.naive_bayes_classifier import run_naive_bayes_classifier
from model.em_random_forest_classifier import run_random_forest
from model.em_xgboost_classifier import run_xgboost


def build_comparison_table():
    
    # comparison table for all the  models
    

    results = []

    model_outputs = [
        ("Logistic Regression", run_logistic_regression()),
        ("Decision Tree Classifier", run_decision_tree_classifier()),
        ("K-Nearest Neighbor Classifier", run_knn()),
        ("Naive Bayes Classifier", run_naive_bayes_classifier()),
        ("Ensemble Model - Random Forest", run_random_forest()),
        ("Ensemble Model - XGBoost", run_xgboost())
    ]

    for model_name, metrics in model_outputs:
        results.append({
            "Model": model_name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "AUC": round(metrics["AUC"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1 Score": round(metrics["F1"], 4),
            "MCC": round(metrics["MCC"], 4)
        })

    comparison_df = pd.DataFrame(results)
    return comparison_df


if __name__ == "__main__":
    df = build_comparison_table()
    print(df)

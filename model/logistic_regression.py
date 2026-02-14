from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from model.data_prep_py import get_preprocessed_data


def run_logistic_regression():
    #logistic regression classifier

    (
        train_features_scaled,
        test_features_scaled,
        train_target,
        test_target
    ) = get_preprocessed_data()

    logisticRegression = LogisticRegression(
        max_iter=200,
        solver="liblinear",
        random_state=47
    )

    # train model
    logisticRegression.fit(train_features_scaled, train_target)

    pred_labels = logisticRegression.predict(test_features_scaled)
    pred_probs = logisticRegression.predict_proba(test_features_scaled)[:, 1]

    # Metrics
    results = {
        "Accuracy": accuracy_score(test_target, pred_labels),
        "AUC": roc_auc_score(test_target, pred_probs),
        "Precision": precision_score(test_target, pred_labels),
        "Recall": recall_score(test_target, pred_labels),
        "F1": f1_score(test_target, pred_labels),
        "MCC": matthews_corrcoef(test_target, pred_labels),
        "Confusion Matrix": confusion_matrix(test_target, pred_labels),
        "Classification Report": classification_report(test_target, pred_labels)
    }

    return logisticRegression

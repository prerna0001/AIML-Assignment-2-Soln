from sklearn.tree import DecisionTreeClassifier
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


def run_decision_tree_classifier():
    # Decision Tree Classifier
    (
        train_features_scaled,
        test_features_scaled,
        train_target,
        test_target
    ) = get_preprocessed_data()

    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=47
    )

    decision_tree.fit(train_features_scaled, train_target)

    pred_labels = decision_tree.predict(test_features_scaled)
    pred_probs = decision_tree.predict_proba(test_features_scaled)[:, 1]

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

    return results

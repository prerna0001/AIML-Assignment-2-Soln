
Matrics of Training:

                            Model  Accuracy     AUC  Precision  Recall  F1 Score     MCC
0             Logistic Regression    0.9580  0.9883     0.9434  0.9434    0.9434  0.9101
1        Decision Tree Classifier    0.9441  0.9461     0.9245  0.9245    0.9245  0.8801
2   K-Nearest Neighbor Classifier    0.9371  0.9816     0.9231  0.9057    0.9143  0.8647
3          Naive Bayes Classifier    0.9231  0.9860     0.8889  0.9057    0.8972  0.8358
4  Ensemble Model - Random Forest    0.9580  0.9855     0.9273  0.9623    0.9444  0.9112
5        Ensemble Model - XGBoost    0.9580  0.9918     0.9608  0.9245    0.9423  0.9098


## Streamlit Web Application

The Streamlit app allows users to:
- Download Test data (CSV)
- Upload test dataset (CSV)
- Select any of the 6 implemented ML models
- View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Visualize confusion matrix

# Machine Learning Assignment – 2  
## Problem Statement - Breast Cancer Classification using Supervised Learning Models

The objective of this assignment is to design, implement, and evaluate multiple supervised machine learning models for binary classification.
Using the **Breast Cancer Wisconsin (Diagnostic)** dataset, the goal is to predict whether a tumor is Malignant or Benign based on numerical diagnostic features, compare the performance of different models using standard evaluation metrics, and deploy an interactive **Streamlit web application** for model evaluation on test data.

---

## Dataset Information

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** Kaggle - (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Number of Samples:** 569
- **Number of Features:** 30 (numeric)
- **Target Variable:** `diagnosis`
  - `M` → Malignant
  - `B` → Benign

>Unnecessary columns such as `id` and `Unnamed: 32` were removed during preprocessing.
---

## Preprocessing

- Dropped non-informative columns (`id`, `Unnamed: 32`)
- Encoded target labels (`M → 1`, `B → 0`)
- Applied **StandardScaler** for feature normalization
- Dataset was split offline into **training and test datasets** using stratified sampling
---

## Machine Learning Models Implemented

The following supervised learning models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier 
5. Ensemble Model – Random Forest Classifier
6. Ensemble Model – XGBoost Classifier

Each model was trained on the same training dataset and evaluated on a common test dataset to ensure fair comparison.

---

## Model Comparison

All models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Area Under the ROC Curve (AUC)

### Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|---------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9580 | 0.9883 | 0.9434 | 0.9434 | 0.9434 | 0.9101 |
| Decision Tree Classifier | 0.9441 | 0.9461 | 0.9245 | 0.9245 | 0.9245 | 0.8801 |
| K-Nearest Neighbor Classifier | 0.9371 | 0.9816 | 0.9231 | 0.9057 | 0.9143 | 0.8647 |
| Naive Bayes Classifier | 0.9231 | 0.9860 | 0.8889 | 0.9057 | 0.8972 | 0.8358 |
| Ensemble Model – Random Forest | 0.9580 | 0.9855 | 0.9273 | 0.9623 | 0.9444 | 0.9112 |
| Ensemble Model – XGBoost | 0.9580 | 0.9918 | 0.9608 | 0.9245 | 0.9423 | 0.9098 |

---

## Observations

| ML Model Name | Observation about Model Performance |
|--------------|------------------------------------|
| Logistic Regression | Achieved high accuracy and AUC, indicating that the dataset is well-separated and suitable for linear decision boundaries. |
| Decision Tree | Performed well but showed slightly lower generalization due to sensitivity to data splits and potential overfitting. |
| kNN | Delivered competitive results but was sensitive to feature scaling and choice of neighbors. |
| Naive Bayes | Performed reasonably well despite strong independence assumptions, showing robustness on this dataset. |
| Random Forest (Ensemble) | Achieved high recall and balanced performance by aggregating multiple decision trees, reducing overfitting. |
| XGBoost (Ensemble) | Achieved the highest AUC 0.9918, indicating excellent discriminative ability and strong ensemble learning performance. and highest Precision among all 0.9608|


---

## Streamlit Web Application

An interactive **Streamlit application** developed to evaluate already trained models on test data.

### Features:
- Downloadable sample test dataset (CSV)
- Test dataset upload (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix visualization

> Models are **not trained** within the Streamlit application.

### Live App Link
https://aiml-assignment-2-soln-prerna.streamlit.app/

---

## Project Structure
```text
AIML-Assignment-2-Soln/
│── app.py
│── requirements.txt
│── README.md
│
├── data/
│ └── data.csv
│ └── test_data.csv
│ └── train_data.csv
│
├── model/
│ ├── data_prep_py.py
│ ├── model_comparison_table.py
│ ├── logistic_regression.py
│ ├── decision_tree_classifier.py
│ ├── knn_classifier.py
│ ├── naive_bayes_classifier.py
│ ├── em_random_forest_classifier.py
│ └── em_xgboost_classifier.py
```
---

## How to Run Locally

1. Install dependencies:
   
    pip install -r requirements.txt

2. Run Streamlit app:

    streamlit run app.py 
    > this on successful run, will take you to http://localhost:8501/

3. Run model comparison:

    python -m model.model_comparison_table
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_preprocessed_data(
    # data_path="../data/data.csv",
    test_size=0.25,
    random_state=27
):
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "data.csv")
    df = pd.read_csv(data_path)

    #dropping non-req columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    #feature–target split
    ip_features = df.drop("diagnosis", axis=1)
    target = df["diagnosis"]

    # training test split
    train_features, test_features, train_target, test_target = train_test_split(
        ip_features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    return (
        train_features_scaled,
        test_features_scaled,
        train_target,
        test_target
    )

# one time to split data
def split_and_save_data_csv(
    test_size=0.25,
    random_state=27
):
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "data.csv")

    df = pd.read_csv(data_path)

    # dropping non-req columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    # encode target
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    #feature–target split
    ip_features = df.drop("diagnosis", axis=1)
    target = df["diagnosis"]

    # training test split
    train_features, test_features, train_target, test_target = train_test_split(
        ip_features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    train_df = pd.concat([train_features, train_target], axis=1)
    test_df = pd.concat([test_features, test_target], axis=1)

    train_path = os.path.join(base_dir, "..", "data", "train_data.csv")
    test_path = os.path.join(base_dir, "..", "data", "test_data.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train and Test CSV files created successfully")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.paths import ProjectPaths

LOGGING_ENABLED = True


def log(message):
    if LOGGING_ENABLED:
        print(f"[LOG] {message}")


def load_and_prepare_data():
    df_path = ProjectPaths.DATA_ML_CSV_FOLDER / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df = pd.read_csv(df_path)
    df.columns = df.columns.str.strip()
    df['Label'] = df['Label'].str.strip()
    df = df[df['Label'].isin(['BENIGN', 'DDoS'])]

    log(f"Loaded dataset: {df_path.name}")
    log(f"Total entries: {len(df)}")
    log(f"Label distribution:\n{df['Label'].value_counts().to_string()}")
    return df


def clean_numeric_features(df):
    numeric_cols = df.select_dtypes(include='number').columns
    df_numeric = df[numeric_cols].copy()
    df_non_numeric = df.drop(columns=numeric_cols)

    inf_mask = df_numeric.isin([np.inf, -np.inf])
    if inf_mask.any().any():
        inf_cols = list(df_numeric.columns[inf_mask.any()])
        log(f"Replacing infinities in columns: {inf_cols}")
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

    nan_cols = df_numeric.columns[df_numeric.isna().any()]
    if len(nan_cols) > 0:
        log(f"Imputing NaNs with column means in: {list(nan_cols)}")
        df_numeric[nan_cols] = df_numeric[nan_cols].fillna(df_numeric[nan_cols].mean())

    if 'Flow Bytes/s' in df_numeric.columns:
        clip_threshold = df_numeric['Flow Bytes/s'].quantile(0.999)
        df_numeric['Flow Bytes/s'] = df_numeric['Flow Bytes/s'].clip(upper=clip_threshold)
        log(f"Clipped 'Flow Bytes/s' at 99.9th percentile: {clip_threshold:.2f}")

    df_cleaned = pd.concat([df_numeric, df_non_numeric], axis=1)

    log("Data cleaning complete.")
    return df_cleaned


def sample_train_test_sets(df_cleaned, train_size=10_000, test_size=20_000):
    train_benign_ratio = 0.8
    test_benign_ratio = 0.57

    train_benign_size = round(train_size * train_benign_ratio)
    train_ddos_size = train_size - train_benign_size
    test_benign_size = round(test_size * test_benign_ratio)
    test_ddos_size = test_size - test_benign_size

    df_benign = df_cleaned[df_cleaned['Label'] == 'BENIGN']
    df_ddos = df_cleaned[df_cleaned['Label'] == 'DDoS']

    train_benign = df_benign.sample(n=train_benign_size)
    train_ddos = df_ddos.sample(n=train_ddos_size)
    df_train = pd.concat([train_benign, train_ddos])

    df_remaining = df_cleaned.drop(df_train.index)
    test_benign = df_remaining[df_remaining['Label'] == 'BENIGN'].sample(n=test_benign_size)
    test_ddos = df_remaining[df_remaining['Label'] == 'DDoS'].sample(n=test_ddos_size)
    df_test = pd.concat([test_benign, test_ddos])

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    log(f"Training set sampled: {df_train.shape[0]} rows ({train_benign_size} BENIGN, {train_ddos_size} DDoS)")
    log(f"Testing set sampled: {df_test.shape[0]} rows ({test_benign_size} BENIGN, {test_ddos_size} DDoS)")
    return df_train, df_test


def apply_svd(df_train):
    X_train = df_train.select_dtypes(include='number')
    U, S, VT = np.linalg.svd(X_train, full_matrices=False)
    svd_train = U[:, :2] * S[:2]
    components = VT[:2, :]
    components_df = pd.DataFrame(components, columns=X_train.columns, index=['SVD1', 'SVD2'])

    train_svd_df = pd.DataFrame(svd_train, columns=['SVD1', 'SVD2'])
    train_svd_df['Label'] = df_train['Label']

    log("SVD applied to training set.")
    log(f"Top 2 components extracted. Shape: {components_df.shape}")
    return train_svd_df, components_df


def project_test_set(df_test, components_df):
    X_test = df_test.select_dtypes(include='number')
    svd_test = X_test @ components_df.T
    svd_test.columns = ['SVD1', 'SVD2']
    svd_test['Label'] = df_test['Label']

    log("Test set projected onto SVD components.")
    return svd_test


def train_and_evaluate_classifier(train_svd_df, test_svd_df):
    X_train_svd = train_svd_df[['SVD1', 'SVD2']]
    y_train_svd = train_svd_df['Label']
    X_test_svd = test_svd_df[['SVD1', 'SVD2']]
    y_test_svd = test_svd_df['Label']

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_svd, y_train_svd)
    y_pred = clf.predict(X_test_svd)

    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            accuracy_score(y_test_svd, y_pred),
            precision_score(y_test_svd, y_pred, pos_label='DDoS'),
            recall_score(y_test_svd, y_pred, pos_label='DDoS'),
            f1_score(y_test_svd, y_pred, pos_label='DDoS')
        ]
    })

    log("Model trained and evaluated on SVD-transformed data.")
    log("Evaluation metrics:")
    for _, row in results_df.iterrows():
        log(f"  {row['Metric']}: {row['Value']:.4f}")
    return results_df


# ============================================
# Pipeline Entry Point
# ============================================

def run_ddos_detection_pipeline():
    log("=== DDoS Detection Pipeline Started ===")
    df_raw = load_and_prepare_data()
    df_cleaned = clean_numeric_features(df_raw)
    df_train, df_test = sample_train_test_sets(df_cleaned)
    train_svd_df, components_df = apply_svd(df_train)
    test_svd_df = project_test_set(df_test, components_df)
    results_df = train_and_evaluate_classifier(train_svd_df, test_svd_df)
    log("=== Pipeline Complete ===")
    return df_cleaned, results_df, df_train, train_svd_df, components_df

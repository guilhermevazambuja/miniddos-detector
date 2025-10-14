from src.ddos_detector_pipeline import run_ddos_detection_pipeline
from src.download import download_dataset
from src.eda import run_eda

if __name__ == "__main__":
    download_dataset()
    df_cleaned, results_df, df_train, train_svd_df, components_df = run_ddos_detection_pipeline()
    run_eda(df_cleaned, df_train, train_svd_df, components_df)

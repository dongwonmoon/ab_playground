import mlflow
import pandas as pd

from src.utils.config_loader import load_config
from src.models.ncf import NCF
from src.models.svd import SVD
from src.models.svd_pytorch import SVD_PyTorch
from src.data.preprocessing import mapping_id_to_unique


def run_training():
    print("Loading Configuration...")
    config = load_config("configs/config.yml")

    mlflow_config = config["mlflow"]
    if "tracking_uri" in mlflow_config:
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    print(f"MLflow tracking URI is now set to: {mlflow.get_tracking_uri()}")

    print("Loading pre-processed data and creating ID maps...")

    full_df = pd.read_parquet(config["data"]["processed_path"])
    train_df = pd.read_parquet(config["data"]["train_data_path"])

    num_users, num_movies, train_df = mapping_id_to_unique(full_df, train_df)

    print(f"Number of users for Embedding: {num_users}")
    print(f"Number of movies for Embedding: {num_movies}")
    print(f"Training data size: {len(train_df)}")

    # 모델 학습
    mlflow.set_experiment(mlflow_config["experiment_name"])
    print("\n--- Starting Training and Logging ---")

    for model_name in config["models"]:
        with mlflow.start_run(run_name=model_name.upper()):
            model_config = config["models"][model_name]
            if model_name == "ncf":
                model = NCF(
                    params=model_config["params"],
                    num_users=num_users,
                    num_movies=num_movies,
                )
            elif model_name == "svd":
                model = SVD(params=model_config["params"])
            elif model_name == "svd_pytorch":
                model = SVD_PyTorch(
                    params=model_config["params"],
                    num_users=num_users,
                    num_movies=num_movies,
                )

            model.train_model(train_df)
            print(f"Logging {model_name.upper()} to MLflow...")
            model.log_to_mlflow(
                run_name=model_name.upper(),
            )

    print("\n--- Training and Logging Complete! ---")
    print("Check MLflow UI to see the results.")


if __name__ == "__main__":
    run_training()

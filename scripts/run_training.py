### 추후 개선 ###
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
### 추후 개선 ###

import mlflow
import pandas as pd

from src.utils.config_loader import load_config
from src.models.ncf import NCF
from src.models.svd import SVD


def run_training():
    print("Loading Configuration...")
    config = load_config("configs/config.yml")

    mlflow_config = config["mlflow"]
    if "tracking_uri" in mlflow_config:
        mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    print(f"MLflow tracking URI is now set to: {mlflow.get_tracking_uri()}")

    print("Loading pre-processed data and creating ID maps...")

    full_df = pd.read_parquet(config["data"]["processed_path"])
    unique_users = full_df["userId"].unique()
    unique_movies = full_df["movieId"].unique()

    # Mapper
    user_to_idx = {original: new for new, original in enumerate(unique_users)}
    movie_to_idx = {original: new for new, original in enumerate(unique_movies)}

    num_users = len(unique_users)
    num_movies = len(unique_movies)

    train_df = pd.read_parquet(config["data"]["train_data_path"])
    train_df["userId"] = train_df["userId"].map(user_to_idx)
    train_df["movieId"] = train_df["movieId"].map(movie_to_idx)

    train_df.dropna(inplace=True)
    train_df["userId"] = train_df["userId"].astype(int)
    train_df["movieId"] = train_df["movieId"].astype(int)

    print(f"Number of users for Embedding: {num_users}")
    print(f"Number of movies for Embedding: {num_movies}")
    print(f"Training data size: {len(train_df)}")

    # 모델 학습
    with mlflow.start_run(run_name="NCF"):
        ncf_config = config["models"]["ncf"]
        model_ncf = NCF(
            params=ncf_config["params"],
            num_users=num_users,
            num_movies=num_movies,
        )

        model_ncf.train_model(train_df)

        print("Logging NCF to MLflow...")
        model_ncf.log_to_mlflow(
            experiment_name=mlflow_config["experiment_name"], run_name="NCF"
        )

    with mlflow.start_run(run_name="SVD"):
        svd_config = config["models"]["svd"]
        model_svd = SVD(params=svd_config["params"])

        model_svd.train_model(train_df)

        print("Logging SVD to MLflow...")
        model_svd.log_to_mlflow(
            experiment_name=mlflow_config["experiment_name"], run_name="SVD"
        )

    print("\n--- Training and Logging Complete! ---")
    print("Check MLflow UI to see the results.")


if __name__ == "__main__":
    run_training()

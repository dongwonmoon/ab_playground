import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import mlflow

from src.models.base_model import BaseModel
from src.models.wrapper import Wrapper
from src.data.datasets import CustomDataset


class SVD_PyTorch(BaseModel, nn.Module):
    def __init__(self, params: dict, num_users: int, num_movies: int):
        BaseModel.__init__(self, params)
        nn.Module.__init__(self)

        n_factors = self.params.get("n_factors", 50)

        self.user_embedding = nn.Embedding(num_users, n_factors)
        self.movie_embedding = nn.Embedding(num_movies, n_factors)

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        self.global_bias = nn.Parameter(torch.randn(1))

    def forward(self, user_indices, movie_indices):
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        user_b = self.user_bias(user_indices).squeeze()
        movie_b = self.movie_bias(movie_indices).squeeze()

        dot_product = (user_embed * movie_embed).sum(1)

        prediction = dot_product + user_b + movie_b + self.global_bias

        return prediction

    def train_model(self, data: pd.DataFrame):
        print("Training SVD_PyTorch model...")

        # Data
        dataset = CustomDataset(data)
        batch_size = self.params.get("batch_size", 256)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Params
        learning_rate = self.params.get("learning_rate", 0.005)
        reg_lambda = self.params.get("reg_lambda", 0.01)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

        criterion = nn.MSELoss()

        self.train()  # pytorch train mode
        epochs = self.params.get("epochs", 20)
        for epoch in range(epochs):
            total_loss = 0
            for users, movies, ratings in loader:
                optimizer.zero_grad()
                predictions = self(users, movies)

                loss = criterion(predictions, ratings)

                # SVD Reg term
                reg_term = self.user_embedding.weight.norm(
                    2
                ) + self.movie_embedding.weight.norm(2)
                loss += reg_lambda * reg_term

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # MLFlow Logging (Every Epoch)
            mlflow.log_metric("train_loss_MSE_with_reg", avg_loss, step=epoch)
        self._trained = True
        print("SVD_PyTorch model training complete.")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """학습된 모델로 예측을 수행함."""
        if not self._trained:
            raise ValueError(
                "Model has not been trained yet. Call train_model() first."
            )

        self.eval()

        user_tensor = torch.tensor(data["userId"].values, dtype=torch.long)
        movie_tensor = torch.tensor(data["movieId"].values, dtype=torch.long)

        with torch.no_grad():
            predictions = self(user_tensor, movie_tensor)

        result_df = data.copy()
        result_df["prediction"] = predictions.numpy()
        return result_df

    def _log_model_to_mlflow(self, run_name: str):
        """NCF 모델을 로깅"""
        print("Using mlflow.pytorch.log_model for SVD_PyTorch...")
        mlflow.pyfunc.log_model(
            python_model=Wrapper(self),
            name=run_name,
        )

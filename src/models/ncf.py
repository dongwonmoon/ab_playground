import pandas as pd
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.base_model import BaseModel
from src.data.datasets import CustomDataset


class NCF(BaseModel, nn.Module):
    """NCF 모델. BaseModel을 상속받음."""

    def __init__(self, params: dict, num_users: int, num_movies: int):
        BaseModel.__init__(self, params)
        nn.Module.__init__(self)

        embedding_dim = self.params.get("embedding_dim", 32)

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def train_model(self, data: pd.DataFrame):  # nn.Module의 train함수와 겹침 방지
        """NCF 모델을 학습"""
        print("Training NCF model...")

        dataset = CustomDataset(data)
        batch_size = self.params.get("batch_size", 256)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        learning_rate = self.params.get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.train()

        epochs = self.params.get("epochs", 5)
        for epoch in range(epochs):
            total_loss = 0
            for users, movies, ratings in loader:
                optimizer.zero_grad()
                predictions = self(users, movies)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            mlflow.log_metric("train_loss_MSE", avg_loss, step=epoch)

        print("NCF model training complete.")

    def forward(self, user_indices, movie_indices):
        user_embed = self.user_embedding(user_indices)
        movie_embed = self.movie_embedding(movie_indices)
        x = torch.cat([user_embed, movie_embed], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x)) * 4.5 + 0.5
        return x.squeeze()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """학습된 모델로 예측을 수행함."""
        if self._model is None:
            raise ValueError(
                "Model has not been trained yet. call train_model() first."
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
        print("Using mlflow.pytorch.log_model for NCF...")
        mlflow.pytorch.log_model(
            pytorch_model=self,
            name=run_name,
        )

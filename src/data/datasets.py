import torch
from torch.utils.data import Dataset
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        if not set(df.columns).issuperset(set(["userId", "movieId", "rating"])):
            raise ValueError("Column must be userId, movieId, rating.")

        self.users = torch.tensor(df["userId"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index], self.movies[index], self.ratings[index]

import pandas as pd
import numpy as np
import mlflow
from surprise import SVD as SurpriseSVD
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate

from src.models.base_model import BaseModel
from src.models.wrapper import Wrapper


class SVD(BaseModel):
    """SVD모델. BaseModel을 상속받아 구현함."""

    def __init__(self, params: dict):
        super().__init__(params)

    def train_model(self, data: pd.DataFrame):
        """Surprise 라이브러리를 통해 SVD 학습."""
        print("Training SVD model...")
        reader = Reader(rating_scale=(0.5, 5.0))
        train_data = Dataset.load_from_df(data[["userId", "movieId", "rating"]], reader)

        # cross_validate를 위해 새로운 모델 객체를 생성.
        svd_for_cv = SurpriseSVD(**self.params)
        print("Performing 5-fold cross-validation...")
        cv_results = cross_validate(
            svd_for_cv, train_data, measures=["RMSE", "MAE"], cv=5, verbose=False
        )

        mean_rmse = np.mean(cv_results["test_rmse"])
        mean_mae = np.mean(cv_results["test_mae"])
        print(
            f"Cross-validation results: Mean RMSE={mean_rmse:.4f}, Mean MAE={mean_mae:.4f}"
        )
        mlflow.log_metrics({"svd_mean_rmse": mean_rmse, "svd_mean_mae": mean_mae})

        # 최종 모델 학습
        trainset = train_data.build_full_trainset()
        self._model = SurpriseSVD(**self.params)
        self._model.fit(trainset)
        self._trained = True
        print("SVD model training complete.")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """학습된 SVD 모델로 예측을 수행"""
        if not self._trained:
            raise ValueError(
                "Model has not been trained yet. Call train_model() first."
            )

        reader = Reader(rating_scale=(0.5, 5.0))
        predict_data = data.copy()
        if "rating" not in predict_data.columns:
            predict_data["rating"] = 0

        surprise_dataset = Dataset.load_from_df(
            predict_data[["userId", "movieId", "rating"]], reader
        )
        testset = surprise_dataset.build_full_trainset().build_testset()
        predictions = self._model.test(testset)

        predictions_df = pd.DataFrame(
            [(pred.uid, pred.iid, pred.est) for pred in predictions],
            columns=["userId", "movieId", "prediction"],
        )
        result_df = pd.merge(data, predictions_df, on=["userId", "movieId"], how="left")
        return result_df

    def _log_model_to_mlflow(self, run_name: str):
        """SVD 모델을 로깅합니다."""
        print("Using mlflow.pyfunc.log_model for SVD with custom wrapper...")
        mlflow.pyfunc.log_model(
            name=run_name,
            python_model=Wrapper(model=self),
        )

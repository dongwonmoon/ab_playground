import pandas as pd
import mlflow
from surprise import SVD as SurpriseSVD
from surprise import Reader, Dataset

from src.models.base_model import BaseModel


class _SVDWrapper(mlflow.pyfunc.PythonModel):
    """MLflow 로깅을 위한 SVD 모델 래퍼 클래스."""

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        predictions = []
        for _, row in model_input.iterrows():
            # Data 구조를 알고 있다는 가정 하에 작성.
            # input Data는 레이블을 제외한 데이터프레임.
            pred = self.model.predict(row["userId"], row["movieId"])
            predictions.append(pred.est)
        model_input["prediction"] = predictions
        return model_input


class SVD(BaseModel):
    """SVD모델. BaseModel을 상속받아 구현함."""

    def train_model(self, data: pd.DataFrame):
        """Surprise 라이브러리를 통해 SVD 학습."""
        print("Training SVD model...")
        reader = Reader(rating_scale=(0.5, 5))
        # 데이터 구조를 알고 있다는 가정 하에 작성.
        train_data = Dataset.load_from_df(data[["userId", "movieId", "rating"]], reader)
        trainset = train_data.build_full_trainset()

        self._model = SurpriseSVD(**self.params)
        self._model.fit(trainset)
        print("SVD model training complete.")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        학습된 SVD 모델로 예측을 수행.

        Args:
            data (pd.DataFrame): 예측에 사용할 'userId', 'itemId' 컬럼을 가진 데이터프레임.

        Returns:
            pd.DataFrame: 입력 데이터에 'prediction' 컬럼이 추가된 데이터프레임.
        """
        if self._model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        reader = Reader(rating_scale=(0.5, 5))
        test_data = Dataset.load_from_df(data[["userId", "movieId"]], reader)
        testset = test_data.build_full_trainset().build_testset()

        predictions = self._model.test(testset)
        pred_map = {(pred.uid, pred.iid): pred.est for pred in predictions}

        result_df = data.copy()
        result_df["prediction"] = result_df.apply(
            lambda row: pred_map.get((row["userId"], row["movieId"])), axis=1
        )

        return result_df

    def _create_mlflow_wrapper(self):
        return _SVDWrapper(self.get_model())

    def _log_model_to_mlflow(self, run_name: str):
        """SVD 모델을 로깅."""
        print("Using mlflow.pyfunc.log_model for SVD...")

        input_example = pd.DataFrame({"userId": [0], "movieId": [0]})

        mlflow.pyfunc.log_model(
            name=run_name,
            python_model=self._create_mlflow_wrapper(),
            input_example=input_example,
        )

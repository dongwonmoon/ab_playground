from abc import ABC, abstractmethod
import mlflow
import pandas as pd


class BaseModel(ABC):
    """학습, 예측 등을 위한 표준 인터페이스 정의"""

    def __init__(self, params: dict):
        """
        모델에 필요한 하이퍼파라미터 초기화

        Args:
            params (dict): 모델의 하이퍼파라미터 딕셔너리
        """
        self.params = params
        self._trained = False

    @abstractmethod
    def train_model(self, data: pd.DataFrame):
        """
        모델 학습. 이 메서드는 하위 클래스에서 반드시 구현해야함.

        Args:
            data (pd.DataFrame): 학습에 사용할 데이터
        """
        pass

    @abstractmethod
    def predict(sef, data: pd.DataFrame) -> pd.DataFrame:
        """
        예측 수행. 이 메서드는 하위 클래스에서 반드시 구현해야함.

        Args:
            data (pd.DataFrame): 예측을 수행할 데이터

        Returns:
            pd.DataFrame: 예측 결과 (Columns: userId, movieId, prediction)
        """
        pass

    @abstractmethod
    def _log_model_to_mlflow(self, run_name: str):
        """각 자식 클래스가 자신의 타입(pytorch, python, ...)에 맞게 모델 로깅 구현"""
        pass

    def log_to_mlflow(self, run_name: str):
        self._log_model_to_mlflow(run_name)
        mlflow.log_params(self.params)

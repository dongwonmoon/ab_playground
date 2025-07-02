import mlflow
import pandas as pd


class Wrapper(mlflow.pyfunc.PythonModel):
    """
    모델을 MLflow pyfunc 표준에 맞게 감싸는 래퍼.
    pyfunc의 predict 요청을 모델의 자체 predict 메서드로 연결.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame):
        return self.model.predict(model_input)

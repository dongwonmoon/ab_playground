import pandas as pd
import numpy as np
import tqdm


class ABTestSimulator:
    def __init__(self, full_df: pd.DataFrame, test_df: pd.DataFrame):
        """시뮬레이션에 사용할 테스트 데이터프레임을 받아 초기화합니다."""
        self.full_df = full_df
        self.test_df = test_df
        self.all_movie_ids = full_df["movieId"].unique()
        self.__num_A = 0
        self.__num_B = 0

    def set_group_num(self, num_A, num_B):
        self.__num_A = num_A
        self.__num_B = num_B

    def get_group_a(self):
        return self.__num_A

    def get_group_b(self):
        return self.__num_B

    def sample_group(self):
        df = self.test_df.copy()
        group_A = df.sample(self.__num_A)
        df.drop(group_A.index, inplace=True)
        group_B = df.sample(self.__num_B)

        del df
        return group_A, group_B

    def run_simulation(
        self, model_a, model_b, top_k: int, success_threshold: float
    ) -> dict:
        """
        두 모델(A, B)을 받아 각 사용자에 대한 추천을 생성하고,
        실제 평점(ground truth)을 기반으로 '성공(전환)' 여부를 판단합니다.
        - Return: {'visitors_a': ..., 'conversions_a': ...,
                     'visitors_b': ..., 'conversions_b': ...}
        """
        group_A, group_B = self.sample_group()

        conversions_a = 0
        conversions_b = 0

        print("\n --- Model A Starting ---")
        for user_id in tqdm.tqdm(group_A["userId"]):
            if self._get_hit_for_user(model_a, user_id, top_k, success_threshold):
                conversions_a += 1

        print("\n --- Model B Starting ---")
        for user_id in tqdm.tqdm(group_B["userId"]):
            if self._get_hit_for_user(model_b, user_id, top_k, success_threshold):
                conversions_b += 1

        return {
            "visitors_a": self.__num_A,
            "conversions_a": conversions_a,
            "visitors_b": self.__num_B,
            "conversions_b": conversions_b,
        }

    def _get_hit_for_user(
        self,
        model,
        user_id: int,
        top_k: int,
        success_threshold: float,
    ) -> bool:
        """
        특정 사용자에 대해 모델의 추천이 "Hit" 했는 지 여부를 반환.
        """
        seen_movies = self.full_df[self.full_df["userId"] == user_id]["movieId"].values
        candidate_movies = np.setdiff1d(self.all_movie_ids, seen_movies)

        predict_df = pd.DataFrame(
            {
                "userId": user_id,
                "movieId": candidate_movies,
            }
        )

        predictions = model.predict(predict_df)

        top_k_recs = predictions.nlargest(top_k, "prediction")["movieId"].values

        ground_truth = self.test_df[
            (self.test_df["userId"] == user_id)
            & (self.test_df["rating"] >= success_threshold)
        ]["movieId"].values

        is_hit = len(np.intersect1d(top_k_recs, ground_truth)) > 0
        return is_hit

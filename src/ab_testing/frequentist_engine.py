from statsmodels.stats.proportion import proportions_ztest
import numpy as np


def get_p_value(
    conversions_a: int, visitors_a: int, conversions_b: int, visitors_b: int
) -> float:
    """
    두 그룹(A/B) 간의 전환률 차이에 대한 p-value 계산.
    - 전환: 모델이 추천한 Item을 사용자가 클릭한 경우.
    - H0: 두 모델 간 성능 차이는 없다.
    """

    if conversions_a == 0 and conversions_b == 0:
        return 1.0

    count = np.array([conversions_a, conversions_b])  # 전체 횟수
    nobs = np.array([visitors_a, visitors_b])  # 성공 횟수

    stat, p_val = proportions_ztest(count, nobs)
    return p_val

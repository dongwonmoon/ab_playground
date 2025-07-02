from scipy.stats import beta
import numpy as np


def get_bayesian_result(
    conversions_a: int,
    visitors_a: int,
    conversions_b: int,
    visitors_b: int,
    simulations=100000,
) -> tuple[float, float]:
    """
    베이지안 분석을 통해 'B 모델이 A 모델보다 나을 확률'을 계산.
    """
    # --- B(성공횟수, 실패횟수) ---
    alpha_a = conversions_a + 1  # 성공 횟수
    beta_a = visitors_a - conversions_a + 1  # 실패 횟수

    alpha_b = conversions_b + 1  # 성공 횟수
    beta_b = visitors_b - conversions_b + 1  # 실패 횟수

    # 베타분포 시뮬레이션
    samples_a = beta.rvs(alpha_a, beta_a, size=simulations)
    samples_b = beta.rvs(alpha_b, beta_b, size=simulations)

    # 시뮬레이션 결과 B모델에서 성공 확률(전환할 확률)이 A모델보다 얼마나 높은가 ?
    prob_b_better_than_a = np.mean(samples_b > samples_a)

    # B모델의 전환률이 A모델보다 높은 정도 [-1, 1]
    uplift_samples = (samples_b - samples_a) / (samples_a + 1e-10)
    expected_uplift = np.mean(uplift_samples)

    return prob_b_better_than_a, expected_uplift

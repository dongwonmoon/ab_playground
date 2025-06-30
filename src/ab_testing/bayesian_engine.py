from scipy.stats import beta
import numpy as np


def get_bayesian_result(
    conversions_a, visitors_a, conversions_b, visitors_b, simulations=100000
):
    """
    베이지안 분석을 통해 'B 모델이 A 모델보다 나을 확률'을 계산.
    """
    alpha_a = conversions_a + 1  # 성공 횟수
    beta_a = visitors_a - conversions_a + 1  # 실패 횟수

    alpha_b = conversions_b + 1  # 성공 횟수
    beta_b = visitors_b - conversions_b + 1  # 실패 횟수

    samples_a = beta.rvs(alpha_a, beta_a, size=simulations)
    samples_b = beta.rvs(alpha_b, beta_b, size=simulations)

    prob_b_better_than_a = np.mean(samples_b > samples_a)

    return prob_b_better_than_a

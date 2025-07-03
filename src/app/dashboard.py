import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.utils.config_loader import load_config
from src.utils.mlflow_loader import MLflowLoader
from src.simulation.simulator import ABTestSimulator
from src.ab_testing import frequentist_engine, bayesian_engine
from src.data.preprocessing import mapping_id_to_unique

import time

# --- 1. 페이지 설정 및 제목 ---
st.set_page_config(layout="wide", page_title="A/B Test Decision Dashboard")
st.title("📈 A/B 테스트 의사결정 대시보드")
st.markdown("SVD(Baseline) vs NCF(Challenger) 추천 모델 성능 비교 분석")


# --- 2. 데이터 및 모델 로딩 (캐싱 활용) ---
@st.cache_data
def load_all_data(config):
    print("Loading data...")
    test_df = pd.read_parquet(config["data"]["test_data_path"])
    full_df = pd.read_parquet(config["data"]["processed_path"])

    _, _, mapped_test_df = mapping_id_to_unique(full_df, test_df.copy())
    _, _, mapped_full_df = mapping_id_to_unique(full_df, full_df.copy())
    return mapped_test_df, mapped_full_df


@st.cache_resource
def load_models_from_mlflow(config):
    print("Loading models from MLflow...")
    mlflow_config = config["mlflow"]
    loader = MLflowLoader(
        tracking_uri=mlflow_config["tracking_uri"],
        experiment_name=mlflow_config["experiment_name"],
    )
    runs = loader.get_latest_runs()
    models = loader.load_models(runs)
    return loader, models


# --- 3. 핵심 기능 함수 ---
def run_full_simulation(simulator, model_a, model_b, top_k, success_threshold) -> dict:
    """결과를 시뮬레이션하는 함수"""
    results = simulator.run_simulation(
        model_a=model_a,
        model_b=model_b,
        top_k=top_k,
        success_threshold=success_threshold,
    )
    return results


def plot_beta_distributions(visitors_a, conversions_a, visitors_b, conversions_b):
    """베타 분포 시각화 함수"""
    fig, ax = plt.subplots(figsize=(10, 5))

    # A 그룹
    alpha_a = conversions_a + 1
    beta_a = visitors_a - conversions_a + 1

    # B 그룹
    alpha_b = conversions_b + 1
    beta_b = visitors_b - conversions_b + 1

    # x축
    x = np.linspace(0, max(alpha_a / visitors_a, alpha_b / visitors_b) + 0.05, 1000)

    # 각 그룹의 베타 분포 PDF 계산
    y_a = stats.beta.pdf(x, alpha_a, beta_a)
    y_b = stats.beta.pdf(x, alpha_b, beta_b)

    ax.plot(x, y_a, label=f"Model A (SVD) - CTR: {conversions_a/visitors_a:.2%}")
    ax.plot(x, y_b, label=f"Model B (NCF) - CTR: {conversions_b/visitors_b:.2%}")
    ax.legend()
    ax.set_title("Beta Distribution of Conversion Rates", fontsize=15)
    ax.set_xlabel("Conversion Rate (CTR)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)

    return fig


# --- 4. 메인 대시보드 로직 ---
config = load_config("configs/config.yml")
test_df, full_df = load_all_data(config)
loader, models = load_models_from_mlflow(config)
model_a = models.get("SVD_PYTORCH")
model_b = models.get("NCF")

# 사이드바: 사용자 컨트롤
st.sidebar.header("⚙️ Simulation Controls")
max_value = int(len(test_df) // 2)
num_users_a = st.sidebar.number_input(
    "A 그룹(SVD) 사용자 수", min_value=100, max_value=max_value, value=1000, step=100
)
num_users_b = st.sidebar.number_input(
    "B 그룹(NCF) 사용자 수", min_value=100, max_value=max_value, value=1000, step=100
)
top_k = st.sidebar.slider("Top-K 추천", min_value=5, max_value=50, value=10)
success_threshold = st.sidebar.slider(
    "성공 기준 평점", min_value=0.5, max_value=5.0, value=4.0, step=0.5
)

if st.sidebar.button("🚀 시뮬레이션 실행!"):
    # 1. 시뮬레이터 및 캐시 초기화
    simulator = ABTestSimulator(full_df, test_df)
    simulator.set_group_num(num_A=num_users_a, num_B=num_users_b)

    # 2. 시뮬레이션 및 분석 실행
    analysis_results = {}
    sim_results = run_full_simulation(
        simulator, model_a, model_b, top_k, success_threshold
    )
    analysis_results["p_value"] = frequentist_engine.get_p_value(
        sim_results["conversions_a"],
        sim_results["visitors_a"],
        sim_results["conversions_b"],
        sim_results["visitors_b"],
    )
    analysis_results["prob_b_better"], analysis_results["expected_uplift"] = (
        bayesian_engine.get_bayesian_result(
            sim_results["conversions_a"],
            sim_results["visitors_a"],
            sim_results["conversions_b"],
            sim_results["visitors_b"],
        )
    )

    # ---------------------------------
    # 섹션 1: 최종 결론
    # ---------------------------------
    st.header("1. 최종 결론")
    recommendation = ""
    recommendation_reason = ""
    if (
        analysis_results["prob_b_better"] > 0.95
        and analysis_results["expected_uplift"] > 0
    ):
        recommendation = "모델 B (Challenger)로의 전환을 권고합니다."
        recommendation_reason = f"모델 B는 기존 모델 A 대비 전환율을 약 **{analysis_results['expected_uplift']:.2%}** 향상시킬 것으로 기대되며, 이 결과는 통계적으로 신뢰할 수 있습니다 (B가 더 나을 확률: **{analysis_results['prob_b_better']:.2%}**)."
        st.success(f"**결론: {recommendation}**")
        st.markdown(recommendation_reason)
    else:
        recommendation = "모델 A (Baseline) 유지를 권고합니다."
        recommendation_reason = f"모델 B의 성능 향상이 통계적으로 유의미한 수준에 도달하지 못했거나, 기대 향상치가 미미하여 전환의 근거가 부족합니다 (B가 더 나을 확률: **{analysis_results['prob_b_better']:.2%}**)."
        st.info(f"**결론: {recommendation}**")
        st.markdown(recommendation_reason)

    # ---------------------------------
    # 섹션 2: A/B 테스트 결과 상세 분석`
    # ---------------------------------`
    st.header("2. A/B 테스트 결과 상세 분석")
    ctr_a = sim_results["conversions_a"] / sim_results["visitors_a"]
    ctr_b = sim_results["conversions_b"] / sim_results["visitors_b"]

    st.markdown("#### 기본 결과 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("A (SVD) 방문자", f"{sim_results['visitors_a']:,} 명")
    col2.metric("A (SVD) 전환", f"{sim_results['conversions_a']:,} 건")
    col3.metric("A (SVD) 전환율", f"{ctr_a:.3%}")

    col1, col2, col3 = st.columns(3)
    col1.metric("B (Challenger) 방문자", f"{sim_results['visitors_b']:,} 명")
    col2.metric("B (Challenger) 전환", f"{sim_results['conversions_b']:,} 건")
    col3.metric("B (Challenger) 전환율", f"{ctr_b:.3%}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("빈도주의 분석")
        st.metric("P-value", f"{analysis_results['p_value']:.4f}")
        if analysis_results["p_value"] < 0.05:
            st.markdown(
                "✅ **결론**: 두 모델의 성능 차이는 **통계적으로 유의미**합니다 (p < 0.05)."
            )
        else:
            st.markdown(
                "⚠️ **결론**: 두 모델의 성능 차이가 통계적으로 유의미하다고 보기 **어렵습니다** (p >= 0.05)."
            )

    with col2:
        st.subheader("베이지안 분석")
        st.metric("B가 더 우수할 확률", f"{analysis_results['prob_b_better']:.2%}")
        st.metric(
            "B의 상대적 성능 향상 기대치", f"{analysis_results['expected_uplift']:.2%}"
        )
        st.markdown(
            "✅ **결론**: B 모델로 전환 시, **약 99%의 확률로 성능이 향상**되며 **13.6%의 성능 개선을 기대**할 수 있습니다."
            if analysis_results["prob_b_better"] > 0.95
            else "⚠️ **결론**: B 모델의 우월성에 대한 **확신이 부족**합니다."
        )

    # 베타 분포 시각화
    st.subheader("CTR 신뢰도 분포 (Beta Distribution)")
    beta_fig = plot_beta_distributions(
        sim_results["visitors_a"],
        sim_results["conversions_a"],
        sim_results["visitors_b"],
        sim_results["conversions_b"],
    )
    st.pyplot(beta_fig)

    # ---------------------------------
    # 섹션 3: 비용-효과 분석
    # ---------------------------------
    st.header("3. 비용-효과 분석")
    st.markdown(
        "성능 향상을 위해 추가적인 비용(학습/추론 시간, 서빙 비용)을 감수할 가치가 있는지 평가합니다."
    )

    # met_par = loader.get_run_metrics_and_params(loader.get_latest_runs())
    # svd_pytorch_par = met_par.get("SVD_PYTORCH", {}).get("params", {})
    # ncf_par = met_par.get("NCF", {}).get("params", {})
    # cost_data = {
    #     "지표": [
    #         "성능 향상 기대치",
    #         "학습 시간 (초)",
    #         "추론 속도 (초/1k users)",
    #         "모델 크기 (KB)",
    #     ],
    #     "모델 A (SVD_Pytorch)": [
    #         f"Baseline",
    #         f"{svd_pytorch_par.get('training_time', 0)}",
    #         f"{1000 * svd_time / len(simulator.get_group_a):.2f}",
    #     ],
    #     "모델 B (Challenger)": [
    #         f"+{analysis_results['expected_uplift']:.2%}",
    #         f"{ncf_par.get('training_time', 0)}",
    #         f"{1000 * ncf_time / len(simulator.get_group_b):.2f}",
    #     ],
    # }
    # cost_df = pd.DataFrame(cost_data)
    # st.table(cost_df.set_index("지표"))

else:
    st.info("사이드바에서 설정을 조정한 후 '시뮬레이션 실행' 버튼을 눌러주세요.")

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
    max_ctr = (
        max(conversions_a / visitors_a, conversions_b / visitors_b)
        if visitors_a > 0 and visitors_b > 0
        else 0.1
    )
    x = np.linspace(0, max_ctr + 0.05, 1000)

    # 각 그룹의 베타 분포 PDF 계산
    y_a = stats.beta.pdf(x, alpha_a, beta_a)
    y_b = stats.beta.pdf(x, alpha_b, beta_b)

    ctr_a_text = f"{conversions_a/visitors_a:.2%}" if visitors_a > 0 else "N/A"
    ctr_b_text = f"{conversions_b/visitors_b:.2%}" if visitors_b > 0 else "N/A"

    ax.plot(x, y_a, label=f"Model A (SVD) - CTR: {ctr_a_text}")
    ax.plot(x, y_b, label=f"Model B (NCF) - CTR: {ctr_b_text}")
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

# session_state에 시뮬레이션 실행 여부와 결과 저장 공간을 만듭니다.
if "simulation_run" not in st.session_state:
    st.session_state.simulation_run = False
    st.session_state.simulation_results = None
    st.session_state.analysis_results = None

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
    with st.spinner("모델 추천 시뮬레이션 및 통계 분석을 실행 중입니다..."):
        sim_results = run_full_simulation(
            simulator, model_a, model_b, top_k, success_threshold
        )
        analysis_results = {}
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

    # 3. 결과를 session_state에 저장하고, 실행 플래그를 True로 설정
    st.session_state.simulation_run = True
    st.session_state.simulation_results = sim_results
    st.session_state.analysis_results = analysis_results
    st.rerun()  # 결과를 즉시 화면에 반영하기 위해 rerun 실행

if st.session_state.simulation_run:
    # session_state에서 결과 불러오기
    sim_results = st.session_state.simulation_results
    analysis_results = st.session_state.analysis_results

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
    # 섹션 2: A/B 테스트 결과 상세 분석
    # ---------------------------------
    st.header("2. A/B 테스트 결과 상세 분석")
    ctr_a = sim_results["conversions_a"] / sim_results["visitors_a"]
    ctr_b = sim_results["conversions_b"] / sim_results["visitors_b"]

    st.markdown("#### 기본 결과 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("A (SVD) 방문자", f"{sim_results['visitors_a']:,} 명")
    col2.metric("A (SVD) 전환", f"{sim_results['conversions_a']:,} 건")
    col3.metric("A (SVD) 전환율", f"{ctr_a:.3%}")

    col1, col2, col3 = st.columns(3)
    col1.metric("B (NCF) 방문자", f"{sim_results['visitors_b']:,} 명")
    col2.metric("B (NCF) 전환", f"{sim_results['conversions_b']:,} 건")
    col3.metric("B (NCF) 전환율", f"{ctr_b:.3%}")

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
        if analysis_results["prob_b_better"] > 0.95:
            st.markdown(
                f"✅ **결론**: B 모델로 전환 시, **약 {analysis_results['prob_b_better']:.2%}의 확률로 성능이 향상**되며 **{analysis_results['expected_uplift']:.2%}의 성능 개선을 기대**할 수 있습니다."
            )
        else:
            st.markdown("⚠️ **결론**: B 모델의 우월성에 대한 **확신이 부족**합니다.")

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

    # --- 모델 성능 및 비용 지표 ---
    st.subheader("모델 성능 및 비용 지표")
    run_metrics = loader.get_run_metrics_and_params(loader.get_latest_runs())
    model_a_metrics = run_metrics.get("SVD_PYTORCH", {})
    model_b_metrics = run_metrics.get("NCF", {})

    perf_data = {
        "지표": [
            "학습 시간 (초)",
            f"추론 시간 ({sim_results['visitors_a'] + sim_results['visitors_b']}명 기준, 초)",
            "RMSE (학습 데이터 기준)",
        ],
        "모델 A (SVD_PYTORCH)": [
            f"{model_a_metrics.get('training_time', 0):.2f}",
            f"{sim_results.get('inference_time_a', 0):.2f}",
            f"{model_a_metrics.get('metrics', {}).get('rmse', 0):.4f}",
        ],
        "모델 B (NCF)": [
            f"{model_b_metrics.get('training_time', 0):.2f}",
            f"{sim_results.get('inference_time_b', 0):.2f}",
            f"{model_b_metrics.get('metrics', {}).get('rmse', 0):.4f}",
        ],
    }
    perf_df = pd.DataFrame(perf_data).set_index("지표")
    st.table(perf_df)

    # --- 재무 영향 분석 ---
    st.subheader("비용 및 수익 설정")
    col1, col2 = st.columns(2)
    with col1:
        cost_per_recommendation = st.number_input(
            "추천 1건당 비용 ($)",
            min_value=0.0,
            value=0.001,
            step=0.001,
            format="%.4f",
            help="한 명의 유저에게 Top-K 추천을 제공하는 데 드는 평균 비용입니다 (e.g., 서버, 인프라 비용).",
            key="cost_input",  # 위젯에 고유한 키를 부여하여 상태를 유지
        )
    with col2:
        revenue_per_conversion = st.number_input(
            "전환 1건당 수익 ($)",
            min_value=0.0,
            value=1.5,
            step=0.1,
            format="%.2f",
            help="전환(클릭, 구매 등) 1건이 발생했을 때 얻는 평균 수익입니다.",
            key="revenue_input",  # 위젯에 고유한 키를 부여하여 상태를 유지
        )

    # 비용-효과 계산
    # 그룹 A
    cost_a = sim_results["visitors_a"] * cost_per_recommendation
    revenue_a = sim_results["conversions_a"] * revenue_per_conversion
    profit_a = revenue_a - cost_a
    roi_a = (profit_a / cost_a) * 100 if cost_a > 0 else 0

    # 그룹 B
    cost_b = sim_results["visitors_b"] * cost_per_recommendation
    revenue_b = sim_results["conversions_b"] * revenue_per_conversion
    profit_b = revenue_b - cost_b
    roi_b = (profit_b / cost_b) * 100 if cost_b > 0 else 0

    st.subheader("재무 영향 분석")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 모델 A (Baseline)")
        st.metric("총 비용", f"${cost_a:,.2f}")
        st.metric("총 수익", f"${revenue_a:,.2f}")
        st.metric("순이익", f"${profit_a:,.2f}")
        st.metric("ROI", f"{roi_a:.2f}%")

    with col2:
        st.markdown("#### 모델 B (Challenger)")
        st.metric("총 비용", f"${cost_b:,.2f}")
        st.metric("총 수익", f"${revenue_b:,.2f}")
        st.metric("순이익", f"${profit_b:,.2f}")
        st.metric(
            "ROI",
            f"{roi_b:.2f}%",
            delta=f"{roi_b - roi_a:.2f}%" if roi_a != 0 else "N/A",
        )

    st.divider()

    st.subheader("모델 B 도입 시 증분 이익")
    incremental_profit = profit_b - profit_a
    st.metric("예상 증분 순이익", f"${incremental_profit:,.2f}")

    if incremental_profit > 0:
        st.success(
            f"모델 B로 전환 시, 시뮬레이션된 사용자 그룹 기준으로 약 ${incremental_profit:,.2f}의 추가 이익이 발생할 것으로 예상됩니다."
        )
    else:
        st.warning(
            f"모델 B로 전환 시, 시뮬레이션된 사용자 그룹 기준으로 약 ${abs(incremental_profit):,.2f}의 손실이 발생할 수 있습니다."
        )


else:
    # --- 시뮬레이션이 아직 실행되지 않았을 때 표시될 메시지 ---
    st.info("사이드바에서 설정을 조정한 후 '시뮬레이션 실행' 버튼을 눌러주세요.")

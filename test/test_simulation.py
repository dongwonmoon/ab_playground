import pandas as pd

from src.utils.config_loader import load_config
from src.utils.mlflow_loader import MLflowLoader
from src.simulation.simulator import ABTestSimulator
from src.ab_testing import frequentist_engine
from src.ab_testing import bayesian_engine


def main():
    """
    A/B 테스트 시뮬레이션 전체 파이프라인을 실행하고 결과를 출력합니다.
    """
    # --- 1. 설정 로드 ---
    print("Step 1: Loading configuration...")
    config = load_config("configs/config.yml")
    mlflow_config = config["mlflow"]
    sim_config = config["simulation"]

    # --- 2. 데이터 로드 ---
    print("Step 2: Loading data...")
    train_df = pd.read_parquet(config["data"]["train_data_path"])
    test_df = pd.read_parquet(config["data"]["test_data_path"])
    full_df = pd.read_parquet(config["data"]["processed_path"])

    # --- 3. MLflow에서 모델 로드 ---
    print("Step 3: Loading models from MLflow...")
    # MLflowLoader를 사용하여 SVD (A 모델)와 NCF (B 모델)를 불러옵니다.
    loader = MLflowLoader(
        tracking_uri=mlflow_config["tracking_uri"],
        experiment_name=mlflow_config["experiment_name"],
    )
    runs = loader.get_latest_runs()
    models = loader.load_models(runs)

    model_a = models.get("SVD_PYTORCH")  # A 모델은 SVD
    model_b = models.get("SVD")  # B 모델은 NCF

    if not model_a or not model_b:
        raise ValueError("Failed to load one or both models. Aborting.")

    # --- 4. 시뮬레이션 실행 ---
    print("\nStep 4: Running A/B test simulation...")
    # ABTestSimulator를 초기화하고 시뮬레이션을 실행합니다.
    simulator = ABTestSimulator(full_df=train_df, test_df=test_df)

    # 대시보드에서 입력받을 값을 임시로 지정합니다.
    # 예시: 총 10,000명의 사용자를 A/B 그룹에 50:50으로 할당
    simulator.set_group_num(num_A=1000, num_B=1000)

    simulation_results = simulator.run_simulation(
        model_a=model_a,
        model_b=model_b,
        top_k=sim_config["top_k"],
        success_threshold=sim_config["success_threshold"],
    )
    print("Simulation complete.")
    print("Results:", simulation_results)

    # --- 5. 통계 분석 수행 ---
    print("\nStep 5: Performing statistical analysis...")
    # 빈도주의 분석
    p_value = frequentist_engine.get_p_value(
        visitors_a=simulation_results["visitors_a"],
        conversions_a=simulation_results["conversions_a"],
        visitors_b=simulation_results["visitors_b"],
        conversions_b=simulation_results["conversions_b"],
    )

    # 베이지안 분석
    prob_b_better = bayesian_engine.get_bayesian_result(
        visitors_a=simulation_results["visitors_a"],
        conversions_a=simulation_results["conversions_a"],
        visitors_b=simulation_results["visitors_b"],
        conversions_b=simulation_results["conversions_b"],
    )
    print("Analysis complete.")

    # --- 6. 최종 결과 출력 ---
    print("\n" + "=" * 50)
    print(" A/B Test Simulation Final Report")
    print("=" * 50)

    # 그룹별 전환율(CTR) 계산
    ctr_a = simulation_results["conversions_a"] / simulation_results["visitors_a"]
    ctr_b = simulation_results["conversions_b"] / simulation_results["visitors_b"]

    print(f"\n[Raw Data]")
    print(
        f"  - Group A (SVD): {simulation_results['conversions_a']} / {simulation_results['visitors_a']} (CTR: {ctr_a:.4f})"
    )
    print(
        f"  - Group B (NCF): {simulation_results['conversions_b']} / {simulation_results['visitors_b']} (CTR: {ctr_b:.4f})"
    )

    print("\n[Frequentist Analysis]")
    print(f"  - P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  - Conclusion: The difference is statistically significant. (p < 0.05)")
    else:
        print(
            "  - Conclusion: The difference is not statistically significant. (p >= 0.05)"
        )

    print("\n[Bayesian Analysis]")
    print(f"  - Probability B is better than A: {prob_b_better:.2%}")
    # print(f"  - Expected Uplift from B: {expected_uplift:.2%}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()

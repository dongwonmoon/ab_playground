# Recommendation System A/B Testing Simulation Platform

## 0. Data
데이터는 MovieLens-tiny 데이터를 사용했습니다.
A/B 시뮬레이션을 해야하지만, 실제 사용자를 구하기는 힘드므로, 데이터를 적절히 Split하여 사용했습니다.
자세한 내용은 `notebooks` 디렉토리를 참조해주세요.

## 1. 프로젝트 개요

본 프로젝트는 추천 시스템 모델의 성능을 평가하고 비교하기 위한 A/B 테스트 시뮬레이션 및 분석 플랫폼입니다.

단순한 오프라인 지표(e.g., RMSE, NDCG)만으로는 실제 비즈니스 임팩트를 예측하기 어렵다는 문제에서 출발했습니다. 이 플랫폼은 두 개의 추천 모델(A: Control, B: Treatment)이 실제 서비스 환경에 배포되었을 때 발생할 수 있는 사용자 행동을 시뮬레이션하고, 그 결과를 통계적으로 분석하여 어떤 모델이 더 우수한 성과를 내는지 데이터 기반으로 검증할 수 있도록 지원합니다.

MLflow를 통해 모델 학습과 실험을 체계적으로 관리하며, Streamlit 기반의 인터랙티브 대시보드를 통해 시뮬레이션 결과를 쉽게 이해하고 분석할 수 있습니다.

## 2. 주요 기능 및 특징

- **다양한 추천 모델 지원**:
    - **SVD (Singular Value Decomposition)**: `Surprise` 라이브러리의 SVD는 추론 속도가 느린 단점이 있어, 이를 **PyTorch 백엔드로 재구현하여 추론 성능을 크게 향상**시켰습니다. (`src/models/svd_pytorch.py`)
    - **NCF (Neural Collaborative Filtering)**
- **A/B 테스트 시뮬레이션**:
    - 두 개의 모델(Control/Treatment)에 대한 가상 사용자 반응(클릭 등) 데이터 생성
- **정교한 통계 분석 엔진**:
    - **빈도주의(Frequentist) 분석**: Z-test를 이용한 그룹 간 성과 차이의 통계적 유의성 검증
    - **베이지안(Bayesian) 분석**: B 모델이 A 모델보다 우수할 확률 등 직관적인 결과 제공
- **실험 관리 및 재현성**:
    - MLflow를 활용한 모델, 파라미터, 결과 지표 로깅 및 관리
- **인터랙티브 시각화 대시보드**:
    - Streamlit을 통해 A/B 테스트 시뮬레이션 결과를 시각화하고 분석

## 3. 프로젝트 구조

```
.
├── configs/
│   └── config.yml         # 프로젝트의 모든 설정을 관리
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_make_ab_test_data.ipynb
├── scripts/
│   ├── run_training.py    # 모델 학습 스크립트
│   └── run_dashboard.py   # Streamlit 대시보드 실행 스크립트
├── src/
│   ├── ab_testing/        # A/B 테스트 통계 분석 엔진
│   │   ├── bayesian_engine.py
│   │   └── frequentist_engine.py
│   ├── app/
│   │   └── dashboard.py     # Streamlit 대시보드 UI 및 로직
│   ├── data/              # 데이터 처리 및 데이터셋 관리
│   │   ├── datasets.py
│   │   └── preprocessing.py
│   ├── models/            # 추천 모델 구현체
│   │   ├── ncf.py
│   │   ├── svd.py
│   │   └── svd_pytorch.py   # 성능 개선된 PyTorch 기반 SVD
│   ├── simulation/        # A/B 테스트 시뮬레이터
│   │   └── simulator.py
│   └── utils/             # 유틸리티 (설정 로더 등)
│       └── config_loader.py
│       └── mlflow_loader.py
├── requirements.txt       # Conda 환경용 패키지 목록
└── README.md
```

## 4. 워크플로우

1.  **데이터 준비 및 전처리** (`notebooks/`): 초기 데이터 탐색(EDA) 및 모델 학습에 필요한 데이터셋을 준비합니다.
2.  **모델 학습** (`scripts/run_training.py`): `configs/config.yml`에 정의된 하이퍼파라미터를 바탕으로 추천 모델들을 학습시킵니다. 학습된 모델과 관련 정보는 MLflow에 저장됩니다.
3.  **A/B 테스트 시뮬레이션 및 분석** (`src/simulation/`): MLflow에서 Control 모델과 Treatment 모델을 불러옵니다. 시뮬레이터를 통해 두 모델의 추천 결과에 대한 가상 사용자 로그를 생성하고, `ab_testing` 엔진으로 통계 분석을 수행합니다.
4.  **결과 시각화** (`scripts/run_dashboard.py`): Streamlit 대시보드를 실행하여 시뮬레이션 결과를 확인합니다. 사용자는 대시보드에서 Control/Treatment 모델을 직접 선택하고, 그에 따른 분석 결과를 인터랙티브하게 탐색할 수 있습니다.

## 5. 설치 및 실행 방법

### 5.1. 설치

**중요**: 본 프로젝트는 **Conda** 환경에 최적화되어 있습니다. `requirements.txt`는 `conda list --export` 명령어로 생성되었습니다.

```bash
conda create --name ab_playground python=3.9
conda activate ab_playground

conda install --file requirements.txt
```

### 5.2. 실행

**중요**: 프로젝트의 모든 스크립트는 모듈 경로 문제를 피하기 위해 프로젝트 루트 디렉토리에서 `python -m` 옵션을 사용하여 실행해야 합니다.

#### MLFlow 서버
```bash
mlflow server \
    --backend-store-uri sqlite:///mlruns_db/mlflow.db \
    --default-artifact-root ./mlruns_artifacts \
```

#### 모델 학습

`config.yml` 파일에서 원하는 모델과 파라미터를 설정한 후, 아래 명령어를 실행하여 모델 학습을 진행합니다. 학습 과정과 결과는 MLflow UI를 통해 확인할 수 있습니다.

```bash
python -m scripts.run_training
```

#### 대시보드 실행

아래 명령어를 실행하여 A/B 테스트 분석을 위한 Streamlit 대시보드를 실행합니다.

```bash
python -m scripts.run_dashboard
```

## 6. 주요 의존성

- `pandas` & `numpy`
- `scikit-learn`
- `pytorch`
- `mlflow`
- `streamlit`
- `plotly`
- `pyyaml`

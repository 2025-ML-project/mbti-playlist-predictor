mbti-playlist-predictor/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── 기능-요청.md
│   │   └── 버그-리포트.md
│   └── pull_request_template.md
│
├── 📂 notebooks/                               # 모델 실험 및 분석 노트북
│   ├── LSTM+Dual.ipynb                         # LSTM + Dual Input 모델 실험
│   ├── cnn+base+model.ipynb                    # CNN Base Model
│   ├── mbti_four_binary_model.ipynb            # 4축 분리(binary) 모델
│   ├── mbti_multitask_model_base_final.ipynb   # Multitask CNN 최종 모델
│   ├── multi_label_test.ipynb
│   └── weighted_binary_crossentropy.ipynb      # 가중치 적용 손실 실험
│
├── 📂 src/                                      # 데이터 처리/모델 입력용 스크립트
│   ├── tf-idf/
│   │   └── fixed_length_playlist.py
│   │
│   ├── 데이터가져오기/
│   │   ├── playlist_track_extract.py
│   │   └── playlist_track_extract_loop.py
│   │
│   ├── 데이터임베딩/
│   │   ├── error_feature_extractor.py
│   │   ├── feature_cleaner.py
│   │   ├── feature_extractor_cache.py
│   │   ├── playlist_track_extract.py
│   │   └── tensor_generator.py
│   │
│   ├── 데이터필터링/
│   │   ├── drop_playlist_by_40.py
│   │   ├── drop_playlist_by_40_save.py
│   │   └── filtering.py
│   │
│   └── 시각화/
│       ├── playlist_barchart.py
│       ├── track_count_boxplot.py
│       ├── visualiise-count.py
│       └── visualiise-track-count.py
│
├── .gitignore
└── README.md


# Dual-Input LSTM MBTI Predictor
> 기존 단일 모델의 한계를 극복하기 위해 LSTM과 Dual-Input 구조를 결합한 모델입니다.

<br>

## 1. 기존모델 개선
### 왜 LSTM + Dual Input인가?
기존 모델(CNN)의 성능 한계를 되짚어 보면서, 두 가지 생각을 했습니다.
1.  순서의 중요성: 플레이리스트는 단순한 집합이 아니라 흐름(Sequence)이 존재합니다.
2.  종합적 취향: 개별 곡뿐만 아니라 플레이리스트 전체적인 분위기도 중요합니다.

=> 따라서, 시계열 패턴(LSTM)과 전체적인 통계 특성(Static)을 동시에 학습하는 모델을 생각했습니다.

<br>

## 2. 모델 구조
두 가지의 입력을 받아 결합하는 Dual-Input 구조



### Input A: 시퀀스 데이터 (Sequence)
* Input: `(60, 10)` 형태의 시계열 데이터 (최근 60곡의 흐름)
* Architecture:
    * `Conv1D`: 지역적인 음악적 패턴(Local Pattern) 추출
    * `Bidirectional LSTM`: 플레이리스트의 앞뒤 맥락(Context)을 양방향으로 학습
* Regularization: `BatchNormalization`, `MaxPooling1D`, `Dropout`을 적용하여 과적합 방지

### Input B: 정적 요약 데이터 (Static)
* Input: RandomForest로 추출한 **Top 5 핵심 특성**의 평균값
* Architecture: 
    * `Dense Layer`: 완전 연결층을 통해 특징을 압축
* 사용자가 선호하는 전반적인 음악 분위기 학습

<br>

## 3. 주요 특성 및 전처리
학습 효율성을 위해 `RandomForestClassifier`를 활용하여 MBTI 예측에 가장 기여도가 높은 Top 5 특성을 추출했습니다. (SHAP 대비 교차 검증 시 속도 개선)

| 순위 | 특성 (Feature) | 설명 |
| :-- | :-- | :-- |
| 1 | **Happiness** | 음악이 주는 행복감/긍정적 분위기 |
| 2 | **Acousticness** | 어쿠스틱 악기 사용 비중 |
| 3 | **Speechiness** | 보컬/말소리의 비중 |
| 4 | **Danceability** | 춤추기 좋은 리듬감 |
| 5 | **Loudness** | 음악의 데시벨/크기 |

> `Happiness`와 `Acousticness`가 상위에 랭크된 것으로 보아, "음악의 정서적 분위기"가 성격을 대변하는 주요 요인임을 확인했습니다.

<br>

## 4. 학습 전략
모델의 일반화 성능을 높이고 과적합을 방지하기 위해 다음과 같은 전략을 사용했습니다.

* Stratified K-Fold: 클래스 불균형을 고려하여 데이터 분포를 유지하며 5-Fold 교차 검증 수행
* Label Smoothing (0.1): 정답에 대한 확신을 낮춰 과적합 방지 및 일반화 성능 향상
* Callbacks:
    * `EarlyStopping`: 검증 손실(val_loss)이 10회 이상 개선되지 않으면 학습 조기 종료
    * `ReduceLROnPlateau`: 학습 정체 시 Learning Rate를 0.5배로 동적으로 감소

<br>

## 5. 평가지표
* accuracy: 정확도
* top3_acc: 상위 3개 예측 확률 내에 정답이 포함될 확률(터무니 없는 예측을 하는지 확인하기 위함)

<br>

## 6. 실험 결과

### 검증(Validation) vs 테스트(Test) 성능

| 구분 | Accuracy (Top-1) | Top-3 Accuracy | 비고 |
| :-- | :-- | :-- | :-- |
| **Validation (Avg)** | 46.4% | 76.2% | K-Fold 평균 |
| **Test Data** | 43.6% | 71.1% | 최종 테스트 |

> 결과 해석: 검증 데이터 결과와 테스트 데이터 결과가 큰 오차가 없는 걸로 보아 안정적인 일반화 성능은 확보했으나, 정확도 측면에서 43.6%로 아쉬움을 보였습니다.

<br>

## 7. 한계점 및 분석
각 데이터에 대한 혼동 행렬을 통해 다음과 같은 문제점을 발견했습니다.

### 1) S vs N 데이터 불균형
* N(직관형) 그룹에 비해 S(감각형) 그룹의 예측 정확도가 현저히 떨어짐.
* 원인: 학습 데이터에 N 유형이 훨씬 많아, 모델이 애매할 경우 다수 클래스인 'N'으로 예측하려는 편향 발생.

### 2) 인접 유형 간의 혼동
* 완전히 반대되는 성향보다는, 4글자 중 알파벳이 하나만 다른 "이웃 유형"끼리 혼동하는 경향이 큼.
* top3_acc의 정확도가 71.1%가 나온 걸로 보아 mbti 4개 영역 중 하나만 혼동하는 걸 알 수 있음

<br>

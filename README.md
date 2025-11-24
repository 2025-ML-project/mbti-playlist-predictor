# 🎧 mbti-playlist-predictor

> Spotify 플레이리스트의 음악적 특성  
> (tempo, happiness, energy, acousticness 등)을 기반으로  
> 사용자의 **MBTI 유형 (E/I, S/N, T/F, P/J)** 을 예측하는 머신러닝 프로젝트입니다.

본 프로젝트는 CNN, LSTM, Dual-Input, Multitask 모델 등 다양한 딥러닝 구조를 실험하며
**“음악 취향과 성격 간의 관계”** 를 탐구하였습니다.

# 🎶 문제정의
- 아이디어 : 사용자의 음악 취향을 통해 성격적 특성을 예측할 수 있을까?
- 목표 : 사용자가 즐겨듣는 음악의 특성(audio features)을 통해 MBTI 성격 유형을 에측하는 모델을 구축하기

# 🛠 개발 환경

| 항목 | 내용 |
|------|------|
| **OS / Hardware** | Google Colab (GPU T4) |
| **IDE** | VSCode, Google Colab |
| **ML Frameworks** | TensorFlow / Keras, Scikit-learn |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | SHAP |
| **Version Control** | Git / GitHub |


# 🎵 Model Overview
총 3가지 모델을 소개하며,
1 → 2 → 3 모델의 순서대로 앞선 모델의 한계점을 해결할 수 있도록 단계적으로 모델을 구축하였습니다.

1. **CNN Base Model**  
   - 곡 시퀀스(sequence)를 1D convolution으로 처리하여  
     MBTI *전체 16개 유형*을 한 번에 분류하는 모델
     
2. **LSTM + Dual Input Model**  
   - 플레이리스트의 흐름(sequence)을 LSTM으로 읽고  
   - SHAP 기반 Top-5 핵심 특성의 *평균값*을 별도 입력으로 넣어  
     두 정보를 결합하여 예측하는 모델

3. **Multitask CNN Model — 최종 선정 모델**  
   - MBTI를 16개를 한 번에 맞추는 대신  
     **4가지 축(E/I, S/N, T/F, P/J)을 각각 예측**하여  
     예측 성능과 안정성을 크게 높인 모델

## [1] CNN Base Model
> CNN이 곡의 feature sequence를 통해 MBTI 전체를 예측하는 단일 다중분류 모델입니다.

<br>

### ① Input & Target
- Input : audio features (n,60,11)
- Target : MBTI 유형을 하나의 클래스로 취급하여, one-hot-encoding

<br>

### ② Model Architecture
#### 1. 모델1 (base model)
#### 2. 모델2 (base model + Early Stopping 적용)
#### 3. 모델3 (dropout 비율 조정)
#### 4. 모델4 (deeper architecture + optimizer=’rmsprop’)
#### 5. 모델5 (modified architecture)

CNN Base Model 구축 시에는, 다양한 하이퍼파라미터 조정 및 optimizer 변경의 다양한 실험을 진행하였습니다.

<br>

### ③ 성능 및 평가
| 항목                | 설명 |
|--------------------|------|
| **Training Loss**       | 계속해서 감소 |
| **Validation Loss**     | 감소하다가 약 10 epoch 이후 증가 → 과대적합 시작 |
| **Training Accuracy**   | 계속해서 증가 |
| **Validation Accuracy** | 증가하다가 일정 수준 이상으로는 올라가지 않음 |

> 테스트 정확도는 0.365

<br>

### ④ 한계점 및 분석
1. cnn base model은 하이퍼파라미터, 모델 구조 개선을 해도 정확도 40%를 넘기지 못했음
2. 이는 MBTI는 4개의 독립적인 축으로 이루어져있다. → 한 글자라도 틀리면 완전 실패하기 때문
3. SHAP의 결과인 상위 5개 주요 특성을 조금 더 반영할 수 있도록 모델을 개선하기 위해서 lstm+dual input model을 만들어 보았음

<br>
 
## [2] Dual-Input LSTM MBTI Predictor
> 기존 단일 모델의 한계를 극복하기 위해 LSTM과 Dual-Input 구조를 결합한 모델입니다.

<br>

### ① 기존모델 개선
#### 왜 LSTM + Dual Input인가?
기존 모델(CNN)의 성능 한계를 되짚어 보면서, 두 가지 생각을 했습니다.
1.  순서의 중요성: 플레이리스트는 단순한 집합이 아니라 흐름(Sequence)이 존재합니다.
2.  종합적 취향: 개별 곡뿐만 아니라 플레이리스트 전체적인 분위기도 중요합니다.

=> 따라서, 시계열 패턴(LSTM)과 전체적인 통계 특성(Static)을 동시에 학습하는 모델을 생각했습니다.

<br>

### ② Model Architecture
두 가지의 입력을 받아 결합하는 Dual-Input 구조


#### Input A: 시퀀스 데이터 (Sequence)
* Input: `(60, 10)` 형태의 시계열 데이터 (최근 60곡의 흐름)
* Architecture:
    * `Conv1D`: 지역적인 음악적 패턴(Local Pattern) 추출
    * `Bidirectional LSTM`: 플레이리스트의 앞뒤 맥락(Context)을 양방향으로 학습
* Regularization: `BatchNormalization`, `MaxPooling1D`, `Dropout`을 적용하여 과적합 방지

#### Input B: 정적 요약 데이터 (Static)
* Input: RandomForest로 추출한 **Top 5 핵심 특성**의 평균값
* Architecture: 
    * `Dense Layer`: 완전 연결층을 통해 특징을 압축
* 사용자가 선호하는 전반적인 음악 분위기 학습

<br>

### ③ 주요 특성 및 전처리
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

### ④ 학습 전략
모델의 일반화 성능을 높이고 과적합을 방지하기 위해 다음과 같은 전략을 사용했습니다.

* Stratified K-Fold: 클래스 불균형을 고려하여 데이터 분포를 유지하며 5-Fold 교차 검증 수행
* Label Smoothing (0.1): 정답에 대한 확신을 낮춰 과적합 방지 및 일반화 성능 향상
* Callbacks:
    * `EarlyStopping`: 검증 손실(val_loss)이 10회 이상 개선되지 않으면 학습 조기 종료
    * `ReduceLROnPlateau`: 학습 정체 시 Learning Rate를 0.5배로 동적으로 감소

<br>

### ⑤ 평가지표
* accuracy: 정확도
* top3_acc: 상위 3개 예측 확률 내에 정답이 포함될 확률(터무니 없는 예측을 하는지 확인하기 위함)

<br>

### ⑥ 실험 결과

#### 검증(Validation) vs 테스트(Test) 성능

| 구분 | Accuracy (Top-1) | Top-3 Accuracy | 비고 |
| :-- | :-- | :-- | :-- |
| **Validation (Avg)** | 46.4% | 76.2% | K-Fold 평균 |
| **Test Data** | 43.6% | 71.1% | 최종 테스트 |

> 결과 해석: 검증 데이터 결과와 테스트 데이터 결과가 큰 오차가 없는 걸로 보아 안정적인 일반화 성능은 확보했으나, 정확도 측면에서 43.6%로 아쉬움을 보였습니다.

<br>

### ⑦ 한계점 및 분석
각 데이터에 대한 혼동 행렬을 통해 다음과 같은 문제점을 발견했습니다.

#### 1) S vs N 데이터 불균형
* N(직관형) 그룹에 비해 S(감각형) 그룹의 예측 정확도가 현저히 떨어짐
* 원인: 학습 데이터에 N 유형이 훨씬 많아, 모델이 애매할 경우 다수 클래스인 'N'으로 예측하려는 편향이 발생

#### 2) 인접 유형 간의 혼동
* 완전히 반대되는 성향보다는, 4글자 중 알파벳이 하나만 다른 "이웃 유형"끼리 혼동하는 경향이 큼
* top3_acc의 정확도가 71.1%가 나온 걸로 보아 mbti 4개 영역 중 하나만 혼동하는 걸 알 수 있음

<br>

## [3] CNN Multitask Base Model
> CNN이 곡의 feature sequence를 보고 MBTI 4축(ei/sn/ft/pj) 를 각각 동시에 예측하는 모델입니다.

<br>

### ① y-labeling
- y_raw는 하나의 플레이리스트에 대응하는 하나의 full mbti를 가지고 있음
- multitask 방식을 이용하기 위해서 y의 shape을 (n,4)로 변환
- (E=1, I=0) / (N=1, S=0) / (T=1, F=0) / (T=1, F=0)

<br>

### ② Model Architecture
- Base CNN 모델에서 dense layer의 마지막 output을 4개의 노드를 가지며, sigmoid activation function을 사용

<br>

### ③ 성능 및 평가
| 항목 | 의미 |
|------|------|
| **Test accuracy (model.evaluate)** | 4축 accuracy 평균 |
| **EI/SN/FT/PJ accuracy** | 축별로 맞춘 비율 |
| **Full accuracy** | 4글자 모두 맞춘 비율 |
| **Partial accuracy** | 샘플당 4축 중 몇 개 맞았는지 평균 |

> 테스트 정확도는 0.744

<br>

### ④ 의의
- MBTI를 16개의 완전한 문자열로 보기보다, 심리학적 구조대로 4개의 독립적인 이진 분류의 조합으로 파악함
- 4축 Multitask 모델로 구현한 결과, 모델 전체 성능은 0.744로 앞선 두 모델의 2배에 달함
- 이는 4개의 축을 각각 예측하기 때문이며, 대략적으로 4글자 중 3글자는 맞춘다는 것을 의미함
- 타모델에 비해 MBTI 해석 측면에서 유용한 점은, 축별 정확도를 계산할 수 있다는 점이다.

# ⭐️ 최종 결론

# 📁 Project Structure
```text
mbti-playlist-predictor/
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── 기능-요청.md
│   │   └── 버그-리포트.md
│   └── pull_request_template.md
│
├── 📂 notebooks/               # 모델 실험 및 분석 노트북
│   ├── LSTM+Dual.ipynb         # LSTM + Dual Input 모델 실험
│   ├── cnn+base+model.ipynb    # CNN Base Model
│   ├── mbti_four_binary_model.ipynb   # 4축 Binary 모델
│   ├── mbti_multitask_model_base_final.ipynb   # Multitask CNN 최종 모델
│   ├── multi_label_test.ipynb
│   └── weighted_binary_crossentropy.ipynb       # 가중치 손실 실험
│
├── 📂 src/                     # 데이터 처리 및 전처리 스크립트
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
```

# 🧑‍💻 Contributors
| 이름 | 역할 |
|------|------|
| 조선환 [@whtjsghks](https://github.com/whtjsghks) | 전반적인 모델링 과정 전담|
| 김예송 [@optiprime27](https://github.com/optiprime27) | 데이터 임베딩, 추가적인 모델링 |
| 박소연 [@soyeoneeii](https://github.com/soyeoneeii) | 데이터 전처리, 추가적인 모델링 |
| 정수원 [@go-wt-flow](https://github.com/go-wt-flow) | 전반적인 모델링 과정 전담|



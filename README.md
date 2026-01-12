# Tense-Aware Sentiment Forecasting

### 시제 분리 감성 정보를 활용한 주가 예측 성능 비교 연구

---

## 1. 연구 배경 및 동기 (Background & Motivation)

**문제 인식**

전통적인 감성 분석 기반 주가 예측 모델은 뉴스 텍스트에서 추출한 감성 점수를 단일 지표로 통합하여 사용한다. 그러나 이는 다음과 같은 한계를 가진다:

- 시간적 정보의 손실: "작년 실적 부진(-), 올해 대규모 수주 예정(+)"과 같이 
  상반된 시제 정보가 혼재된 뉴스의 경우, 단일 점수로는 미묘한 뉘앙스를 포착할 수 없음
- 시장 효율성 미반영: EMH에 따라 과거 사건과 미래 전망은 시장에 반영되는 속도가 다르나, 기존 모델은 이를 구분하지 않음

### 연구 질문
**"뉴스 감성 정보를 시제별로 분리하면 주가 예측 성능이 향상되는가?"**

---

## 2. 방법론 (Methodology)

**분석 대상 및 기간**

- 대상: 국내 **방산 섹터** 주요 4개 기업 (HD현대중공업, LIG넥스원, 한국항공우주, 한화시스템)
  - 선정 이유: 수주 기반 산업 특성상 미래 지향적 정보의 중요도가 타 섹터 대비 높음
- 기간: **2022-01-02 ~ 2024-12-30**

**주요 피처 엔지니어링 (Feature Engineering)**

- Common price features
  * `candle_body_length` = (`Close` − `Open`) / `Open`
  * `candle_high_low_length` = (`High` − `Low`) / `Close`

- Group A (Baseline)
  * `score_total`: 시제 구분 없이 통합된 전체 감정 지수

- Group B (Experimental)

  - 과거/현재 시제 감성 점수 (`score_past`) 
  
   $$ \frac{(\sum Past_{pos} + \sum Present_{pos}) - (\sum Past_{neg} + \sum Present_{neg})}{(\sum Past_{pos} + \sum Present_{pos}) + (\sum Past_{neg} + \sum Present_{neg})} $$

  - 미래 시제 감성 점수 (`score_future`)

   $$ \frac{\sum Future_{pos} - \sum Future_{neg}}{\sum Future_{pos} + \sum Future_{neg}} $$

**실험 설계**

- 수익률 기반 라벨링
  - 1일 후 수익률이 2% 이상: label 1
  - 1일 후 수익률이 2% 미만: label 0

- 2% 임계값 설정 근거:
  - 거래 비용(증권거래세 + 수수료) 고려 시 실질적 수익 창출 가능 구간

**하이퍼파라미터 최적화**
- 방법: Optuna를 활용하여 피처 그룹별로 최적의 하이퍼파라미터를 독립적으로 탐색
- 최적화 대상: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`
- 목적 함수: MCC 최대화

**평가 지표 (Evaluation Metrics)**

금융 데이터의 특성상 발생하는 클래스 불균형(Class Imbalance) 문제를 고려하여 다음 지표를 핵심 평가지표로 활용:

* **Precision (정밀도)**: 모델이 *상승*으로 예측한 사례 중 실제 상승 비율을 측정하며, 잘못된 매수 신호(False Positive)를 억제하는 능력을 평가

* **F1-Score**: 전반적인 분류 성능을 평가

* **MCC (Matthews Correlation Coefficient)**: 실제 값과 예측 값 사이의 상관 구조를 반영

---

## 3. 데이터 신뢰성 및 통계적 검증 (Data Validation)

- **독립성 검정**: `score_past`와 `score_future` 간의 상관계수(평균 0.6398)를 확인하여, 두 지표가 다중공선성 문제 없이 서로 독립적인 신호를 제공함을 검증.
- **데이터 분포 최적화**: **Jarque–Bera Test**로 왜도와 첨도를 분석한 후, **Yeo–Johnson 변환** 및 **Winsorizing**을 통해 이상치 영향을 최소화.
- **시계열 정상성 확인**: **ADF(Augmented Dickey-Fuller) Test**를 통해 모든 피처의 정상성 검증.

---

## 4. 결과 요약 (Key Results)

| Evaluation Metric | Avg. Group A |  Avg. Group B  | Avg. $\Delta$ |
|:-----------------:|:------------:|:--------------:|:-------------:|
|     Precision     |    0.241     |   **0.280**    |    +0.039     |
|     F1-Score      |    0.271     |   **0.294**    |    +0.023     |
|        MCC        |    -0.012    |   **0.053**    |    +0.065     |

**주요 통찰 (Insights)**

1. **미래 지향적 정보의 우위성**: Feature Importance 분석 결과, `Score_future`가 4개 종목 전부 `Score_past`보다 높은 기여도 보임
2. **Precision 향상을 통한 리스크 관리**: '상승' 신호에 대한 오탐(False Positive)을 줄임으로써 실전 트레이딩 시 잘못된 매수 진입을 억제하는 효과 기대
3. **MCC의 유의미한 상승**: 모든 종목에서 MCC가 평균적으로 가장 큰 폭으로 개선되었다. 이는 시제 분리가 예측값과 실제값 사이의 상관 구조를 더 정교하게 포착함을 증명

<details>
  <summary>🔍 여기를 클릭해서 상세 실험 결과 확인</summary>
  
  ### 실험 결과 요약

  1. HD 현대중공업
  
  |  Metric   |  Group A  |  Group B  | $\Delta$ |
  |:---------:|:---------:|:---------:|:--------:|
  | Precision |   0.250   | **0.263** |  +0.013  |
  | F1-Score  | **0.282** |   0.266   |  -0.016  |
  |    MCC    |  -0.002   | **0.015** |  +0.017  |

  2. LIG넥스원
  
 |  Metric   | Group A |  Group B  | $\Delta$ |
 |:---------:|:-------:|:---------:|:--------:|
 | Precision |  0.200  | **0.274** |  +0.074  |
 | F1-Score  |  0.261  | **0.318** |  +0.057  |
 | MCC       | -0.113  | **0.038** |  +0.151  |
  
  3. 한국항공우주
  
 |  Metric   | Group A |  Group B   | $\Delta$ |
 |:---------:|:-------:|:----------:|:--------:|
 | Precision |  0.190  | **0.230**  |  +0.04   |
 | F1-Score  |  0.228  | **0.268**  |  +0.04   |
 | MCC       |  0.000  | **0.061**  |  +0.061  |

  4. 한화시스템
  
 |  Metric   | Group A |  Group B  | $\Delta$ |
 |:---------:|:-------:|:---------:|:--------:|
 | Precision |  0.324  | **0.352** |  +0.028  |
 | F1-Score  |  0.311  | **0.324** |  +0.013  |
 | MCC       |  0.068  | **0.099** |  +0.031  |

</details>

---

## 5. 재현성 (Reproducibility)

**환경**:
- Python 3.10+ 
- 필수 라이브러리: `requirements.txt` 참조
- Random seed 고정

---

## 6. 한계 및 향후 연구

**방법론적 한계**

1. **표본 편항**
- 방산 섹터만 분석하여 일반화 가능성 제한
2. **시제 분류 및 감성 분석 정확도**
- 완벽한 시제 분류와 감성 분석 모델이 존재하지 않는 문제
3. **시계열 특성 미반영**
- XGBoost는 시계열 데이터의 순차적 의존성을 학습하지 못함.

**향후 연구 방향**
  - **섹터 확장**: 바이오(FDA 승인 뉴스), 자동차(신모델 출시) 등 미래 지향적 정보가 중요한 산업군으로 확대
  - **실전 백테스팅**: 거래비용을 반영한 Sharpe ratio, MDD 산출
  - **딥러닝 적용**: LSTM 기반 시계열 모델 활용하여 시계열 데이터 특성 반영

---

## 7. 프로젝트 구조
```bash
├── data/                            # Datasets
├── src/
│   ├── eda.py                       # 탐색적 데이터 분석
│   ├── feature_engineering.py       # 피처 생성 및 라벨링
│   └── modeling_evaluation.py       # 모델 학습 및 성과 분석
├── results/                         # 시각화 결과물
├── requirements.txt                 # Python 패키지 의존성
└── README.md                        # 프로젝트 문서
```

---

## 연락처 (Contact)

- Author: [arkjackson](https://github.com/arkjackson)
- Email: mihy1968@gmail.com

---
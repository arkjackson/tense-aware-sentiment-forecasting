# Tense-Aware Sentiment Forecasting

## 시제 분리 감성 정보를 활용한 주가 예측 성능 비교 연구

---

## 1. 연구 배경 및 동기

전통적인 **감성 분석 기반 주가 예측 모델**은 뉴스 텍스트에서 추출한 감성 점수를 하나의 지표로 **통합**하여 사용한다. 그러나 이러한 접근은 다음과 같은 구조적 한계를 가진다.

### 기존 접근의 한계

* **시간 정보 손실**

  *“작년 실적 부진(-), 올해 대규모 수주 예정(+)”* 과 같이
  서로 다른 시제의 정보가 혼재된 뉴스는 단일 감성 점수로 미묘한 뉘앙스를 반영하기 어렵다.

* **시장 효율성 미반영**

  효율적 시장 가설(EMH)에 따르면 과거 사건과 미래 전망은 시장에 반영되는 속도가 다르지만,
  기존 모델은 이를 구분하지 않는다.

> **연구 질문**
> 👉 *뉴스 감성 정보를 시제별로 분리하면 주가 예측 성능이 향상되는가?*

---

## 2. 방법론

### 2.1 분석 대상 및 기간

* **대상 기업**: 국내 방산 섹터 주요 4개 기업
  *(HD현대중공업, LIG넥스원, 한국항공우주, 한화시스템)*

* **선정 이유**:
  수주 기반 산업 특성상 **미래 지향적 뉴스 정보의 중요도**가 높은 섹터

* **분석 기간**: `2022-01-02 ~ 2024-12-30`

---

### 2.2 Feature Engineering

#### (1) 공통 가격 피처

* `candle_body_length` = (`Close` − `Open`) / `Open`
* `candle_high_low_length` = (`High` − `Low`) / `Close`

---

#### (2) 감성 피처 구성

##### 🔹 Group A: Baseline

* `score_total`: 시제 구분 없이 통합된 전체 감정 지수

---

##### 🔹 Group B: Experimental (시제 분리)

**① 과거·현재 시제 감성 점수 (`score_past`)**

$$
\frac{(\sum Past_{pos} + \sum Present_{pos}) - (\sum Past_{neg} + \sum Present_{neg})}
{(\sum Past_{pos} + \sum Present_{pos}) + (\sum Past_{neg} + \sum Present_{neg})}
$$

**② 미래 시제 감성 점수 (`score_future`)**

$$
\frac{\sum Future_{pos} - \sum Future_{neg}}
{\sum Future_{pos} + \sum Future_{neg}}
$$

---

### 2.3 수익률 기반 라벨링

* **Label = 1**: 1일 후 수익률 ≥ **+2%**
* **Label = 0**: 1일 후 수익률 < **+2%**

**임계값 설정 근거**

* 증권거래세 및 수수료를 고려할 때 **실질 수익이 가능한 구간**

---

### 2.4 하이퍼파라미터 최적화

* **기법**: Optuna
* **전략**: 피처 그룹별 **독립적 탐색**
* **대상 파라미터**:

  * `n_estimators`
  * `max_depth`
  * `learning_rate`
  * `min_child_weight`

* **목적 함수**: **MCC (Matthews Correlation Coefficient)** 최대화

---

### 2.5 평가 지표

* **Precision**

  상승으로 예측한 사례 중 실제 상승 비율 → *False Positive 억제 능력*

* **F1-Score**

  전반적인 분류 성능 평가

* **MCC**

  예측값과 실제값의 **상관 구조**를 반영하는 균형 지표

---

## 3. 데이터 신뢰성 및 통계적 검증

* **독립성 검정**

  `score_past`–`score_future` 상관계수 평균 **0.6398** → 다중공선성 문제 없음

* **분포 안정화**

  * Jarque–Bera Test
  * Yeo–Johnson 변환
  * Winsorizing 적용

* **정상성 검증**

  ADF(Augmented Dickey-Fuller) Test로 모든 피처 정상성 확인

---

## 4. 주요 결과 요약

| Metric    | Group A | Group B   | Δ      |
| --------- | ------- | --------- | ------ |
| Precision | 0.241   | **0.280** | +0.039 |
| F1-Score  | 0.271   | **0.294** | +0.023 |
| MCC       | -0.012  | **0.053** | +0.065 |

---

### 핵심 인사이트

1. **미래 감성의 우위성**

   `score_future`가 모든 종목에서 `score_past`보다 높은 기여도

2. **리스크 관리 측면 개선**

   Precision 상승 → 잘못된 매수 신호 감소

3. **MCC의 유의미한 개선**

   예측–실제 간 상관 구조를 더 정교하게 포착

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

## 5. 재현성

* Python 3.10+
* `requirements.txt` 기반 환경 구성
* Random Seed 고정

---

## 6. 한계 및 향후 연구

### 한계

* 방산 섹터 단일 분석 → 일반화 한계
* XGBoost의 시계열 의존성 미반영

### 향후 연구

* 섹터 확장 (바이오, 자동차 등)
* 거래비용 반영 백테스팅
* LSTM 등 시계열 딥러닝 모델 적용

---

## 7. 프로젝트 구조

```bash
├── data/
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   └── modeling_evaluation.py
├── results/
├── requirements.txt
└── README.md
```

---

## Contact

* **Author**: arkjackson
* **Email**: [mihy1968@gmail.com](mailto:mihy1968@gmail.com)

---

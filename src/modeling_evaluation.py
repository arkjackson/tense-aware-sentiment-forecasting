from sklearn.metrics import precision_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
import os

from feature_engineering import triple_barrier_labeling_volatility, set_features, preprocess_features
from utils import setup_korean_font, load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        # 하이퍼파라미터 범위 설정
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 15),

        # 고정 파라미터
        'objective': 'binary:logistic',
        'n_estimators': 2000,
        'random_state': 42,
        'n_jobs': 1,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
    }

    # 모델 생성 및 학습
    model = XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 검증 (MCC 기준)
    preds = model.predict(X_val)
    score = matthews_corrcoef(y_val, preds)

    return score

def compare_feature_set_performance(train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame, ticker_name:str, predict_duration:int):
    print(f"==================== Experiment: Optuna Optimization ({ticker_name}) ====================")

    # 피처 그룹 정의
    common_features = ['Candle_body_length', 'High_low_length']
    features_A = common_features + ['Score_total']
    features_B = common_features + ['Score_future', 'Score_past']
    target_col = 'Target'

    groups = {'Group A (Baseline)': features_A, 'Group B (Experiment)': features_B}

    final_results = []

    # Optimization Loop (그룹별 최적화 수행)
    for group_name, feats in groups.items():
        print(f"\n Optimizing Hyperparameters for [{group_name}]...")

        # 데이터 셋
        X_train = train_df[feats]
        y_train = train_df[target_col]
        X_val = val_df[feats]
        y_val = val_df[target_col]
        X_test = test_df[feats]
        y_test = test_df[target_col]

        # Optuna Study 생성 (MCC 최대화 목적)
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)
        print(f" ✅ Best MCC (Val): {study.best_value:.4f}")
        print(f" ✅ Best Params: {study.best_params}")

        # 최적 파라미터로 최종 모델 재학습
        best_params = study.best_params
        best_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric='logloss')
        best_model.fit(X_train, y_train)

        # 테스트 데이터 기준 성능 지표
        test_preds = best_model.predict(X_test)
        final_results.append({
            'Group': group_name,
            'Best LR': best_params['learning_rate'],
            'Precision': precision_score(y_test, test_preds, zero_division=0),
            'F1-Score': f1_score(y_test, test_preds, zero_division=0),
            'MCC': matthews_corrcoef(y_test, test_preds),
        })

    res_df = pd.DataFrame(final_results)

    # 시각화를 위한 Melt
    metrics = ['Precision', 'F1-Score', 'MCC']
    res_melted = res_df.melt(
        id_vars=['Group'],
        value_vars=metrics,
        var_name='Metric',
        value_name='Score'
    )

    plt.figure(figsize=(10, 6))

    # 막대 그래프
    ax = sns.barplot(
        data=res_melted,
        x="Metric", y="Score", hue="Group",
        palette={'Group A (Baseline)': '#95a5a6', 'Group B (Experiment)': '#e74c3c'},
        edgecolor='black', linewidth=1
    )

    # 그래프 스타일링
    plt.title(f'[Final Result] Baseline vs Proposed (Optuna Tuned) : {ticker_name}',
              fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0, 1.15)  # 텍스트 공간 확보
    plt.xlabel("Evaluation Metrics (Test Set)", fontsize=12, fontweight='bold')
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.legend(title="Feature Group", fontsize=10, loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 값 표시
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=4, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'../results/result_{ticker_name}_{predict_duration}d.png')
    plt.show()

    # 요약표 출력
    print(f"\n[Final Performance Summary: {ticker_name}]")
    display_cols = ['Group', 'Precision', 'F1-Score', 'MCC']
    print(f"{res_df[display_cols].round(4).to_string(index=False)}\n")

    return res_df

if __name__ == "__main__":
    setup_korean_font()

    ticker_name_list = ["HD현대중공업", "LIG넥스원", "한국항공우주", "한화시스템"]
    data_folder_path = '../data'

    for ticker_name in ticker_name_list:
        data_path = os.path.join(data_folder_path, ticker_name + ".parquet")
        df = load_data(data_path)

        # 예측기간값 설정: 5, 10, 20일
        predict_duration = 5

        df = triple_barrier_labeling_volatility(df, 'Close', predict_duration, predict_duration, 1, 1)

        df = set_features(df) # 피처 세팅

        # train/val/test: 70%/15%/15%
        train_end = int(len(df) * 0.7)
        val_end = int(len(df) * 0.85)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        preprocessed_train_df, preprocessed_val_df, preprocessed_test_df = preprocess_features(train_df, val_df, test_df) # 피처 전처리

        compare_feature_set_performance(preprocessed_train_df, preprocessed_val_df, preprocessed_test_df, ticker_name, predict_duration) # 모델 최적화 및 성능 비교
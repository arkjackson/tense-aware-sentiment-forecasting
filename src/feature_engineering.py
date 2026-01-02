from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np

def triple_barrier_labeling_volatility(
    df: pd.DataFrame,
    close_col: str = 'Close',
    window: int = 10,
    volatility_window: int = 20,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    drop_last: bool = True
) -> pd.DataFrame:

    prices = df[close_col]
    returns = prices.pct_change()

    # rolling volatility
    volatility = returns.rolling(volatility_window).std()

    n = len(df)
    labels = np.full(n, np.nan)

    prices_np = prices.values
    vol_np = volatility.values

    for t in range(n):
        if t + 1 >= n:
            break

        if np.isnan(vol_np[t]):
            continue  # 변동성 추정 불가 구간 skip

        end = min(t + window, n - 1)
        p0 = prices_np[t]

        upper_barrier = pt_mult * vol_np[t]
        lower_barrier = -sl_mult * vol_np[t]

        for j in range(t + 1, end + 1):
            ret = (prices_np[j] - p0) / p0

            if ret >= upper_barrier:
                labels[t] = 1   # profit-taking
                break
            elif ret <= lower_barrier:
                labels[t] = 0  # stop-loss
                break

        if np.isnan(labels[t]):
            labels[t] = 0  # time barrier hit

    df_out = df.copy()
    df_out['Target'] = labels

    if drop_last:
        df_out = df_out.iloc[:-window]

    return df_out

def set_features(df):
    df['Candle_body_length'] = (df['Close'] - df['Open']) / df['Open']
    df['High_low_length'] = ((df['High'] - df['Low']) / df['Close'])

    epsilon = 1e-5

    # Feature 1. Score_past: (과거 + 현재) 통합
    past_present_pos = df['past_pos_sen_cnt'] + df['present_pos_sen_cnt']
    past_present_neg = df['past_neg_sen_cnt'] + df['present_neg_sen_cnt']
    df['Score_past'] = (past_present_pos - past_present_neg) / (past_present_pos + past_present_neg + epsilon)

    # (2) Score_future: 미래
    future_pos = df['future_pos_sen_cnt']
    future_neg = df['future_neg_sen_cnt']
    df['Score_future'] = (future_pos - future_neg) / (future_pos + future_neg + epsilon)

    # (3) Score_total: 전체 시제 감정 분석
    df['Score_total'] = (past_present_pos + future_pos - past_present_neg - future_neg) / (past_present_pos + future_pos + past_present_neg + future_neg + epsilon)

    return df


def clean_dataset(df):
    """불필요한 피처 제거 및 타겟 결측치 행 제거"""
    df = df.copy()

    # 제거할 피처 리스트
    cols_to_drop = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'past_pos_sen_cnt', 'present_pos_sen_cnt',
        'past_neg_sen_cnt', 'present_neg_sen_cnt',
        'future_pos_sen_cnt', 'future_neg_sen_cnt',
        'pos_sen_ratio', 'neg_sen_ratio', 'company_name'
    ]

    # 1. 존재하는 컬럼만 선택적으로 제거 (에러 방지)
    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols, inplace=True)

    # 2. 타겟 레이블 결측치 제거
    df.dropna(subset=['Target'], inplace=True)

    return df

def preprocess_features(train_df:pd.DataFrame, val_df:pd.DataFrame, test_df:pd.DataFrame):
    """피처 전처리 부분입니다."""
    # Yeo-Johnson 변환
    pt = PowerTransformer(method='yeo-johnson')
    features_to_transform = ['Score_future', 'High_low_length']
    train_df.loc[:, features_to_transform] = pt.fit_transform(train_df[features_to_transform])
    val_df.loc[:, features_to_transform] = pt.transform(val_df[features_to_transform])
    test_df.loc[:, features_to_transform] = pt.transform(test_df[features_to_transform])

    # Winsorizing
    winsorize_features = ['Candle_body_length']
    for feature in winsorize_features:
        # Train 데이터의 1%, 99% 지점의 '실제 값'을 계산
        lower_bound = train_df[feature].quantile(0.01)
        upper_bound = train_df[feature].quantile(0.99)

        # Train
        train_df.loc[:, feature] = train_df[feature].clip(lower_bound, upper_bound)

        # Validation / Test
        val_df.loc[:, feature] = val_df[feature].clip(lower_bound, upper_bound)
        test_df.loc[:, feature] = test_df[feature].clip(lower_bound, upper_bound)

    train_df = clean_dataset(train_df)
    val_df = clean_dataset(val_df)
    test_df = clean_dataset(test_df)

    return train_df, val_df, test_df
from sklearn.preprocessing import PowerTransformer
import pandas as pd

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

def preprocess_features(train_df:pd.DataFrame, another_df:pd.DataFrame):
    """피처 전처리 부분입니다."""
    # Yeo-Johnson 변환
    pt = PowerTransformer(method='yeo-johnson')
    yeo_johnson_features = ['Score_future', 'High_low_length']
    winsorize_features = ['Candle_body_length']
    for col in train_df.columns:
        if col in yeo_johnson_features:
            train_df.loc[:, col] = pt.fit_transform(train_df[[col]]).flatten()
            another_df.loc[:, col] = pt.transform(another_df[[col]]).flatten()
        elif col in winsorize_features:
            # Winsorizing
            # Train 데이터의 1%, 99% 지점의 '실제 값'을 계산
            lower_bound = train_df[col].quantile(0.01)
            upper_bound = train_df[col].quantile(0.99)
            # Train
            train_df.loc[:, col] = train_df[col].clip(lower_bound, upper_bound)
            # Validation / Test
            another_df.loc[:, col] = another_df[col].clip(lower_bound, upper_bound)

    return train_df, another_df

def create_return_target(df:pd.DataFrame, threshold=0.01):
    # 내일의 수익률 계산 (오늘 종가 대비 내일 종가 상승률)
    df['Next_Return'] = df['Close'].pct_change().shift(-1)

    # 라벨링 (임계값 이상이면 1, 아니면 0)
    df['Target'] = (df['Next_Return'] >= threshold).astype(int)

    return df
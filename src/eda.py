from scipy.stats import skew, pearsonr, jarque_bera
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import platform
import os

from feature_engineering import triple_barrier_labeling_volatility, set_features

def setup_korean_font():
    system_name = platform.system()

    if system_name == 'Darwin':  # Mac
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system_name == 'Windows':  # Windows
        plt.rcParams['font.family'] = 'Malgun Gothic'

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    print(f"시스템 환경: {system_name}, 한글 폰트 설정 완료")

def load_data(data_path):
    try:
        df = pd.read_parquet(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'] >= '2022-01-02'].sort_values('Date').set_index('Date')
        return df
    except FileNotFoundError:
        print(f"Error: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {data_path}")
    except pd.errors.EmptyDataError:
        print("Error: 파일이 비어 있습니다.")
    except Exception as e:
        # 기타 발생할 수 있는 모든 에러(파일 손상, 메모리 부족 등) 처리
        print(f"Error: 데이터를 로드하는 중 알 수 없는 오류가 발생했습니다.\n상세 내용: {e}")
    return None

# EDA 1: 시제별 zero ratio 확인
def check_zero_ratio_by_tense(df:pd.DataFrame, ticker_name:str):
    print(f"--- Analyzing {ticker_name} ---")

    zero_ratio_by_tesnse = []

    past_present_zeros = (df['past_pos_sen_cnt'] + df['past_neg_sen_cnt'] + df['present_pos_sen_cnt'] + df['present_neg_sen_cnt'] == 0).mean() * 100
    future_zeros = (df['future_pos_sen_cnt'] + df['future_neg_sen_cnt'] == 0).mean() * 100

    zero_ratio_by_tesnse.append({'Ticker': ticker_name, 'Tense': 'Past + Present', 'Zero_Ratio': past_present_zeros})
    zero_ratio_by_tesnse.append({'Ticker': ticker_name, 'Tense': 'Future', 'Zero_Ratio': future_zeros})

    df_zero_ratio = pd.DataFrame(zero_ratio_by_tesnse)
    print(df_zero_ratio)

    # 막대 그래프 그리기
    ax = sns.barplot(
        data=df_zero_ratio,
        x='Ticker',
        y='Zero_Ratio',
        hue='Tense',
        palette={'Past + Present': '#95a5a6', 'Future': '#2ecc71'}  # 회색(과거), 노랑(현재), 초록(미래)
    )

    # 값 표시 (바 위에 숫자 적기)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

    plt.title('종목별/시제별 문장이 없는 날의 비율', fontsize=15)
    plt.ylabel('Zero Ratio (%)', fontsize=12)
    plt.xlabel('종목명', fontsize=12)
    plt.legend(title='시제 (Tense)', loc='upper right')
    plt.ylim(0, 100)  # Y축 0~100% 고정
    plt.tight_layout()

    plt.show()

# EDA 2: 시제별 데이터 비율
def check_data_ratio_by_tense(df:pd.DataFrame, ticker_name:str):
    # 시제별 분포 가 한쪽으로 치우쳐져 있지 않은가?
    # past + present, future 각각의 비율을 시각화하여 확인.
    print(f"--- Analyzing {ticker_name} ---")

    past_cnt = df['past_pos_sen_cnt'].sum() + df['past_neg_sen_cnt'].sum()
    present_cnt = df['present_pos_sen_cnt'].sum() + df['present_neg_sen_cnt'].sum()
    future_cnt = df['future_pos_sen_cnt'].sum() + df['future_neg_sen_cnt'].sum()

    total_cnt = past_cnt + present_cnt + future_cnt
    tense_data = []

    tense_data.append({
        'Ticker': ticker_name,
        'Past + Present': ((past_cnt  + present_cnt) / total_cnt) * 100,
        'Future': (future_cnt / total_cnt) * 100
    })

    df_ratio = pd.DataFrame(tense_data)
    print(df_ratio)

    df_melted = df_ratio.melt(id_vars='Ticker', var_name='Tense', value_name='Ratio')

    plt.figure(figsize=(12, 6))

    # 색상 지정: 과거(회색), 현재(주황), 미래(초록-강조)
    custom_palette = {'Past + Present': '#95a5a6', 'Future': '#2ecc71'}

    # 막대 그래프 그리기
    ax = sns.barplot(
        data=df_melted,
        x='Ticker',
        y='Ratio',
        hue='Tense',
        palette=custom_palette
    )

    # 그래프 꾸미기
    plt.title(f'시제별 분포 비율 비교', fontsize=15)
    plt.ylabel('비율 (%)', fontsize=12)
    plt.xlabel('종목명', fontsize=12)
    plt.legend(title='시제 (Tense)')
    plt.ylim(0, df_melted['Ratio'].max() + 10)  # Y축 여유 공간 확보

    # 막대 위에 수치 표시
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()

# EDA 3: 두 feature 분포 비교
def compare_distributions(df:pd.DataFrame, ticker_name:str, col1:str, col2:str):
    # Score_past
    # var: 0.15 ~ 0.18
    # Score_future
    # var: 0.13 ~ 0.21

    print(f"--- Analyzing {ticker_name} ---")

    # Statistics Calculation (통계량 계산)
    stats_col1 = {
        'Mean': df[col1].mean(),
        'Var': df[col1].var(),
        'Skew': skew(df[col1])
    }

    stats_col2 = {
        'Mean': df[col2].mean(),
        'Var': df[col2].var(),
        'Skew': skew(df[col2])
    }

    # Visualization (KDE Plot + Stats Box)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()  # 현재 축 가져오기

    # KDE Plot (밀도 그래프)
    # alpha: 투명도, fill: 내부 채우기, warn_singular: 분산 0일 때 경고 끄기
    sns.kdeplot(df[col1], fill=True, color='#3498db', alpha=0.3, label=col1, warn_singular=False)
    sns.kdeplot(df[col2], fill=True, color='#e74c3c', alpha=0.3, label=col2, warn_singular=False)

    plt.title(f'{ticker_name} : {col1} vs {col2} 분포 비교', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Sentiment Score (-1.0 : 부정, +1.0 : 긍정)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-1.1, 1.1)  # X축 범위 고정
    plt.axvline(0, color='black', linestyle=':', linewidth=1)  # 0점 기준선
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper left', fontsize=11)

    # Stats Box 추가 (오른쪽 상단)
    stats_text = (
        f"[Statistics Summary]\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{col1}\n"
        f"  Mean : {stats_col1['Mean']: .4f}\n"
        f"  Var  : {stats_col1['Var']: .4f}\n"
        f"  Skew : {stats_col1['Skew']: .4f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{col2}\n"
        f"  Mean : {stats_col2['Mean']: .4f}\n"
        f"  Var  : {stats_col2['Var']: .4f}\n"
        f"  Skew : {stats_col2['Skew']: .4f}"
    )
    print(stats_text)

    # 텍스트 박스 스타일 지정
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1)

    # 그래프 좌표계(transAxes) 기준 (1.02, 1.0) 위치에 텍스트 배치
    ax.text(1.02, 1.0, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# EDA 4: feature 간의 상관구조 분석
def check_correlation_two_features(df:pd.DataFrame, ticker_name:str, col1:str, col2:str):
    # 상관계수 0.45 ~ 0.76 -> 어느정도 경향성이 같기도 하지만, 다른 정보도 상당 부분 존재함.
    # --- Analyzing HD현대중공업 ---
    # Correlation Coefficient (r): 0.7689
    # P-value: 4.5495e-96
    # --- Analyzing LIG넥스원 ---
    # Correlation Coefficient (r): 0.4597
    # P-value: 8.7817e-27
    # --- Analyzing 한국항공우주 ---
    # Correlation Coefficient (r): 0.6255
    # P-value: 3.9395e-54
    # --- Analyzing 한화시스템 ---
    # Correlation Coefficient (r): 0.6468
    # P-value: 6.2419e-59

    print(f"--- Analyzing {ticker_name} ---")

    # Correlation Calculation (상관계수 계산)
    corr_val, p_val = pearsonr(df[col1], df[col2])

    print(f"Correlation Coefficient (r): {corr_val:.4f}")
    print(f"P-value: {p_val:.4e}")

    # Visualization (Regplot)
    plt.figure(figsize=(8, 8))  # 정사각형 비율 추천

    # regplot: 산점도 + 회귀선
    ax = sns.regplot(
        data=df,
        x=f'{col1}',
        y=f'{col2}',
        color='#5D3F6A',  # 점 색상 (보라색 계열)
        scatter_kws={'alpha': 0.3, 's': 30},  # 점 투명도(0.3) 및 크기(30)
        line_kws={'color': '#e74c3c', 'linewidth': 2}  # 회귀선 (빨간색)
    )

    # 4. Styling & Text
    plt.title(f'Correlation Analysis ({ticker_name})', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(f'{col1}', fontsize=12)
    plt.ylabel(f'{col2}', fontsize=12)

    # 축 범위 고정 (-1.1 ~ 1.1)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    # 기준선 (0점)
    plt.axhline(0, color='black', linestyle=':', linewidth=1)
    plt.axvline(0, color='black', linestyle=':', linewidth=1)
    plt.grid(True, alpha=0.3, linestyle='--')

    # 상관계수 해석 텍스트 박스
    if abs(corr_val) < 0.4:
        interpretation = "Independent (Good)"
    elif abs(corr_val) < 0.7:
        interpretation = "Moderate Corr"
    else:
        interpretation = "High Corr (Redundant)"

    stats_text = (
        f"Pearson r : {corr_val:.3f}\n"
        f"P-value   : {p_val:.1e}\n"
        f"Result    : {interpretation}"
    )

    # 텍스트 박스 추가 (좌측 상단)
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.show()

# EDA 5: 피처별 분포 분석 (JB Test)
def check_normality_jb(df:pd.DataFrame, ticker_name:str):
    """피처별 왜도, 첨도를 분석"""
    # --- Analyzing Normality (Q-Q Plot) for HD현대중공업 ---
    # Candle_body_length: skewness: 0.316 , Excess Kurtosis: 0.912 -> 약한 fat tail
    # High_low_length: skewness: 1.298 -> highly skewed(pos), Excess Kurtosis: 2.391 -> 뚜렷한 fat tail
    # Score_past: skewness: -0.294, Excess Kurtosis: -0.524
    # Score_future: skewness: -0.487, Excess Kurtosis: -0.241
    # Score_total: skewness: -0.363, Excess Kurtosis: -0.389
    # --- Analyzing Normality (Q-Q Plot) for LIG넥스원 ---
    # Candle_body_length: skewness: -0.265, Excess Kurtosis: 1.27 -> 뚜렷한 fat tail
    # High_low_length: skewness: 1.54 -> highly skewed(pos), Excess Kurtosis: 3.208 -> highly fat tail
    # Score_past: skewness: -0.763, Excess Kurtosis: 0.078
    # Score_future: skewness: -1.295 -> highly skewed(neg), Excess Kurtosis: 1.397 -> 뚜렷한 fat tail
    # Score_total: skewness: -0.855, Excess Kurtosis: 0.094
    # --- Analyzing Normality (Q-Q Plot) for 한국항공우주 ---
    # Candle_body_length: skewness: -0.415, Excess Kurtosis: 2.404 -> 뚜렷한 fat tail
    # High_low_length: skewness: 1.770 -> highly skewed(pos), Excess Kurtosis: 5.075 -> highly fat tail
    # Score_past: skewness: -0.761, Excess Kurtosis: 0.359
    # Score_future: skewness: -1.318 -> highly skewed(neg), Excess Kurtosis: 1.478 -> 뚜렷한 fat tail
    # Score_total: skewness: -0.878, Excess Kurtosis: 0.571
    # --- Analyzing Normality (Q-Q Plot) for 한화시스템 ---
    # Candle_body_length: skewness: 0.341, Excess Kurtosis: 3.656 -> highly fat tail
    # High_low_length: skewness: 2.010 -> highly skewed(pos), Excess Kurtosis: 6.177 -> highly fat tail
    # Score_past: skewness: -0.715, Excess Kurtosis: 0.281
    # Score_future: skewness: -1.122 -> highly skewed(neg), Excess Kurtosis: 1.331 -> 뚜렷한 fat tail
    # Score_total: skewness: -0.823, Excess Kurtosis: 0.575

    # 'High_low_length': 공통적으로 highly skewed(pos) & fat tail -> Yeo-Johnson 변환 -> 왜도/첨도 완화 성공
    # 'Score_future': 공통적으로 highly skewed(neg) & fat tail -> Yeo-Johnson 변환 -> 왜도/첨도 완화 성공
    # 'Candle_body_length': 공통적으로 fat tail -> winsorizing 적용 -> 첨도 완화 성공

    print(f"--- Analyzing Normality (Q-Q Plot) for {ticker_name} ---")

    for column in df.columns:
        if column not in ['Candle_body_length', 'High_low_length', 'Score_past', 'Score_future', 'Score_total']:
            continue
        # NaN 제거 (Q-Q Plot: 결측치 있으면 에러 발생)
        df_temp = df.dropna(subset=[column])

        if len(df_temp[column]) < 2:
            print("데이터 부족으로 시각화 불가")
            return

        # 정규성 검정: Jarque-Bera Test
        # statistic: JB 통계량 (클수록 정규분포와 다름)
        # p_value: 유의확률 (0.05 미만이면 '정규분포 아님'으로 결론)
        jb_stat, jb_p_value = jarque_bera(df_temp[column])

        # P-value 해석: 정규성 검증
        if jb_p_value < 0.05:
            normality = "Normality rejected (JB test)"
            color_res = 'red'
        else:
            normality = "Normality accepted (JB test)"
            color_res = 'blue'

        # Q-Q Plot Visualization
        fig, ax = plt.subplots(figsize=(8, 7))

        # scipy.stats.probplot을 이용해 Q-Q plot 데이터 생성 및 시각화
        # dist='norm': 정규분포와 비교
        (osm, osr), (slope, intercept, r) = stats.probplot(df_temp[column], dist="norm", plot=ax)

        # 스타일링
        ax.get_lines()[0].set_marker('o')  # 데이터 점
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[0].set_color('#2980b9')
        ax.get_lines()[0].set_alpha(0.4)

        ax.get_lines()[1].set_color('#c0392b')  # 정규분포 기준선 (빨강)
        ax.get_lines()[1].set_linewidth(2)
        ax.get_lines()[1].set_linestyle('--')

        # 타이틀 및 라벨
        ax.set_title(f'Normality Check : {column} ({ticker_name})', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
        ax.set_ylabel(f'{column} Quantiles', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Result Annotation
        skewness = stats.skew(df_temp[column])
        kurtosis = stats.kurtosis(df_temp[column])  # Excess Kurtosis

        stats_text = (
            f"Skewness : {skewness:.3f}\n"
            f"Kurtosis : {kurtosis:.3f}\n"
            f"--------------------\n"
            f"JB Stat  : {jb_stat:.1f}\n"
            f"P-value  : {jb_p_value:.1e}\n"
            f"Result   : {normality}"
        )
        print(f"컬럼: {column}\n{stats_text}")

        # 텍스트 박스 추가
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color_res)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, fontfamily='monospace')

        plt.tight_layout()
        plt.show()

# EDA 6: 피처별 정상성 체크 (ADF Test)
def check_stationarity_adf(df:pd.DataFrame, ticker_name:str):
    # 비정상성 피처 리스트
    # 1. High_low_length (HD현대중공업) -> 차분 적용
    print(f"--- Running ADF Test for {ticker_name} ---")
    for column in df.columns:
        if column not in ['Candle_body_length', 'High_low_length', 'Score_past', 'Score_future', 'Score_total']:
            continue

        df = df.dropna(subset=[column])
        # ADF 검증 최소 데이터 개수 확인
        if len(df[column]) < 10:
            print("데이터 부족으로 ADF 테스트 불가")
            return

        # ADF Test Execution
        result = adfuller(df[column], autolag='AIC') # autolag='AIC': 검정에 필요한 시차(Lag)를 자동으로 결정

        adf_stat = result[0]
        p_value = result[1]

        # 판정 (p-value 0.05 기준)
        if p_value < 0.05:
            verdict = "Stationary"
            color_res = 'blue'
        else:
            verdict = "Non-Stationary"
            color_res = 'red'

        # Visualization
        plt.figure(figsize=(10, 5))

        # 시계열 그래프
        plt.plot(df[column], color='#34495e', linewidth=1, label=column)
        plt.axhline(0, color='orange', linestyle='--', linewidth=1.5)  # 0 기준선

        # 타이틀
        plt.title(f'ADF Test Result : {column} ({ticker_name})', fontsize=15, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel(f'{column}')
        plt.grid(True, alpha=0.3)

        # 4. Result Text Box
        stats_text = (
            f"ADF Statistic : {adf_stat:.4f}\n"
            f"P-value       : {p_value:.4e}\n"
            f"Used Lags     : {result[2]}\n"
            f"-----------------------\n"
            f"Result: {verdict}"
        )
        print(f"{column}\n{stats_text}")

        # 텍스트 박스 스타일 (결과에 따라 테두리 색상 변경)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color_res, linewidth=2)

        plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=props, fontfamily='monospace')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    setup_korean_font()

    ticker_name_list = ["HD현대중공업", "LIG넥스원", "한국항공우주", "한화시스템"]
    data_folder_path = '../data'

    for ticker_name in ticker_name_list:
        data_path = os.path.join(data_folder_path, ticker_name + ".parquet")
        df = load_data(data_path)
        if df is None:
            continue

        # 단기(5일) 예측
        # df = triple_barrier_labeling_volatility(df, 'Close', 5, 20, 1, 1)

        # 장기(20일) 예측
        df = triple_barrier_labeling_volatility(df, 'Close', 20, 20, 1, 1)

        df = set_features(df)

        # train / val / test: 70% / 15% / 15%
        train_end = int(len(df) * 0.7)
        train_df = df.iloc[:train_end]

        # ===== EDA 수행 =====
        # 주석 풀어서 확인하기

        # EDA 1: 시제별 zero ratio 확인
        # check_zero_ratio_by_tense(train_df, ticker_name)

        # EDA 2: 시제별 분포 확인
        # check_data_ratio_by_tense(train_df, ticker_name)

        # EDA 3: Score_past - Score_future 피처 분포 비교
        # compare_distributions(train_df, ticker_name, 'Score_past', 'Score_future')

        # EDA 4: Score_past - Score_future 피처 상관구조 분석
        # check_correlation_two_features(train_df, ticker_name, 'Score_past', 'Score_future')

        # EDA 5-1: 피처별 분포 분석 (JB Test)
        # check_normality_jb(train_df, ticker_name)

        # EDA 5-2: 왜도, 첨도 완화 작업 후 분포 재확인
        # Yeo-Johnson 변환
        # pt = PowerTransformer(method='yeo-johnson')
        # features_to_transform = ['Score_future', 'High_low_length']
        # train_df.loc[:, features_to_transform] = pt.fit_transform(train_df[features_to_transform])
        # Winsorizing
        # winsorize_features = ['Candle_body_length']
        # for feature in winsorize_features:
        #     train_df.loc[:, feature] = winsorize(train_df[feature], limits=[0.01, 0.01])  # 임계값 1% ~ 99%
        # check_normality_jb(train_df, ticker_name)

        # EDA 6: 피처별 정상성 체크 (ADF Test)
        check_stationarity_adf(train_df, ticker_name)
import matplotlib.pyplot as plt
import pandas as pd
import platform

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
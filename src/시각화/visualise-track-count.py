import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

#LLM을 활용한 MBTI별 플레이리스트 트랙 개수 분포 시각화 및 이상치 강조

# 1. 현재 디렉터리에서 모든 MBTI CSV 파일 목록을 찾습니다.
# '*_df.csv' 패턴을 사용하여 ENFJ_df.csv, INFP_df.csv 등의 파일을 모두 찾습니다.
all_csv_files = glob.glob('*_df.csv')
csv_files = [f for f in all_csv_files if 'combined_mbti_df.csv' not in f.lower()]

# 파일들을 임시로 담을 리스트 초기화
df_list = []

print(f"발견된 MBTI 데이터 파일 수: {len(csv_files)}개")

# 2. 각 파일을 읽어와서 MBTI 유형 정보를 추가하고 리스트에 저장합니다.
for file_name in csv_files:
    # 파일 이름에서 MBTI 유형을 추출합니다 (예: ENFJ_df.csv -> ENFJ).
    mbti_type = file_name.replace('_df.csv', '').upper()
    
    try:
        # 파일 로드
        df_temp = pd.read_csv(file_name)
        
        # MBTI 유형 컬럼을 추가합니다.
        df_temp['mbti'] = mbti_type
        df_list.append(df_temp)
        
    except Exception as e:
        print(f"오류 발생: {file_name} 파일을 로드하지 못했습니다. - {e}")
        continue

# 3. 리스트의 모든 데이터프레임을 하나로 합쳐서 df_combined를 완성합니다.
if df_list:
    df_combined = pd.concat(df_list, ignore_index=True)
    print("-----------------------------------------------------")
    print(f"SUCCESS: 전체 데이터셋 로드 및 결합 완료. 총 행(플레이리스트) 개수: {len(df_combined)}개")
    print(f"결합된 데이터셋 (df_combined)의 첫 5개 행:")
    print(df_combined.head())
    print("-----------------------------------------------------")
else:
    # 파일이 하나도 발견되지 않았을 경우
    print("ERROR: 로드할 '*_df.csv' 데이터 파일이 현재 디렉터리에 없습니다.")
    
# 이제 df_combined를 사용하여 이전에 작성했던 이상치 분석 코드를 실행할 수 있습니다.
# df_combined 데이터프레임이 이미 로드되고 'track_count', 'mbti' 컬럼이 있다고 가정합니다.

# 1. 이상치 기준 값 계산 (IQR 방식)
Q1 = df_combined['track_count'].quantile(0.25) # 25th percentile
Q3 = df_combined['track_count'].quantile(0.75) # 75th percentile
IQR = Q3 - Q1

# 통계적 이상치 기준 (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
lower_bound_IQR = Q1 - 1.5 * IQR
upper_bound_IQR = Q3 + 1.5 * IQR

print(f"--- IQR 기반 이상치 기준 ---")
print(f"Q1 (25%): {Q1:.2f} 곡")
print(f"Q3 (75%): {Q3:.2f} 곡")
print(f"IQR (Q3-Q1): {IQR:.2f} 곡")
print(f"이상치 하한선: {lower_bound_IQR:.2f} 곡")
print(f"이상치 상한선: {upper_bound_IQR:.2f} 곡")

# 2. 새로운 'is_outlier' 컬럼 생성 및 색상 구분
# 통계적 이상치 기준을 벗어나는 경우를 'Outlier'로 정의합니다.
df_combined['is_outlier'] = 'Normal'
df_combined.loc[
    (df_combined['track_count'] < lower_bound_IQR) | (df_combined['track_count'] > upper_bound_IQR), 
    'is_outlier'
] = 'Outlier'

# 3. 시각화: 스트립 플롯 (Strip Plot)

plt.figure(figsize=(16, 8))

# Stripplot을 사용하여 모든 데이터 포인트를 점으로 표시합니다.
# x축: track_count, y축: mbti (요청하신 가로/세로)
sns.stripplot(
    x='track_count', 
    y='mbti', 
    data=df_combined, 
    hue='is_outlier', 
    palette={'Normal': 'gray', 'Outlier': 'red'}, # 이상치를 빨간색으로 강조
    size=5,
    jitter=True, # 점들이 겹치지 않도록 무작위 분산 추가
    legend=False 
)

# 4. 축과 제목 설정
plt.title('Track Count Distribution and IQR-Based Outliers by MBTI Type', fontsize=18)
plt.xlabel('Track Count per Playlist', fontsize=14)
plt.ylabel('MBTI Type', fontsize=14)

# 5. 이상치 강조를 위한 선 추가
plt.axvline(x=lower_bound_IQR, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.axvline(x=upper_bound_IQR, color='red', linestyle='--', linewidth=1, alpha=0.7)

# 6. 범례 수동 설정 (이상치 강조)
plt.legend(
    handles=[plt.scatter([], [], color='red', label=f'Outlier (<{lower_bound_IQR:.0f} or >{upper_bound_IQR:.0f})')],
    title='Data Point Type',
    loc='lower right'
)

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#LLM을 활용한 MBTI별 플레이리스트 개수 시각화

# 1. 데이터 불러오기
# 파일 이름이 combined_mbti_df.csv 라고 가정합니다.
try:
    df = pd.read_csv('combined_mbti_df.csv')
    print("데이터 로드 성공!")
except FileNotFoundError:
    print("오류: combined_mbti_df.csv 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 파일이 없으면 더 이상 진행하지 않습니다.
    exit()

# 2. MBTI별 플레이리스트 개수 계산 (데이터프레임의 행(row) 수를 카운트)
# 이 데이터셋은 각 행이 '하나의 플레이리스트'의 집계 데이터라고 가정합니다.
# 따라서 MBTI별로 그룹화하고 개수를 세면 됩니다.
mbti_counts = df.groupby('mbti').size().reset_index(name='playlist_count')

# 3. 플레이리스트 개수(playlist_count) 기준으로 내림차순 정렬
mbti_counts = mbti_counts.sort_values(by='playlist_count', ascending=False)

# 4. 통계량 계산
count_min = mbti_counts['playlist_count'].min()
count_max = mbti_counts['playlist_count'].max()
count_mean = mbti_counts['playlist_count'].mean()

# 5. 시각화 준비 (막대 그래프)

# 색상 설정 (파란색, 하늘색 계열)
# 16개의 막대에 서로 다른 색상을 적용하기 위해 Matplotlib의 'Blues' 컬러맵을 사용합니다.
n_bars = len(mbti_counts)
colors = plt.cm.Blues(np.linspace(0.4, 1, n_bars))

# 그래프 크기 설정
plt.figure(figsize=(15, 7))

# 막대 그래프 그리기
bars = plt.bar(mbti_counts['mbti'], mbti_counts['playlist_count'], color=colors)

# 제목 및 축 라벨 설정
plt.title('Number of Playlists by MBTI Type', fontsize=16)
plt.xlabel('MBTI type', fontsize=12)
plt.ylabel('Number of Playlists', fontsize=12)

# 가로축(MBTI) 라벨을 보기 좋게 회전
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7) # 수평 그리드 라인 추가

# 각 막대 위에 숫자(개수) 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.tight_layout() # 그래프 요소가 잘리지 않도록 조정
plt.show()

# 6. 계산된 통계량 출력
print("\n--- 분석 결과 요약 ---")
print(f"총 MBTI 유형 개수: {n_bars}개")
print(f"최저 플레이리스트 수: {count_min:.0f}개")
print(f"최고 플레이리스트 수: {count_max:.0f}개")
print(f"플레이리스트 수 평균: {count_mean:.2f}개")
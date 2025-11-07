# mbti가 제목에 포함되지 않은 플레이리스트 제외하기

import pandas as pd
import os

# 필터링할 MBTI
target_mbti = "ISTP"

# 기존 파일 경로
mbti_folder = "./data/proceed"
input_file = os.path.join(mbti_folder, f"{target_mbti}_playlist_tracklists.csv")

# CSV 로드
df = pd.read_csv(input_file)

print(f"원래 playlist 개수: {len(df)}")

# MBTI 이름이 playlist_name에 포함된 것만 남기기 (대소문자 무시)
filtered_df = df[df["playlist_name"].str.contains(target_mbti, case=False, na=False)]

print(f"필터링 후 playlist 개수: {len(filtered_df)}")

# 결과 저장
output_file = os.path.join(mbti_folder, f"{target_mbti}_playlist_tracklists_filtered.csv")
filtered_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 필터링된 결과 저장 완료 → {output_file}")
import os
import pandas as pd
import matplotlib.pyplot as plt

proceed_folder = "./data/proceed/filtering" # barchart 보고 싶은 폴더 넣기

mbti_counts = {}

for filename in os.listdir(proceed_folder):
    if filename.endswith(".csv"):
        # MBTI = 파일명 앞 4글자
        mbti = filename[:4].upper()

        df = pd.read_csv(os.path.join(proceed_folder, filename))
        mbti_counts[mbti] = len(df)

# MBTI 알파벳 순 정렬
mbti_list = sorted(mbti_counts.keys())
counts = [mbti_counts[m] for m in mbti_list]

plt.figure(figsize=(10, 5))
bars = plt.bar(mbti_list, counts)

# ✅ 막대 위에 숫자 표시
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(height),
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

plt.title("Playlists per MBTI", fontsize=16)
plt.xlabel("MBTI", fontsize=14)
plt.ylabel("Playlist Count", fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
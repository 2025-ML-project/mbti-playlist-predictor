import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# normalized playlist 폴더
folder = "./data/proceed/filtering"

all_dfs = []

def parse_ids(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x

# 모든 MBTI CSV 읽기
for file in os.listdir(folder):
    if file.endswith(".csv"):
        mbti = file.split("_")[0]
        df = pd.read_csv(os.path.join(folder, file))

        df["track_ids"] = df["track_ids"].apply(parse_ids)
        df["track_count"] = df["track_ids"].apply(len)
        df["mbti"] = mbti

        all_dfs.append(df)

# 하나로 합치기
data = pd.concat(all_dfs, ignore_index=True)

print("총 playlist 수:", len(data))

# ✅ 전체 playlist 곡 수 분포 통계
print("\n✅ 곡 수 통계")
print(data["track_count"].describe())

# ✅ 전체 boxplot
plt.figure(figsize=(10,4))
sns.boxplot(data["track_count"])
plt.title("Playlist Track Count Distribution (All MBTI)")
plt.grid(True)
plt.show()

# ✅ MBTI별 boxplot
plt.figure(figsize=(14,6))
sns.boxplot(x="mbti", y="track_count", data=data)
plt.title("Track Count Distribution by MBTI")
plt.grid(True)
plt.show()

# ✅ histogram
plt.figure(figsize=(10,4))
plt.hist(data["track_count"], bins=40, edgecolor='black')
plt.title("Playlist Track Count Histogram")
plt.xlabel("Track Count")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
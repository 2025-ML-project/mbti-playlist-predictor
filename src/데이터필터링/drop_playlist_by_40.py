import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

input_folder = "./data/proceed/filtering"

def parse_ids(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x

all_dfs = []

# ✅ Read all MBTI CSV files & compute track_count
for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue
    
    mbti = file.split("_")[0]
    df = pd.read_csv(os.path.join(input_folder, file))

    df["track_ids"] = df["track_ids"].apply(parse_ids)
    df["track_count"] = df["track_ids"].apply(len)
    df["mbti"] = mbti

    all_dfs.append(df)

# ✅ Combine into one DataFrame
data = pd.concat(all_dfs, ignore_index=True)

print("✅ Total playlists (before drop):", len(data))
print(data["track_count"].describe())

# ✅ Drop playlists with <40 tracks
data_40 = data[data["track_count"] >= 40]

print("\n✅ Total playlists (after drop, >=40):", len(data_40))
print(data_40["track_count"].describe())

# ✅ Playlist count by MBTI
before = data.groupby("mbti").size()
after = data_40.groupby("mbti").size()

print("\n✅ Playlist count by MBTI (before drop)")
print(before)

print("\n✅ Playlist count by MBTI (after drop, >=40)")
print(after)

# ✅ Visualization

## TRACK COUNT DISTRIBUTION (Before vs After)
plt.figure(figsize=(14,4))

plt.subplot(1,2,1)
sns.boxplot(data["track_count"])
plt.title("Track Count Distribution (Before Drop)")
plt.grid(True)

plt.subplot(1,2,2)
sns.boxplot(data_40["track_count"])
plt.title("Track Count Distribution (After Drop ≥ 40)")
plt.grid(True)

plt.show()

## HISTOGRAM (After Drop)
plt.figure(figsize=(12,4))
plt.hist(data_40["track_count"], bins=40, edgecolor='black')
plt.title("Histogram of Track Count (After ≥40 Drop)")
plt.xlabel("Number of Tracks")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

## ✅ MBTI Playlist Counts (Before Drop) — ADD LABELS
plt.figure(figsize=(12,4))
ax = sns.barplot(x=before.index, y=before.values)
plt.title("Number of Playlists per MBTI (Before Drop)")
plt.grid(True)

for i, v in enumerate(before.values):
    ax.text(i, v + 1, str(v), ha='center', va='bottom')

plt.show()

## ✅ MBTI Playlist Counts (After Drop) — ADD LABELS
plt.figure(figsize=(12,4))
ax = sns.barplot(x=after.index, y=after.values)
plt.title("Number of Playlists per MBTI (After Drop ≥ 40)")
plt.grid(True)

for i, v in enumerate(after.values):
    ax.text(i, v + 1, str(v), ha='center', va='bottom')

plt.show()
import os
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
input_folder = "./data/proceed/filtering"             # 원본 CSV들
output_folder = "./data/proceed/filtering_ge40"       # 드롭 후 저장 위치
os.makedirs(output_folder, exist_ok=True)

def parse_ids(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return []
    return x

# -------- 1) 파일별로 ≥40 필터 적용 + 저장 --------
saved_files = []
for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue

    src = os.path.join(input_folder, file)
    df = pd.read_csv(src)

    # track_count 계산
    df["track_ids"] = df["track_ids"].apply(parse_ids)
    df["track_count"] = df["track_ids"].apply(len)

    # ≥40 필터
    df_ge40 = df[df["track_count"] >= 40].copy()

    # 저장 파일명: 원래 이름 + _ge40.csv
    base, ext = os.path.splitext(file)
    dst = os.path.join(output_folder, f"{base}_ge40{ext}")
    df_ge40.to_csv(dst, index=False, encoding="utf-8-sig")
    saved_files.append(dst)

    print(f"Saved: {dst}  (kept {len(df_ge40)}/{len(df)})")

# -------- 2) (옵션) 저장된 결과를 합쳐서 분포/카운트 확인 --------
# 파일명에서 MBTI를 앞 4글자로 추출하는 기존 관례 유지
all_dfs = []
for path in saved_files:
    file = os.path.basename(path)
    mbti = file[:4]  # 파일명 앞 4글자 (예: ENFP_)
    df = pd.read_csv(path)

    # 시각화를 위한 보조 컬럼
    df["track_ids"] = df["track_ids"].apply(parse_ids)
    df["track_count"] = df["track_ids"].apply(len)
    df["mbti"] = mbti

    all_dfs.append(df)

if all_dfs:
    data = pd.concat(all_dfs, ignore_index=True)

    print("\n✅ Total playlists (after saving, ≥40 only):", len(data))
    print(data["track_count"].describe())

    # MBTI별 개수
    counts = data.groupby("mbti").size().sort_index()
    print("\n✅ Number of Playlists per MBTI (After Drop ≥ 40)")
    print(counts)

    # --- Visualization ---
    # Boxplot
    plt.figure(figsize=(10,4))
    sns.boxplot(data["track_count"])
    plt.title("Track Count Distribution (Saved ≥ 40 Only)")
    plt.grid(True)
    plt.show()

    # Histogram
    plt.figure(figsize=(10,4))
    plt.hist(data["track_count"], bins=40, edgecolor='black')
    plt.title("Histogram of Track Count (Saved ≥ 40 Only)")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Barplot with value labels
    plt.figure(figsize=(12,4))
    ax = sns.barplot(x=counts.index, y=counts.values)
    plt.title("Number of Playlists per MBTI (Saved ≥ 40 Only)")
    plt.ylabel("Count")
    plt.grid(True, axis="y")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha='center', va='bottom')
    plt.show()
else:
    print("\n(No files saved; nothing to visualize.)")
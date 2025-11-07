import os
import ast
import math
import json
import random
import pandas as pd
from collections import Counter, defaultdict

# ===경로설정===
INPUT_FOLDER  = "./data/proceed/filtering_ge40"
OUTPUT_FOLDER = "./data/proceed/fixed_len_100_mbtiidf"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FIX_LEN = 100
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PAD_TOKEN = "PAD" # 후에 zero-vector로 매핑해야 한다.

# track_ids 파싱
def parse_ids(x):
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, str):
                return [v]
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return x if isinstance(x, list) else []

# 1) MBTI별 bag 만들기 (문서=MBTI, 토큰=track_id)
mbti_bags = defaultdict(list) # mbti별로 track_id가 모두 들어있는 bag 생성
playlist_rows = []

for fname in os.listdir(INPUT_FOLDER):
    if not fname.endswith(".csv"):
        continue
    mbti = fname[:4].upper()
    path = os.path.join(INPUT_FOLDER, fname)
    df = pd.read_csv(path)
    if "track_ids" not in df.columns:
        print(f"[WARN] {fname}: 'track_ids' missing, skip")
        continue

    for _, row in df.iterrows():
        tids = [t for t in parse_ids(row.get("track_ids", [])) if t and isinstance(t, str)]
        if not tids:
            continue
        mbti_bags[mbti].extend(tids) # mbti bag에 track 추가 (중복 포함 -> tf 계산해야되서)
        playlist_rows.append((
            mbti,
            row.get("playlist_id", ""),
            row.get("playlist_name", ""),
            tids
        ))

all_mbtis = sorted(mbti_bags.keys())
print(f"MBTI docs found: {all_mbtis}") # 16개의 bag이 나왔다.


# 2) DF/IDF 계산 (문서=16개 MBTI bag)
#    DF(track) = 등장한 MBTI 문서 수
#    IDF = log((D+1)/(DF+1)) + 1,  D=문서 수(최대 16)

D = len(all_mbtis) # 문서수 = 16개
track_in_mbti = {mb: set(bag) for mb, bag in mbti_bags.items()}  # mbti 문서 내 고유 트랙 집합(df용)
df_counts = Counter() # 트랙별 df집계
for mb, trackset in track_in_mbti.items():
    for t in trackset:
        df_counts[t] += 1 # 해당 문서(mbti)에 존재하는 모든 트랙에 대해 df값 올려줌

idf = defaultdict(float)
for t, dfc in df_counts.items():
    # idf : 문서가 적을수록(=여러 mbti에 잘 안나올 수록) 값이 커짐 -> mbti 고유성 반영
    idf[t] = math.log((D + 1) / (dfc + 1)) + 1.0

# 3) MBTI별 TF 계산 및 점수표 생성
#    TF(track, MBTI) = count_in_mbti_bag / total_tracks_in_mbti_bag (정규화 TF)
#    score_mbti[track] = TF * IDF
#    → 이 점수는 "그 MBTI에서의 고유성"을 반영 (모든 플레이리스트에 동일하게 적용)
score_mbti = {}  # mbti -> dict(track -> score)
for mb, bag in mbti_bags.items():
    tf_counts = Counter(bag)
    total = sum(tf_counts.values()) if tf_counts else 1
    s = {}
    for t, c in tf_counts.items():
        tf = c / total
        s[t] = tf * idf[t] # tfidf 점수
    score_mbti[mb] = s

# 4) 각 플레이리스트를 100곡으로 고정
#    - len <= 100: PAD 추가
#    - len > 100 : (그 MBTI의 score_mbti) 낮은 곡부터 제거
#    동점은 랜덤 셔플 후 정렬로 편향 최소화
RESULTS = defaultdict(list)  # mbti -> rows

for (mbti, playlist_id, playlist_name, tids) in playlist_rows:
    tids = [t for t in tids if t and isinstance(t, str)]
    if len(tids) == 0:
        continue

    if len(tids) <= FIX_LEN:
        # 길이가 모자르면(100개가 안되면) PAD_TOKEN으로 뒤를 채워서 고정 길이를 만든다.
        padded = tids + [PAD_TOKEN] * (FIX_LEN - len(tids))
        RESULTS[mbti].append({
            "mbti": mbti,
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "orig_len": len(tids),
            "removed": 0,
            "padded": FIX_LEN - len(tids),
            "fixed_track_ids": json.dumps(padded, ensure_ascii=False)
        })
        continue

    # 길이가 100을 초과하는 경우 -> tf-idf 낮은 곡부터 제거(대중적인 곡을 지우는 의미)
    sdict = score_mbti.get(mbti, {})
    scored = []
    for t in tids:
        # bag에 아예 없을 수는 드묾. 그래도 안전하게 기본값 아주 작게.
        score = sdict.get(t, 0.0)
        scored.append((t, score))

    # 동점 편향 방지: 먼저 셔플, 그 다음 점수 내림차순 정렬
    random.shuffle(scored)
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = [t for t, _ in scored[:FIX_LEN]]
    removed_cnt = max(0, len(tids) - FIX_LEN)

    RESULTS[mbti].append({
        "mbti": mbti,
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "orig_len": len(tids),
        "removed": removed_cnt,
        "padded": 0,
        "fixed_track_ids": json.dumps(kept, ensure_ascii=False)
    })


# 5) MBTI별 CSV 저장
for mbti, rows in RESULTS.items():
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_FOLDER, f"{mbti}_fixed100.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {mbti}: {len(out_df)} playlists → {out_path}")

print("\n✅ Done. Fixed-length=100 using MBTI-doc TF-IDF (trim+pad).")
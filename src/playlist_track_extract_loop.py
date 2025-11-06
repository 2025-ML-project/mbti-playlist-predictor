import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import random,time


# Spotify API 인증
load_dotenv()
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
#     client_id=CLIENT_ID,
#     client_secret=CLIENT_SECRET,
#     requests_timeout=30
# ))


sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    ),
    requests_timeout=30,
    retries=10,           # 자동 재시도 횟수
    status_retries=5,     # HTTP 오류 시 재시도
    backoff_factor=2      # 재시도 대기시간 2배씩 증가 (1s → 2s → 4s → …)
)

# 실험할 MBTI 지정
target_mbti = "INTP"

# MBTI별 CSV 파일 폴더
mbti_folder = "./data/raw/MBTI_playlist"
file_path = os.path.join(mbti_folder, f"{target_mbti}_df.csv")

print(f"▶ {target_mbti} 실험 시작 ({file_path})")

df = pd.read_csv(file_path)
if "playlist_id" not in df.columns:
    print(f"{file_path}에 playlist_id 컬럼이 없습니다. 중단합니다.")
    exit()

playlist_rows = []

for _, row in df.iterrows():
    playlist_id = row["playlist_id"]

    try:
        playlist = sp.playlist(playlist_id)
        playlist_name = playlist["name"]
        results = sp.playlist_tracks(playlist_id)

        tracks = results["items"]
        while results["next"]:
            results = sp.next(results)
            tracks.extend(results["items"])

        track_ids = []
        for item in tracks:
            track = item["track"]
            if track and track["id"]:
                track_ids.append(track["id"])

        playlist_rows.append({
            "mbti": target_mbti,
            "playlist_id": playlist_id,
            "playlist_name": playlist_name,
            "track_ids": track_ids
        })

        print(f"  ✅ {playlist_name}: {len(track_ids)}곡 수집 완료")
        #time.sleep(0.3) # 원래 코드
        #time.sleep(random.uniform(1.0, 2.5))
        # ✅ Spotify API에 너무 자주 요청하지 않도록 랜덤하게 대기
        sleep_time = random.uniform(1.0, 2.5)
        print(f"  ⏳ Waiting {sleep_time:.2f} seconds before next playlist...")
        time.sleep(sleep_time)

    except Exception as e:
        print(f"  ⚠️ {playlist_id} 실패: {e}")
        continue

# DataFrame 생성 및 저장
out_df = pd.DataFrame(playlist_rows)
save_name = f"{target_mbti}_playlist_tracklists.csv"
out_path = os.path.join(mbti_folder, save_name)
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"💾 {target_mbti} 결과 저장 완료 ({len(out_df)}개 playlist) → {save_name}")
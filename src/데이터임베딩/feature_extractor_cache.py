import os
import pandas as pd
import time
import requests
from dotenv import load_dotenv
import ast


BASE_DIR = "/Users/gim-yesong/TMD/mbti-playlist-predictor"
ENV_PATH = os.path.join(BASE_DIR, ".env", "rapidapi.env")
load_dotenv(dotenv_path=ENV_PATH)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

# print("🔑 RAPIDAPI_KEY:", bool(RAPIDAPI_KEY))
# print("🌐 RAPIDAPI_HOST:", RAPIDAPI_HOST)

MBTI = "ESFP" # 실행할 MBTI 지정
INPUT_FILE = f"{BASE_DIR}/data/fixed_len_60_mbtiidf/{MBTI}_fixed60.csv"
OUTPUT_FILE = f"{BASE_DIR}/data/features_by_mbti_60/{MBTI}_features.csv"


# 캐싱 로드
TRACK_CACHE = {} 
API_CALL_COUNT = 0 
API_FEATURE_COLS = ["tempo", "energy", "danceability", "happiness", "acousticness", 
                    "instrumentalness", "liveness", "speechiness", "loudness", 
                    "mode", "key", "camelot"]
column_order = ["mbti", "playlist_id", "track_id"] + API_FEATURE_COLS + ["error"]

def load_all_mbti_cache():
    
    global TRACK_CACHE
    
    CACHE_DIR = f"{BASE_DIR}/data/features_by_mbti_60"
    if not os.path.exists(CACHE_DIR):
        print(f" 캐시 디렉토리 ({CACHE_DIR})가 존재하지 않습니다.")
        return

    csv_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_features.csv')]
    
    if not csv_files:
        print(" 로드할 MBTI 캐시 파일이 없습니다.")
        return

    total_track_count = 0
    
    for filename in csv_files:
        filepath = os.path.join(CACHE_DIR, filename)
        try:
            cache_df = pd.read_csv(filepath)
            for _, row in cache_df.iterrows():
                if row.get('track_id') == 'PAD' or row.get('error', False) == True:
                    continue
                
                track_id = row['track_id']
                if track_id not in TRACK_CACHE: 
                    features = row[API_FEATURE_COLS].to_dict()
                    features = {k: (v if pd.notna(v) else 0) for k, v in features.items()}
                    
                    TRACK_CACHE[track_id] = features
                    total_track_count += 1
        except Exception as e:
            print(f"캐시 파일 로드 중 오류 발생 ({filename}): {e}")

    print(f"캐시 로드 완료: {total_track_count}개의 트랙 정보가 모든 MBTI 파일에서 캐시되었습니다.")

load_all_mbti_cache()

def get_track_features(track_id):
    global API_CALL_COUNT

    if track_id == "PAD":
        return {
            "tempo": 0, "energy": 0, "danceability": 0, "happiness": 0,
            "acousticness": 0, "instrumentalness": 0, "liveness": 0,
            "speechiness": 0, "loudness": 0, "mode": "none", "key": "none", "camelot": "none"
        }

    if track_id in TRACK_CACHE:
        return TRACK_CACHE[track_id]
        
    url = f"https://track-analysis.p.rapidapi.com/pktx/spotify/{track_id}"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            excluded = {"id", "name", "album"}
            features = {k: v for k, v in data.items() if k not in excluded}
            
            TRACK_CACHE[track_id] = features 
            API_CALL_COUNT += 1
            return features
        else:
            print(f"호출 실패 ({response.status_code}) : {track_id}")
            time.sleep(0.3) 
            return None
    except Exception as e:
        print(f"예외 발생: {e}")
        time.sleep(3) 
        return None


df = pd.read_csv(INPUT_FILE)

df["fixed_track_ids"] = df["fixed_track_ids"].apply(lambda x: ast.literal_eval(x))
print(f" {MBTI} 데이터 로드 완료: {df.shape}")

# 이전 호출 이어서 실행할 경우 인덱스 조절 
START_INDEX = 68 

try:
    for playlist_idx, row in df.iloc[START_INDEX:].iterrows():
            
        playlist_id = row["playlist_id"]
        track_list = row["fixed_track_ids"]
        results = []
        
        current_progress = playlist_idx + 1 
        total_rows = len(df)
        print(f"\n [MBTI: {MBTI}] 플레이리스트 {current_progress}/{total_rows} (ID: {playlist_id})")

        for track_id in track_list:
            
            is_api_called = track_id not in TRACK_CACHE and track_id != "PAD"
            
            feature_data = get_track_features(track_id)

            if feature_data is None:
                feature_data = {"error": True}
            else:
                feature_data["error"] = False 

            feature_data["mbti"] = MBTI
            feature_data["playlist_id"] = playlist_id
            feature_data["track_id"] = track_id

            results.append(feature_data)
            
            '''# 5. 캐시 미스(API 호출)가 발생했을 때만 지연 시간을 적용
            if is_api_called:
                time.sleep(0.15)''' 

        features_df = pd.DataFrame(results)
        
        for col in column_order:
            if col not in features_df.columns:
                features_df[col] = None
        
        features_df = features_df[column_order]

        if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
            features_df.to_csv(OUTPUT_FILE, mode="a", index=False, header=False)
        else:
            features_df.to_csv(OUTPUT_FILE, index=False) 

        print(f" 중간 저장 완료 ({current_progress}/{total_rows} 플레이리스트, 총 API 호출: {API_CALL_COUNT})")

except KeyboardInterrupt:
    print("\n 실행 중단 감지! 지금까지의 결과는 안전하게 저장되었습니다.")


print(f"\n 전체 완료 → {OUTPUT_FILE}")
print(f"총 API 호출 횟수: {API_CALL_COUNT}")
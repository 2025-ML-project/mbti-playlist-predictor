import os
import pandas as pd
import time
import requests
import numpy as np
from dotenv import load_dotenv
import sys

# 설정 및 경로(각자 변경)
BASE_DIR = "/Users/gim-yesong/TMD/mbti-playlist-predictor"
ENV_PATH = os.path.join(BASE_DIR, ".env", "rapidapi.env")
load_dotenv(dotenv_path=ENV_PATH)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

# MBTI 지정
MBTI = "ISFJ" 
OUTPUT_FILE = f"{BASE_DIR}/data/features_by_mbti_60/{MBTI}_features.csv"

# print("RAPIDAPI_KEY:", bool(RAPIDAPI_KEY))
# print("RAPIDAPI_HOST:", RAPIDAPI_HOST)

print(f"재시도 대상 파일: {OUTPUT_FILE}")


def get_track_features(track_id):
    if track_id == "PAD":
        return {
            "tempo": 0, "energy": 0, "danceability": 0, "happiness": 0,
            "acousticness": 0, "instrumentalness": 0, "liveness": 0, 
            "speechiness": 0, "loudness": 0, "mode": "none", 
            "key": "none", "camelot": "none", "error": False
        }

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
            features["error"] = False
            return features
        else:
            print(f"  호출 실패 ({response.status_code}) : {track_id}")
            time.sleep(0.3)
            return None
    except Exception as e:
        print(f"  예외 발생: {e}")
        return None


try:
    df = pd.read_csv(OUTPUT_FILE)
    
    # error == True 재시도
    error_is_true_mask = df["error"] == True
    
    # NaN 재시도
    has_critical_nan_mask = df[CRITICAL_FEATURES].isna().any(axis=1)
    
    retry_mask = error_is_true_mask | has_critical_nan_mask
    
    retry_df = df[retry_mask].copy()
        
    if retry_df.empty:
        print("재시도할 트랙이 없습니다. 모든 데이터가 채워져 있습니다.")
        sys.exit(0)

    print(f"총 {len(df)}개 레코드 중, {len(retry_df)}개의 실패 레코드를 재시도합니다.")
    
    try:
        for index, row in retry_df.iterrows():
            track_id = row["track_id"]
            
            if track_id == "PAD" and row["tempo"] == 0:
                print(f"PAD 트랙 건너뛰기: {track_id}")
                continue

            print(f"\n 인덱스 {index} 트랙 재시도: {track_id}")
            time.sleep(0.2)
            
            feature_data = get_track_features(track_id)

            if feature_data:
                for k, v in feature_data.items():
                    df.loc[index, k] = v
                print(f"  성공: 인덱스 {index} 트랙 데이터 채움.")
            else:
                df.loc[index, "error"] = True
                print(f"  재실패: 인덱스 {index} 트랙 데이터 유지.")

            time.sleep(0.3)
            
        # 최종 저장
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n 모든 재시도 완료. 업데이트된 데이터는 {OUTPUT_FILE}에 저장되었습니다.")

    # 중간 저장을 위한 코드 (KeyboardInterrupt)
    except KeyboardInterrupt:
        df.to_csv(OUTPUT_FILE, index=False)
        print("\n\n 실행 중단, 지금까지 성공한 데이터는 파일에 저장되었습니다.")
        sys.exit(0) 


except FileNotFoundError:
    print(f" 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {OUTPUT_FILE}")
except Exception as e:
    print(f"\n 최종 예외 발생: {e}")
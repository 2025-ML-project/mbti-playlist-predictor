# tensor_generator.py

import os
import pandas as pd
import numpy as np
from typing import Tuple

BASE_DIR = "/Users/gim-yesong/TMD/mbti-playlist-predictor" 
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed_features")
TEMP_PROCESSED_DIR = os.path.join(OUTPUT_DIR, "temp_processed_csv")

FIXED_TRACK_LEN = 60

FINAL_FEATURES = [
    'tempo', 'loudness', 'mode', 
    'danceability', 'energy', 'speechiness', 'acousticness', 
    'liveness', 'happiness', 'instrumentalness'
]
NUM_FEATURES = len(FINAL_FEATURES)


def get_zero_pad_row(mbti: str, playlist_id: str) -> pd.DataFrame:
    
    pad_data = {
        'mbti': mbti, 
        'playlist_id': playlist_id, 
        'track_id': 'PAD',
        'tempo': 0.0, 
        'loudness': 0.0, 
        'mode': 0, 
        'danceability': 0.0, 
        'energy': 0.0, 
        'speechiness': 0.0, 
        'acousticness': 0.0, 
        'liveness': 0.0, 
        'happiness': 0.0, 
        'instrumentalness': 0.0
    }
    return pd.DataFrame([pad_data])


def process_playlist_data(playlist_df: pd.DataFrame, mbti: str, playlist_id: str) -> pd.DataFrame:
 
    current_len = len(playlist_df)
    
    if current_len > FIXED_TRACK_LEN:
        print(f" {mbti}-{playlist_id}: {current_len}개 -> 60개로 자르기.")
        return playlist_df.head(FIXED_TRACK_LEN)
        
    elif current_len < FIXED_TRACK_LEN:
        num_padding = FIXED_TRACK_LEN - current_len
        print(f" {mbti}-{playlist_id}: {current_len}개 -> {num_padding}개 Zero Pad 추가.")
        
        pad_rows = pd.concat([get_zero_pad_row(mbti, playlist_id)] * num_padding, ignore_index=True)
        
        return pd.concat([playlist_df, pad_rows], ignore_index=True)
    
    else:
        return playlist_df


def create_rank3_tensor() -> Tuple[np.ndarray, np.ndarray]:
    
    all_playlists_data = []
    all_mbti_labels = []
    
    processed_files = [f for f in os.listdir(TEMP_PROCESSED_DIR) if f.endswith("_processed.csv")]
    mbti_list = [f.split('_processed.csv')[0] for f in processed_files]


    for mbti in mbti_list:
        processed_file = os.path.join(TEMP_PROCESSED_DIR, f"{mbti}_processed.csv")
        if not os.path.exists(processed_file):
            print(f"경고: {mbti} 처리 파일이 없습니다. 건너뜁니다.")
            continue
            
        df = pd.read_csv(processed_file)
        
        grouped = df.groupby('playlist_id')
        
        for playlist_id, playlist_df in grouped:
            
            processed_df = process_playlist_data(playlist_df, mbti, playlist_id)
                
            feature_matrix = processed_df[FINAL_FEATURES].values
            
            all_playlists_data.append(feature_matrix)
            all_mbti_labels.append(mbti)

    X = np.array(all_playlists_data) 
    y = np.array(all_mbti_labels)
    
    return X, y

def main():
    X_tensor, y_labels = create_rank3_tensor()

    print("\n--- 3차원 텐서 및 라벨 생성 완료 ---")
    print(f" 최종 3차원 텐서 (X) 형태: {X_tensor.shape}")
    print(f"   -> (총 플레이리스트 수, 노래 개수={FIXED_TRACK_LEN}, 특성 개수={NUM_FEATURES})")
    print(f" 최종 라벨 배열 (y) 형태: {y_labels.shape}")
    print(f"   -> 포함된 MBTI 라벨 종류: {np.unique(y_labels)}")
    
    np.save(os.path.join(OUTPUT_DIR, 'X_rank3_tensor.npy'), X_tensor)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y_labels)
    print(f"\n데이터가 {OUTPUT_DIR} 에 X_rank3_tensor.npy, y_labels.npy로 저장되었습니다.")

if __name__ == "__main__":
    if not os.path.exists(TEMP_PROCESSED_DIR) or not os.listdir(TEMP_PROCESSED_DIR):
        print(" 오류: 'data_cleaner.py'를 먼저 실행하여 중간 CSV 파일을 생성해야 합니다.")
    else:
        main()
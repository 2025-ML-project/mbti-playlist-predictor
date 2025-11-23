# data_cleaner.py

import os
import pandas as pd
from typing import List
import numpy as np

# 설정 및 경로(각자 변경)

BASE_DIR = "/Users/gim-yesong/TMD/mbti-playlist-predictor" 
INPUT_DIR = os.path.join(BASE_DIR, "data/features_by_mbti_60")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed_features")
TEMP_PROCESSED_DIR = os.path.join(OUTPUT_DIR, "temp_processed_csv")

FINAL_FEATURES = [
    'tempo', 'loudness', 'mode', 
    'danceability', 'energy', 'speechiness', 'acousticness', 
    'liveness', 'happiness', 'instrumentalness'
]
ZERO_PAD_VALUES = {
    'tempo': 0.0, 'loudness': 0.0, 'mode': 0, # mode는 0 또는 1이므로 0으로 설정
    'danceability': 0.0, 'energy': 0.0, 'speechiness': 0.0, 'acousticness': 0.0, 
    'liveness': 0.0, 'happiness': 0.0, 'instrumentalness': 0.0
}

# Loudness 특성 : ' dB'를 제거, 숫자로 변환
def clean_loudness(val):
    if isinstance(val, str) and val.endswith(' dB'):
        try:
            return float(val.replace(' dB', '').strip())
        except ValueError:
            return 0.0
    return float(val) if pd.notna(val) else 0.0

# Mode 특성 : 0(minor) 또는 1(major)로 변환
def clean_mode(val):
    if pd.isna(val):
        return 0
    
    if isinstance(val, str):
        val = val.lower().strip()
        if val in ('major', '1'):
            return 1 
        elif val in ('minor', '0'):
            return 0 
        else:
            return 0
    
    try:
        val = int(val)
        return 1 if val == 1 else 0
    except (ValueError, TypeError):
        return 0 

def process_data(input_file: str, output_file: str, mbti: str) -> bool:

    try:
        df = pd.read_csv(input_file, na_values=['', ' '], keep_default_na=True)

        df['error'] = df['error'].astype(str).str.lower().isin(['true', '1'])
        
        cols_to_keep = ['mbti', 'playlist_id', 'track_id', 'error'] + FINAL_FEATURES
        df = df[df.columns.intersection(cols_to_keep)].copy() 
        
        has_nan_in_features = df[FINAL_FEATURES].isnull().any(axis=1)

        pad_mask = df['error'] | has_nan_in_features
        
        numerical_features = list(ZERO_PAD_VALUES.keys())
        
        for col in numerical_features:
            if col in df.columns:
                df.loc[pad_mask, col] = ZERO_PAD_VALUES.get(col, 0.0)

        df.loc[pad_mask, 'track_id'] = 'PAD'
        
        df = df[['mbti', 'playlist_id', 'track_id'] + FINAL_FEATURES].copy()
        
        features_to_divide = [
            'danceability', 'energy', 'speechiness', 'acousticness', 
            'liveness', 'happiness', 'instrumentalness'
        ]
        df[features_to_divide] = df[features_to_divide] / 100
        
        df['loudness'] = df['loudness'].apply(clean_loudness)

        df['mode'] = df['mode'].apply(clean_mode)
        
        df['mbti'] = mbti 
        
        df.to_csv(output_file, index=False)
        return True
    
    except Exception as e:
        print(f" {mbti} 파일 처리 중 오류 발생: {e}")
        return False

def main():
    os.makedirs(TEMP_PROCESSED_DIR, exist_ok=True)
    print(f" 입력 디렉토리: {INPUT_DIR}")
    print(f" 임시 출력 디렉토리: {TEMP_PROCESSED_DIR}")

    mbti_list = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith("_features.csv"):
            mbti = filename.replace("_features.csv", "")
            mbti_list.append(mbti)
            
            input_file = os.path.join(INPUT_DIR, filename)
            output_file = os.path.join(TEMP_PROCESSED_DIR, f"{mbti}_processed.csv")
            
            process_data(input_file, output_file, mbti)
            print(f"   -> {mbti} 데이터 처리 완료 및 임시 저장.")

    print("\n--- 데이터 클리닝 및 정규화 완료 ---")

if __name__ == "__main__":
    main()
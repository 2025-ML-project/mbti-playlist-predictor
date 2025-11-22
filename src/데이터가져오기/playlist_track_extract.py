import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# .env에서 환경변수 로드
load_dotenv()
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

# Spotipy 인증 객체 생성
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# 플레이리스트 ID (공개 playlist)
playlist_id = "4AE4DBt4YjJJ8v4Hk9myWl"

# 플레이리스트의 트랙 정보 가져오기
results = sp.playlist_tracks(playlist_id)

# track ID 추출 및 출력
track_ids = []
for item in results['items']:
    track = item['track']
    track_ids.append(track['id'])
    print(f"{track['name']} → {track['id']}")

print(f"\n총 {len(track_ids)}개의 트랙을 가져왔습니다.")
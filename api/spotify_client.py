import os
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# ── Robust .env loading ──────────────────────────────────────────────
# __file__ is this script: .../api/spotify_client.py
# .parent is .../api/
# .parent.parent is .../project_root/  (where .env lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

if not _ENV_PATH.is_file():
    raise FileNotFoundError(
        f".env file not found at {_ENV_PATH}\n"
        f"Create it with SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and SPOTIPY_REDIRECT_URI."
    )

load_dotenv(dotenv_path=str(_ENV_PATH))

# ── Read credentials ─────────────────────────────────────────────────
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI")

_missing = [
    name for name, val in [
        ("SPOTIFY_CLIENT_ID", client_id),
        ("SPOTIFY_CLIENT_SECRET", client_secret),
        ("SPOTIPY_REDIRECT_URI", redirect_uri),
    ] if not val
]
if _missing:
    raise EnvironmentError(
        f"Missing environment variables in {_ENV_PATH}: {', '.join(_missing)}"
    )

# ── Spotify client ───────────────────────────────────────────────────
sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="playlist-read-private playlist-read-collaborative"
    )
)


def search_tracks(query, limit=5):
    results = sp.search(q=query, limit=limit)
    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            "track_name": item['name'],
            "artist": item['artists'][0]['name'],
            "track_id": item['id']
        })
    return tracks


def get_playlist_tracks(playlist_id, region, playlist_type):
    results = sp.playlist_tracks(playlist_id)
    tracks_data = []

    while results:
        for item in results['items']:
            track = item.get('track')
            if track and track.get('id'):
                tracks_data.append({
                    "track_id": track['id'],
                    "track_name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "region": region,
                    "playlist_type": playlist_type
                })

        if results['next']:
            results = sp.next(results)
        else:
            results = None

    return tracks_data


def get_audio_features(track_ids):
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        audio_features = sp.audio_features(batch)
        for af in audio_features:
            if af:
                features.append(af)
    return features

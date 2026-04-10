import os
import requests
import base64

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "REDACTED_SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "REDACTED_SPOTIFY_CLIENT_SECRET")

_token_cache = {"token": None, "expires_at": 0}

def _get_token() -> str:
    import time
    if _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]
    creds = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {creds}"},
        data={"grant_type": "client_credentials"},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data["expires_in"] - 60
    return _token_cache["token"]

def _va_to_spotify_params(valence_norm: float, arousal_norm: float) -> dict:
    spotify_valence = round((valence_norm + 1) / 2, 3)
    spotify_energy = round((arousal_norm + 1) / 2, 3)
    tempo_target = int(80 + arousal_norm * 60)
    danceability = round(0.4 + arousal_norm * 0.3, 2)
    return {
        "target_valence": max(0.0, min(1.0, spotify_valence)),
        "target_energy": max(0.0, min(1.0, spotify_energy)),
        "target_tempo": max(50, min(180, tempo_target)),
        "target_danceability": max(0.1, min(0.9, danceability)),
        "min_valence": max(0.0, spotify_valence - 0.2),
        "max_valence": min(1.0, spotify_valence + 0.2),
        "min_energy": max(0.0, spotify_energy - 0.2),
        "max_energy": min(1.0, spotify_energy + 0.2),
    }

def recommend_songs(valence: float, arousal: float, limit: int = 5) -> list:
    v_norm = (valence - 5) / 4
    a_norm = (arousal - 5) / 4
    token = _get_token()

    MOOD_QUERIES = {
        (True, True):   ["happy upbeat pop", "feel good dance", "euphoric electronic"],
        (True, False):  ["calm acoustic", "peaceful ambient", "serene piano"],
        (False, True):  ["angry rock", "intense metal", "aggressive rap"],
        (False, False): ["sad ballad", "melancholic indie", "emotional piano"],
    }
    queries = MOOD_QUERIES[(v_norm >= 0, a_norm >= 0)]

    results = []
    for query in queries:
        if len(results) >= limit:
            break
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 3, "market": "US"},
            timeout=10
        )
        if resp.status_code != 200:
            continue
        tracks = resp.json().get("tracks", {}).get("items", [])
        for t in tracks:
            if len(results) >= limit:
                break
            results.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"] if t["artists"] else "Unknown",
                "album": t["album"]["name"],
                "image": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
                "preview_url": t.get("preview_url"),
                "spotify_url": t["external_urls"].get("spotify"),
                "duration_ms": t["duration_ms"],
            })
    return results[:limit]

import sys
import os
import time
import sqlite3
import pickle
import numpy as np
import librosa
from feature_extraction import extract_features, SR, WINDOW_SEC, DB_FILE

TOP_K   = 5
MIN_SEC = 5

# Trong so 18 features: [pitch, zcr, energy, centroid, bandwidth, mfcc_1..13]
# MFCC chiem 0.60 vi mo ta am sac tot nhat
WEIGHTS = np.array(
    [0.15, 0.05, 0.05, 0.08, 0.07] + [0.60 / 13] * 13,
    dtype=np.float32
)


def load_songs():
    if not os.path.exists(DB_FILE):
        print(f"Chua co {DB_FILE}. Hay chay: python main.py build")
        sys.exit(1)
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT name, file_path, offset, features FROM songs"
    ).fetchall()
    conn.close()
    return [
        {"name": r[0], "path": r[1], "offset": r[2],
         "features": np.array(pickle.loads(r[3]), dtype=np.float32)}
        for r in rows
    ]


def load_query(query_path):
    duration = librosa.get_duration(path=query_path)
    if duration < MIN_SEC:
        print(f"Query qua ngan ({duration:.1f}s). Can it nhat {MIN_SEC}s.")
        sys.exit(1)

    audio, sr = librosa.load(query_path, sr=SR, mono=True)
    target_len = WINDOW_SEC * sr

    if len(audio) < target_len:
        audio = np.tile(audio, int(np.ceil(target_len / len(audio))))

    return audio[:target_len], sr


def weighted_distance(q, d):
    denom = np.maximum(np.abs(q), np.abs(d))
    denom[denom == 0] = 1e-9
    return float(np.sum(np.abs(q - d) / denom * WEIGHTS))


def search(query_path):
    print("=" * 55)
    print(f"FILE TRUY VAN : {os.path.basename(query_path)}")
    print(f"DO DO         : Weighted Normalized")
    print("=" * 55)

    # Buoc 1: trich feature query
    print("\n[1] Trich xuat dac trung...")
    t0 = time.time()
    audio, sr = load_query(query_path)
    q_vec = np.array(extract_features(audio, sr), dtype=np.float32)
    print(f"    Thoi gian: {time.time()-t0:.2f}s")

    # Buoc 2: tinh distance voi tat ca windows trong DB
    print("\n[2] Tim kiem...")
    songs = load_songs()
    t0    = time.time()

    all_results = []
    for song in songs:
        dist = weighted_distance(q_vec, song["features"])
        all_results.append((dist, song))

    all_results.sort(key=lambda x: x[0])
    t_ms = (time.time() - t0) * 1000
    print(f"    {len(songs)} windows | Thoi gian: {t_ms:.2f}ms")

    # Buoc 3: deduplicate — giu moi bai 1 lan (distance nho nhat)
    seen = set()
    top5 = []
    for dist, song in all_results:
        if song["name"] not in seen:
            seen.add(song["name"])
            top5.append({**song, "distance": dist})
        if len(top5) == TOP_K:
            break

    # Buoc 4: hien thi ket qua
    print(f"\n[3] KET QUA TOP {TOP_K}:")
    print(f"    {'Hang':<6} {'Tuong dong':>12}  Ten bai")
    print(f"    {'-'*50}")

    max_dist = top5[-1]["distance"] if top5[-1]["distance"] > 0 else 1.0
    for rank, r in enumerate(top5, 1):
        sim = max(0.0, 1.0 - r["distance"] / max_dist) * 100
        r["similarity"] = sim
        print(f"    {rank:<6} {sim:>11.2f}%  {r['name']}")

    print("=" * 55)
    return top5


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cach dung: python search.py <file_audio.wav>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1]):
        print(f"Khong tim thay file: {sys.argv[1]}")
        sys.exit(1)
    search(sys.argv[1])

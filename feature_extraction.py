import os
import time
import numpy as np
import librosa
import sqlite3
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

DATASET_DIR = "dataset_beat"
DB_FILE     = "music_database.db"
SR          = 44100
WINDOW_SEC  = 10
STEP_SEC    = 5
N_WORKERS   = max(1, multiprocessing.cpu_count() - 1)


def extract_features(audio, sr):
    """Trích xuất vector 18 chiều từ audio array."""
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )
    pitch     = float(np.nanmean(f0[voiced_flag])) if voiced_flag.any() else 0.0
    zcr       = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    energy    = float(np.mean(audio ** 2))
    centroid  = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    mfcc_mean = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1).tolist()
    return [pitch, zcr, energy, centroid, bandwidth] + mfcc_mean


def extract_features_from_file(file_path):
    audio, sr = librosa.load(file_path, sr=SR, mono=True)
    return extract_features(audio, sr)


def process_file(file_path):
    """Xử lý toàn bộ một file WAV, trả về list (name, file_path, offset, features)."""
    window_len = WINDOW_SEC * SR
    step_len   = STEP_SEC * SR
    name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")
    audio, sr = librosa.load(file_path, sr=SR, mono=True)
    results = []
    for start in range(0, len(audio) - window_len + 1, step_len):
        window = audio[start : start + window_len]
        results.append((name, file_path, start / SR, extract_features(window, sr)))
    return results


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            file_path TEXT NOT NULL,
            offset    REAL NOT NULL,
            features  BLOB NOT NULL
        )
    """)
    conn.commit()


def insert_window(conn, name, file_path, offset, features):
    blob = pickle.dumps(np.array(features, dtype=np.float32))
    conn.execute(
        "INSERT INTO songs (name, file_path, offset, features) VALUES (?,?,?,?)",
        (name, file_path, offset, blob)
    )
    conn.commit()


def insert_windows_batch(conn, rows):
    """Chèn tất cả windows của 1 file cùng lúc — nhanh hơn commit từng row."""
    data = [
        (name, fp, offset, pickle.dumps(np.array(feats, dtype=np.float32)))
        for name, fp, offset, feats in rows
    ]
    conn.executemany(
        "INSERT INTO songs (name, file_path, offset, features) VALUES (?,?,?,?)",
        data
    )
    conn.commit()


def load_all_songs(conn):
    rows = conn.execute(
        "SELECT id, name, file_path, offset, features FROM songs"
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "path": r[2], "offset": r[3],
         "features": pickle.loads(r[4])}
        for r in rows
    ]


def build_database():
    # Xóa DB cũ nếu schema cũ (chưa có cột offset)
    if os.path.exists(DB_FILE):
        _conn = sqlite3.connect(DB_FILE)
        cols = [r[1] for r in _conn.execute("PRAGMA table_info(songs)").fetchall()]
        _conn.close()
        if "offset" not in cols:
            print("Phát hiện DB cũ → xóa và build lại.\n")
            os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    existing_paths = set(
        r[0] for r in conn.execute("SELECT DISTINCT file_path FROM songs").fetchall()
    )

    wav_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".wav") and not f.startswith("_temp")
    ])

    todo = [
        os.path.join(DATASET_DIR, f)
        for f in wav_files
        if os.path.join(DATASET_DIR, f) not in existing_paths
    ]

    total     = len(wav_files)
    skipped   = total - len(todo)
    new_count = 0
    t_start   = time.time()

    print(f"Tìm thấy {total} file WAV trong '{DATASET_DIR}'")
    print(f"Bỏ qua (đã có): {skipped} | Cần xử lý: {len(todo)}")
    print(f"Window: {WINDOW_SEC}s | Step: {STEP_SEC}s | Workers: {N_WORKERS}\n")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_file, fp): fp for fp in todo}

        done = 0
        for future in as_completed(futures):
            fp   = futures[future]
            done += 1
            fname = os.path.basename(fp)
            try:
                t0   = time.time()
                rows = future.result()
                insert_windows_batch(conn, rows)
                new_count += 1
                print(f"  [{done:3}/{len(todo)}] {fname}: {len(rows)} windows | {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  [{done:3}/{len(todo)}] {fname}: Lỗi - {e}")

    songs      = load_all_songs(conn)
    conn.close()
    total_time = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"Tong bai trong DB  : {len(set(s['name'] for s in songs))}")
    print(f"Tong windows       : {len(songs)}")
    print(f"Them moi lan nay   : {new_count}")
    print(f"Thoi gian xu ly    : {total_time:.1f}s")
    print(f"Database           : {DB_FILE}")


if __name__ == "__main__":
    build_database()

import os
import time
import numpy as np
import librosa
import sqlite3
import pickle
from scipy.spatial import KDTree

DATASET_DIR = "dataset_beat"
DB_FILE     = "music_database.db"
KDTREE_FILE = "kdtree.pkl"
SR          = 44100


def extract_features(file_path):
    """
    Trích xuất vector 18 chiều từ file WAV:
      [pitch, zcr, energy, centroid, bandwidth, mfcc_1..mfcc_13]
    """
    audio, sr = librosa.load(file_path, sr=SR, mono=True)

    # 1. Pitch — tần số cơ bản (Hz), nhận diện giai điệu
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    pitch = float(np.nanmean(f0[voiced_flag])) if voiced_flag.any() else 0.0

    # 2. Zero-Crossing Rate — tốc độ đổi dấu tín hiệu
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    # 3. Average Energy — năng lượng trung bình
    energy = float(np.mean(audio ** 2))

    # 4. Spectral Centroid — trọng tâm phổ (Hz)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

    # 5. Spectral Bandwidth — băng thông phổ
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))

    # 6. MFCC — 13 hệ số đặc trưng âm sắc
    mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).tolist()

    return [pitch, zcr, energy, centroid, bandwidth] + mfcc_mean


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            file_path TEXT NOT NULL,
            pitch     REAL,
            zcr       REAL,
            energy    REAL,
            centroid  REAL,
            bandwidth REAL,
            features  BLOB NOT NULL
        )
    """)
    conn.commit()


def insert_song(conn, name, file_path, features):
    blob = pickle.dumps(np.array(features, dtype=np.float32))
    conn.execute(
        """INSERT INTO songs
           (name, file_path, pitch, zcr, energy, centroid, bandwidth, features)
           VALUES (?,?,?,?,?,?,?,?)""",
        (name, file_path,
         features[0], features[1], features[2], features[3], features[4],
         blob)
    )
    conn.commit()


def load_all_songs(conn):
    rows = conn.execute(
        "SELECT id, name, file_path, features FROM songs"
    ).fetchall()
    return [
        {"id": r[0], "name": r[1], "path": r[2],
         "features": pickle.loads(r[3])}
        for r in rows
    ]


def build_kdtree(songs):
    vectors = np.array([s["features"] for s in songs], dtype=np.float32)
    return KDTree(vectors), vectors


def build_database():
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    existing = set(
        r[0] for r in conn.execute("SELECT file_path FROM songs").fetchall()
    )

    wav_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".wav") and not f.startswith("_temp")
    ])

    total     = len(wav_files)
    new_count = 0
    t_start   = time.time()

    print(f"Tìm thấy {total} file WAV trong '{DATASET_DIR}'\n")

    for idx, filename in enumerate(wav_files, 1):
        file_path = os.path.join(DATASET_DIR, filename)

        if file_path in existing:
            print(f"  [{idx:3}/{total}] Bỏ qua (đã có): {filename}")
            continue

        name = os.path.splitext(filename)[0].replace("_", " ")
        print(f"  [{idx:3}/{total}] Trích xuất: {filename}")

        try:
            t0       = time.time()
            features = extract_features(file_path)
            elapsed  = time.time() - t0
            insert_song(conn, name, file_path, features)
            new_count += 1
            print(f"          pitch={features[0]:7.1f}Hz | zcr={features[1]:.4f} | "
                  f"energy={features[2]:.6f} | ({elapsed:.1f}s)")
        except Exception as e:
            print(f"          ✗ Lỗi: {e}")

    songs = load_all_songs(conn)
    conn.close()

    if not songs:
        print("DB trống.")
        return

    tree, _ = build_kdtree(songs)
    with open(KDTREE_FILE, "wb") as f:
        pickle.dump({"tree": tree, "songs": songs}, f)

    total_time = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"Tổng bài trong DB  : {len(songs)}")
    print(f"Thêm mới lần này   : {new_count}")
    print(f"Số chiều vector    : {len(songs[0]['features'])}")
    print(f"Thời gian xử lý    : {total_time:.1f}s")
    print(f"KD-Tree            : {KDTREE_FILE}")
    print(f"Database           : {DB_FILE}")


if __name__ == "__main__":
    build_database()

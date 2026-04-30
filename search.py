import sys
import os
import time
import pickle
import numpy as np
from scipy.spatial import KDTree
from feature_extraction import extract_features

DB_FILE     = "music_database.db"
KDTREE_FILE = "kdtree.pkl"
TOP_K       = 5


def load_kdtree():
    if not os.path.exists(KDTREE_FILE):
        print(f"Chưa có {KDTREE_FILE}. Hãy chạy feature_extraction.py trước.")
        sys.exit(1)
    with open(KDTREE_FILE, "rb") as f:
        data = pickle.load(f)
    return data["tree"], data["songs"]


def search(query_path, metric="l2"):
    print("=" * 55)
    print(f"FILE TRUY VẤN : {os.path.basename(query_path)}")
    print(f"ĐỘ ĐO         : {'L2 (Euclidean)' if metric == 'l2' else 'L1 (Manhattan)'}")
    print("=" * 55)

    # Bước 1: Trích xuất vector của file query
    print("\n[1] Trích xuất đặc trưng file truy vấn...")
    t0           = time.time()
    query_vector = np.array(extract_features(query_path), dtype=np.float32)
    t_extract    = time.time() - t0

    labels = ["Pitch(Hz)", "ZCR", "Energy", "Centroid", "Bandwidth"] + \
             [f"MFCC_{i}" for i in range(1, 14)]
    print(f"    Thời gian trích xuất : {t_extract:.2f}s")
    print(f"    Vector {len(query_vector)} chiều:")
    for label, val in zip(labels[:5], query_vector[:5]):
        print(f"      {label:12s} = {val:.6f}")
    print(f"      MFCC_1..13   = [{', '.join(f'{v:.2f}' for v in query_vector[5:])}]")

    # Bước 2: Load KD-Tree và tìm k-NN
    print(f"\n[2] Tìm kiếm {TOP_K} láng giềng gần nhất (KD-Tree)...")
    tree, songs = load_kdtree()

    t0 = time.time()
    if metric == "l1":
        # KDTree dùng p=1 cho L1
        distances, indices = tree.query(query_vector, k=TOP_K, p=1)
    else:
        # Mặc định p=2 cho L2
        distances, indices = tree.query(query_vector, k=TOP_K, p=2)
    t_search = time.time() - t0

    print(f"    Thời gian tìm kiếm   : {t_search*1000:.2f}ms")

    # Bước 3: Bảng khoảng cách trung gian
    print(f"\n[3] Khoảng cách đến {TOP_K} kết quả:")
    print(f"    {'#':<4} {'Khoảng cách':>14}  Tên bài")
    print(f"    {'-'*50}")
    for rank, (dist, idx) in enumerate(zip(distances, indices), 1):
        song = songs[idx]
        marker = " ← CHÍNH NÓ" if dist < 1e-3 else ""
        print(f"    {rank:<4} {dist:>14.6f}  {song['name']}{marker}")

    # Bước 4: Kết quả cuối
    print(f"\n[4] KẾT QUẢ TOP {TOP_K} BÀI NHẠC TƯƠNG ĐỒNG:")
    print(f"    {'Hạng':<6} {'Độ tương đồng':>14}  {'Tên bài'}")
    print(f"    {'-'*55}")

    results = []
    max_dist = distances[-1] if distances[-1] > 0 else 1.0
    for rank, (dist, idx) in enumerate(zip(distances, indices), 1):
        song       = songs[idx]
        similarity = max(0.0, 1.0 - dist / max_dist) * 100
        results.append({**song, "distance": dist, "similarity": similarity})
        print(f"    {rank:<6} {similarity:>13.2f}%  {song['name']}")
        print(f"           {'File:':10} {song['path']}")

    print("=" * 55)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python search.py <file_audio.wav> [l1|l2]")
        print("Ví dụ    : python search.py query.wav l2")
        sys.exit(1)

    query_path = sys.argv[1]
    metric     = sys.argv[2].lower() if len(sys.argv) > 2 else "l2"

    if not os.path.exists(query_path):
        print(f"Không tìm thấy file: {query_path}")
        sys.exit(1)

    search(query_path, metric=metric)

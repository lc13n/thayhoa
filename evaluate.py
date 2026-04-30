import sys
import os
import time
import pickle
import sqlite3
import numpy as np
from scipy.spatial import KDTree
from feature_extraction import extract_features
from search import search, load_kdtree, TOP_K

DB_FILE     = "music_database.db"
KDTREE_FILE = "kdtree.pkl"


# ─── Precision & Recall ────────────────────────────────────────────────────────

def precision_recall(results, query_name, relevant_names):
    """
    results        : danh sách kết quả trả về từ search()
    query_name     : tên bài truy vấn
    relevant_names : danh sách tên bài được coi là đúng (cùng bài/thể loại)
    """
    retrieved = [r["name"] for r in results]

    # Correct accepted (ca): lấy đúng
    ca = sum(1 for r in retrieved if r in relevant_names)
    # False accepted (fa): lấy sai
    fa = len(retrieved) - ca
    # False dismissed (fd): đúng nhưng không lấy
    total_relevant = len(relevant_names)
    fd = total_relevant - ca

    precision = ca / (ca + fa) if (ca + fa) > 0 else 0.0
    recall    = ca / (ca + fd) if (ca + fd) > 0 else 0.0

    return {
        "ca": ca, "fa": fa, "fd": fd,
        "precision": precision,
        "recall": recall
    }


# ─── So sánh tốc độ KD-Tree vs Brute-force ────────────────────────────────────

def benchmark_search(query_vector, songs, runs=5):
    vectors = np.array([s["features"] for s in songs], dtype=np.float32)

    # Brute-force: tính L2 với toàn bộ DB
    t0 = time.time()
    for _ in range(runs):
        dists = np.linalg.norm(vectors - query_vector, axis=1)
        top_idx = np.argsort(dists)[:TOP_K]
    t_brute = (time.time() - t0) / runs

    # KD-Tree
    tree = KDTree(vectors)
    t0 = time.time()
    for _ in range(runs):
        tree.query(query_vector, k=TOP_K, p=2)
    t_kdtree = (time.time() - t0) / runs

    return t_brute, t_kdtree


# ─── Demo Case 1: file CÓ trong DB ────────────────────────────────────────────

def demo_in_db(query_path, relevant_names=None):
    print("\n" + "█" * 55)
    print("  DEMO CASE 1: FILE ĐÃ CÓ TRONG DB")
    print("█" * 55)

    results = search(query_path, metric="l2")

    query_name = os.path.splitext(os.path.basename(query_path))[0].replace("_", " ")

    # Kiểm tra top 1 phải là chính nó (khoảng cách = 0)
    top1 = results[0]
    print(f"\n[KIỂM TRA] Top 1 có phải chính nó không?")
    if top1["distance"] < 1e-3:
        print(f"  ✓ ĐÚNG — '{top1['name']}' với khoảng cách = {top1['distance']:.6f}")
    else:
        print(f"  ✗ SAI  — Top 1 là '{top1['name']}' (distance={top1['distance']:.4f})")

    # Precision & Recall
    if relevant_names is None:
        relevant_names = [query_name]  # mặc định chỉ chính nó là đúng

    metrics = precision_recall(results, query_name, relevant_names)
    print(f"\n[ĐÁNH GIÁ]")
    print(f"  Correct Accepted (ca) : {metrics['ca']}")
    print(f"  False Accepted   (fa) : {metrics['fa']}")
    print(f"  False Dismissed  (fd) : {metrics['fd']}")
    print(f"  Precision             : {metrics['precision']:.2%}")
    print(f"  Recall                : {metrics['recall']:.2%}")

    return results, metrics


# ─── Demo Case 2: file NGOÀI DB ───────────────────────────────────────────────

def demo_out_db(query_path, relevant_names=None):
    print("\n" + "█" * 55)
    print("  DEMO CASE 2: FILE CHƯA CÓ TRONG DB")
    print("█" * 55)

    results = search(query_path, metric="l2")

    query_name = os.path.splitext(os.path.basename(query_path))[0].replace("_", " ")

    print(f"\n[KIỂM TRA] Top 1 không phải chính nó (file ngoài DB)")
    top1 = results[0]
    print(f"  → Bài gần nhất: '{top1['name']}' (distance={top1['distance']:.4f})")

    # Precision & Recall (cần truyền relevant_names thủ công)
    if relevant_names:
        metrics = precision_recall(results, query_name, relevant_names)
        print(f"\n[ĐÁNH GIÁ]")
        print(f"  Correct Accepted (ca) : {metrics['ca']}")
        print(f"  False Accepted   (fa) : {metrics['fa']}")
        print(f"  False Dismissed  (fd) : {metrics['fd']}")
        print(f"  Precision             : {metrics['precision']:.2%}")
        print(f"  Recall                : {metrics['recall']:.2%}")
    else:
        print("  (Không có danh sách relevant → bỏ qua Precision/Recall)")
        metrics = None

    return results, metrics


# ─── Benchmark tốc độ ─────────────────────────────────────────────────────────

def demo_benchmark(query_path):
    print("\n" + "█" * 55)
    print("  BENCHMARK: KD-TREE vs BRUTE-FORCE")
    print("█" * 55)

    print("\n  Trích xuất vector query...")
    query_vector = np.array(extract_features(query_path), dtype=np.float32)

    _, songs = load_kdtree()
    print(f"  Số bài trong DB : {len(songs)}")
    print(f"  Chạy benchmark (5 lần / phương pháp)...\n")

    t_brute, t_kdtree = benchmark_search(query_vector, songs, runs=5)

    speedup = t_brute / t_kdtree if t_kdtree > 0 else 0
    print(f"  {'Phương pháp':<20} {'Thời gian TB':>14}")
    print(f"  {'-'*36}")
    print(f"  {'Brute-force':<20} {t_brute*1000:>11.3f} ms")
    print(f"  {'KD-Tree':<20} {t_kdtree*1000:>11.3f} ms")
    print(f"  {'-'*36}")
    print(f"  KD-Tree nhanh hơn : {speedup:.1f}x")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HƯỚNG DẪN SỬ DỤNG evaluate.py")
    print("=" * 55)
    print("""
from evaluate import demo_in_db, demo_out_db, demo_benchmark

# Case 1: file CÓ trong DB
demo_in_db("dataset_beat/Hoa_Hải_Đường.wav")

# Case 2: file NGOÀI DB (truyền thêm relevant nếu biết)
demo_out_db("query_ngoai_db.wav", relevant_names=["Hoa Hải Đường"])

# Benchmark tốc độ
demo_benchmark("dataset_beat/Hoa_Hải_Đường.wav")
""")
    print("Hoặc chạy trực tiếp:")
    print("  python evaluate.py in   dataset_beat/Hoa_Hải_Đường.wav")
    print("  python evaluate.py out  query_ngoai_db.wav")
    print("  python evaluate.py bench dataset_beat/Hoa_Hải_Đường.wav")

    if len(sys.argv) >= 3:
        mode = sys.argv[1]
        path = sys.argv[2]
        if mode == "in":
            demo_in_db(path)
        elif mode == "out":
            relevant = sys.argv[3].split(",") if len(sys.argv) > 3 else None
            demo_out_db(path, relevant_names=relevant)
        elif mode == "bench":
            demo_benchmark(path)

import sys
import os
import time
import numpy as np
from feature_extraction import extract_features_from_file, SR, WINDOW_SEC
from search import search, load_songs, weighted_distance, WEIGHTS, TOP_K, load_query


# Precision & Recall
def precision_recall(results, relevant_names):
    retrieved = [r["name"] for r in results]
    ca = sum(1 for r in retrieved if r in relevant_names)
    fa = len(retrieved) - ca
    fd = len(relevant_names) - ca
    precision = ca / (ca + fa) if (ca + fa) > 0 else 0.0
    recall    = ca / (ca + fd) if (ca + fd) > 0 else 0.0
    return {"ca": ca, "fa": fa, "fd": fd, "precision": precision, "recall": recall}


# Benchmark: Weighted Normalized vs L2 brute-force
def benchmark_search(query_vector, songs, runs=5):
    vectors = np.array([s["features"] for s in songs], dtype=np.float32)

    # Weighted Normalized
    t0 = time.time()
    for _ in range(runs):
        denom = np.maximum(np.abs(query_vector), np.abs(vectors))
        denom[denom == 0] = 1e-9
        dists_wn = np.sum(np.abs(query_vector - vectors) / denom * WEIGHTS, axis=1)
        np.argsort(dists_wn)[:TOP_K]
    t_wn = (time.time() - t0) / runs

    # L2 brute-force (de so sanh)
    t0 = time.time()
    for _ in range(runs):
        dists_l2 = np.linalg.norm(vectors - query_vector, axis=1)
        np.argsort(dists_l2)[:TOP_K]
    t_l2 = (time.time() - t0) / runs

    return t_wn, t_l2


# Demo Case 1: file CO trong DB
def demo_in_db(query_path, relevant_names=None):
    print("\n" + "=" * 55)
    print("  DEMO CASE 1: FILE DA CO TRONG DB")
    print("=" * 55)

    results = search(query_path)
    query_name = os.path.splitext(os.path.basename(query_path))[0].replace("_", " ")

    top1 = results[0]
    print(f"\n[KIEM TRA] Top 1 co phai chinh no khong?")
    if top1["distance"] < 1e-3:
        print(f"  DUNG — '{top1['name']}' voi khoang cach = {top1['distance']:.6f}")
    else:
        print(f"  SAI  — Top 1 la '{top1['name']}' (distance={top1['distance']:.4f})")

    if relevant_names is None:
        relevant_names = [query_name]

    metrics = precision_recall(results, relevant_names)
    print(f"\n[DANH GIA]")
    print(f"  Correct Accepted (ca) : {metrics['ca']}")
    print(f"  False Accepted   (fa) : {metrics['fa']}")
    print(f"  False Dismissed  (fd) : {metrics['fd']}")
    print(f"  Precision             : {metrics['precision']:.2%}")
    print(f"  Recall                : {metrics['recall']:.2%}")

    return results, metrics


# Demo Case 2: file NGOAI DB
def demo_out_db(query_path, relevant_names=None):
    print("\n" + "=" * 55)
    print("  DEMO CASE 2: FILE CHUA CO TRONG DB")
    print("=" * 55)

    results = search(query_path)
    top1    = results[0]
    print(f"\n[KIEM TRA] Bai gan nhat: '{top1['name']}' (distance={top1['distance']:.4f})")

    if relevant_names:
        metrics = precision_recall(results, relevant_names)
        print(f"\n[DANH GIA]")
        print(f"  Correct Accepted (ca) : {metrics['ca']}")
        print(f"  False Accepted   (fa) : {metrics['fa']}")
        print(f"  False Dismissed  (fd) : {metrics['fd']}")
        print(f"  Precision             : {metrics['precision']:.2%}")
        print(f"  Recall                : {metrics['recall']:.2%}")
    else:
        print("  (Khong co danh sach relevant → bo qua Precision/Recall)")
        metrics = None

    return results, metrics


# Benchmark toc do
def demo_benchmark(query_path):
    print("\n" + "=" * 55)
    print("  BENCHMARK: Weighted Normalized vs L2")
    print("=" * 55)

    print("\n  Trich xuat vector query...")
    audio, sr    = load_query(query_path)
    query_vector = np.array(extract_features_from_file(query_path), dtype=np.float32)

    songs = load_songs()
    print(f"  So windows trong DB : {len(songs)}")
    print(f"  Chay benchmark (5 lan / phuong phap)...\n")

    t_wn, t_l2 = benchmark_search(query_vector, songs, runs=5)

    print(f"  {'Phuong phap':<25} {'Thoi gian TB':>14}")
    print(f"  {'-'*41}")
    print(f"  {'Weighted Normalized':<25} {t_wn*1000:>11.3f} ms")
    print(f"  {'L2 Brute-force':<25} {t_l2*1000:>11.3f} ms")


if __name__ == "__main__":
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

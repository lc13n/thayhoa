import sys
import os

def usage():
    print("""
╔══════════════════════════════════════════════════════╗
║        HỆ THỐNG TÌM KIẾM NHẠC THEO NỘI DUNG        ║
╚══════════════════════════════════════════════════════╝

Cách dùng:
  python main.py build                        → Trích xuất đặc trưng + build DB
  python main.py search <file.wav> [l1|l2]   → Tìm 5 bài giống nhất
  python main.py demo_in   <file.wav>         → Demo file có trong DB
  python main.py demo_out  <file.wav> [tên,tên,...] → Demo file ngoài DB
  python main.py bench     <file.wav>         → Benchmark KD-Tree vs Brute-force

Ví dụ:
  python main.py build
  python main.py search dataset_beat/Hoa_Hải_Đường.wav
  python main.py demo_in  dataset_beat/Hoa_Hải_Đường.wav
  python main.py demo_out query.wav "Hoa Hải Đường,Nơi Này Có Anh"
  python main.py bench    dataset_beat/Hoa_Hải_Đường.wav
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    # ── Build database ──────────────────────────────────────────────────────
    if cmd == "build":
        from feature_extraction import build_database
        build_database()

    # ── Search ─────────────────────────────────────────────────────────────
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Thiếu file: python main.py search <file.wav> [l1|l2]")
            sys.exit(1)
        from search import search
        query  = sys.argv[2]
        metric = sys.argv[3].lower() if len(sys.argv) > 3 else "l2"
        if not os.path.exists(query):
            print(f"Không tìm thấy file: {query}")
            sys.exit(1)
        search(query, metric=metric)

    # ── Demo file có trong DB ───────────────────────────────────────────────
    elif cmd == "demo_in":
        if len(sys.argv) < 3:
            print("Thiếu file: python main.py demo_in <file.wav>")
            sys.exit(1)
        from evaluate import demo_in_db
        demo_in_db(sys.argv[2])

    # ── Demo file ngoài DB ──────────────────────────────────────────────────
    elif cmd == "demo_out":
        if len(sys.argv) < 3:
            print("Thiếu file: python main.py demo_out <file.wav> [tên,tên,...]")
            sys.exit(1)
        from evaluate import demo_out_db
        relevant = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        demo_out_db(sys.argv[2], relevant_names=relevant)

    # ── Benchmark ───────────────────────────────────────────────────────────
    elif cmd == "bench":
        if len(sys.argv) < 3:
            print("Thiếu file: python main.py bench <file.wav>")
            sys.exit(1)
        from evaluate import demo_benchmark
        demo_benchmark(sys.argv[2])

    else:
        print(f"Lệnh không hợp lệ: '{cmd}'")
        usage()

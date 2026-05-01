import sys
import os

def usage():
    print("""
HE THONG TIM KIEM NHAC THEO NOI DUNG (CBMIR)

Cach dung:
  python main.py build                     → Trich xuat dac trung + build DB
  python main.py search <file.wav>         → Tim 5 bai giong nhat
  python main.py demo_in   <file.wav>      → Demo file co trong DB
  python main.py demo_out  <file.wav> [ten,ten,...] → Demo file ngoai DB
  python main.py bench     <file.wav>      → Benchmark Weighted vs L2

Vi du:
  python main.py build
  python main.py search dataset_beat/Hoa_Hai_Duong.wav
  python main.py demo_in  dataset_beat/Hoa_Hai_Duong.wav
  python main.py demo_out query.wav "Hoa Hai Duong,Noi Nay Co Anh"
  python main.py bench    dataset_beat/Hoa_Hai_Duong.wav
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "build":
        from feature_extraction import build_database
        build_database()

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py search <file.wav>")
            sys.exit(1)
        from search import search
        query = sys.argv[2]
        if not os.path.exists(query):
            print(f"Khong tim thay file: {query}")
            sys.exit(1)
        search(query)

    elif cmd == "demo_in":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py demo_in <file.wav>")
            sys.exit(1)
        from evaluate import demo_in_db
        demo_in_db(sys.argv[2])

    elif cmd == "demo_out":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py demo_out <file.wav> [ten,ten,...]")
            sys.exit(1)
        from evaluate import demo_out_db
        relevant = sys.argv[3].split(",") if len(sys.argv) > 3 else None
        demo_out_db(sys.argv[2], relevant_names=relevant)

    elif cmd == "bench":
        if len(sys.argv) < 3:
            print("Thieu file: python main.py bench <file.wav>")
            sys.exit(1)
        from evaluate import demo_benchmark
        demo_benchmark(sys.argv[2])

    else:
        print(f"Lenh khong hop le: '{cmd}'")
        usage()

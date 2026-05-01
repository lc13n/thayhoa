import subprocess
import sys
import time
import os
from datetime import datetime

GENRE_FILES = [
    "nhac_tre.txt",
    "nhac_vang.txt",
    "nhac_do.txt",
    "nhac_quan_ho_dan_ca_bac_bo.txt",
    "nhac_chau_van.txt",
    "nhac_cai_luong.txt",
    "nhac_dan_ca_nam_bo.txt",
    "nhac_remix.txt",
]

LOG_FILE = "download_all_log.txt"


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def count_wav(prefix):
    if not os.path.exists("dataset_beat"):
        return 0
    return sum(
        1 for f in os.listdir("dataset_beat")
        if f.startswith(prefix) and f.endswith(".wav") and not f.startswith("_temp")
    )


def run_download(genre_file):
    prefix = os.path.splitext(genre_file)[0].lower()
    total  = count_lines(genre_file)

    print(f"\n{'='*60}")
    print(f"  THE LOAI : {genre_file}")
    print(f"  SO BAI   : {total}")
    print(f"  BAT DAU  : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "download_beat.py", genre_file],
        text=True
    )
    elapsed = time.time() - t0

    downloaded = count_wav(prefix)
    status = "HOAN THANH" if result.returncode == 0 else "LOI"

    summary = {
        "file":       genre_file,
        "total":      total,
        "downloaded": downloaded,
        "elapsed":    elapsed,
        "status":     status,
    }

    print(f"\n  Ket qua  : {downloaded}/{total} bai")
    print(f"  Thoi gian: {elapsed/60:.1f} phut")
    print(f"  Trang thai: {status}")

    return summary


def main():
    # Kiem tra cac file ton tai
    missing = [f for f in GENRE_FILES if not os.path.exists(f)]
    if missing:
        print("Khong tim thay cac file sau:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         TAI NHAC BEAT KARAOKE - TAT CA THE LOAI         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Tong so the loai : {len(GENRE_FILES)}")
    print(f"  Tong so bai      : {sum(count_lines(f) for f in GENRE_FILES)}")
    print(f"  Bat dau luc      : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    t_start   = time.time()
    summaries = []

    for genre_file in GENRE_FILES:
        summary = run_download(genre_file)
        summaries.append(summary)

    # In tong ket
    total_time = time.time() - t_start
    total_songs = sum(s["total"]      for s in summaries)
    total_dl    = sum(s["downloaded"] for s in summaries)

    report = []
    report.append("\n" + "="*60)
    report.append("  TONG KET DOWNLOAD")
    report.append("="*60)
    report.append(f"  {'The loai':<40} {'Tai duoc':>8}  {'Tong':>5}  {'Ti le':>6}")
    report.append(f"  {'-'*60}")
    for s in summaries:
        rate = s["downloaded"] / s["total"] * 100 if s["total"] > 0 else 0
        report.append(f"  {s['file']:<40} {s['downloaded']:>8}  {s['total']:>5}  {rate:>5.0f}%")
    report.append(f"  {'-'*60}")
    report.append(f"  {'TONG CONG':<40} {total_dl:>8}  {total_songs:>5}  {total_dl/total_songs*100:>5.0f}%")
    report.append(f"\n  Thoi gian tong   : {total_time/60:.1f} phut")
    report.append(f"  Ket thuc luc     : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append("="*60)

    print("\n".join(report))

    # Ghi log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n  Log da luu: {LOG_FILE}")


if __name__ == "__main__":
    main()

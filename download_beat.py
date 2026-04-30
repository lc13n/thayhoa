import os
import sys
import subprocess
import time
import yt_dlp

if len(sys.argv) < 2:
    print("Cách dùng: python download_beat.py <TenFile.txt>")
    print("Ví dụ:     python download_beat.py Nhac_Tre.txt")
    sys.exit(1)

INPUT_FILE  = sys.argv[1]
# Lấy tên file không có đuôi, chuyển về chữ thường làm prefix
CATEGORY    = os.path.splitext(os.path.basename(INPUT_FILE))[0].lower()
OUTPUT_DIR  = "dataset_beat"
DURATION    = 60        # giây
TARGET_SR   = 44100
LOG_FILE    = f"{CATEGORY}_download_log.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_song_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        songs = [line.strip() for line in f if line.strip()]
    return songs


def search_and_download(song_name, output_dir):
    query = f"beat karaoke {song_name}"
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    temp_path = os.path.join(output_dir, f"_temp_{safe_name}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_path + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "default_search": "ytsearch1",  # tìm 1 kết quả đầu tiên trên YouTube
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"  → Đang tìm: \"{query}\"")
            ydl.download([f"ytsearch1:{query}"])
        return temp_path + ".wav"
    except Exception as e:
        print(f"  ✗ Lỗi tải: {e}")
        return None


def cut_audio(input_wav, output_wav, duration=DURATION):
    # Dùng ffmpeg: lấy 60 giây từ đầu, chuẩn hóa 44100Hz mono 16-bit
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-t", str(duration),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_wav
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ✗ Lỗi ffmpeg: {result.stderr.decode()[-200:]}")
        return False
    return True


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Không tìm thấy file: {INPUT_FILE}")
        sys.exit(1)

    songs = read_song_list(INPUT_FILE)
    total = len(songs)
    print(f"File   : {INPUT_FILE}")
    print(f"Prefix : {CATEGORY}")
    print(f"Số bài : {total}\n")

    success_list = []
    fail_list    = []

    for idx, song in enumerate(songs, 1):
        print(f"[{idx}/{total}] Xử lý: {song}")
        safe_name   = song.replace(" ", "_").replace("/", "-")
        output_wav  = os.path.join(OUTPUT_DIR, f"{CATEGORY}_{safe_name}.wav")

        # Bỏ qua nếu đã tải
        if os.path.exists(output_wav):
            print(f"  ✓ Đã có sẵn, bỏ qua.")
            success_list.append(song)
            continue

        # Tải về
        temp_wav = search_and_download(song, OUTPUT_DIR)
        if not temp_wav or not os.path.exists(temp_wav):
            print(f"  ✗ Không tải được: {song}")
            fail_list.append(song)
            continue

        # Cắt 1 phút + chuẩn hóa
        ok = cut_audio(temp_wav, output_wav)
        if ok:
            os.remove(temp_wav)  # xóa file tạm
            size_kb = os.path.getsize(output_wav) // 1024
            print(f"  ✓ Lưu: {output_wav} ({size_kb} KB)")
            success_list.append(song)
        else:
            fail_list.append(song)

        time.sleep(1)  # tránh bị YouTube chặn
        print()

    # Ghi log kết quả
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== KẾT QUẢ TẢI ===\n")
        f.write(f"Thành công: {len(success_list)}/{total}\n\n")
        f.write("THÀNH CÔNG:\n")
        for s in success_list:
            f.write(f"  ✓ {s}\n")
        f.write("\nTHẤT BẠI:\n")
        for s in fail_list:
            f.write(f"  ✗ {s}\n")

    print("=" * 50)
    print(f"Hoàn thành: {len(success_list)}/{total} bài")
    print(f"File lưu tại: ./{OUTPUT_DIR}/")
    print(f"Log chi tiết: ./{LOG_FILE}")


if __name__ == "__main__":
    main()

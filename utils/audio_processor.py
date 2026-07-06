import yt_dlp
import subprocess

import glob
import os

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR,exist_ok = True)

def download_youtube_audio(url:str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename



def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        output_path,
    ]

    subprocess.run(command, check=True)

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10):
    chunk_dir = os.path.splitext(wav_path)[0] + "_chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    output_pattern = os.path.join(chunk_dir, "chunk_%03d.wav")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-f", "segment",
        "-segment_time", str(chunk_minutes * 60),
        "-c", "copy",
        output_pattern,
    ], check=True)

    return sorted(glob.glob(os.path.join(chunk_dir, "*.wav")))

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
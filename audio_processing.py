import subprocess
import os

def convert_mp4_to_wav(mp4_file, wav_file):
    command = f"ffmpeg -i {mp4_file} -q:a 0 -map a {wav_file}"
    subprocess.call(command, shell=True)
    print(f"Converted {mp4_file} to {wav_file}")

def split_wav_file(wav_file, chunk_length=600):
    command = f"ffmpeg -i {wav_file} -f segment -segment_time {chunk_length} -c copy chunk_%03d.wav"
    subprocess.call(command, shell=True)
    chunks = [f for f in os.listdir('.') if f.startswith('chunk_') and f.endswith('.wav')]
    print(f"Split WAV file into chunks: {chunks}")
    return chunks

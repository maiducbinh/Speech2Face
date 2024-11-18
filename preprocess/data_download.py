import pandas as pd
import os
import subprocess
import argparse
from speaker import *
from video_generator import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def main():
    videos_path = "data/videos/"
    audios_path = "data/audios/"
    data = pd.read_csv("avspeech_train.csv", header=None, names=["id", "start", "end", "x", "y"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--from_id", type=int, default=30000)
    parser.add_argument("--to_id", type=int, default=80000)
    parser.add_argument("--low_memory", type=str, default="no")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--mono", type=str, default=True)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--stride", type=int, default=160)
    parser.add_argument("--fft_length", type=int, default=512)
    parser.add_argument("--amp_norm", type=int, default=0.3)
    parser.add_argument("--face_extraction_model", type=str, default="cnn")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sb = Speaker(
        sample_rate=args.sample_rate,
        duration=args.duration,
        mono=args.mono,
        window=args.window,
        stride=args.stride,
        fft_length=args.fft_length,
        amp_norm=args.amp_norm,
        verbose=args.verbose
    )

    vs = VideoExtract(args.fps, args.duration, args.face_extraction_model, args.verbose)

    for i in range(args.from_id, args.to_id):
        video_id = data.loc[i, "id"]
        start_time = float(data.loc[i, "start"])
        end_time = float(data.loc[i, "end"])
        duration = min(args.duration, end_time - start_time)
        output_file = os.path.join(videos_path, f"{video_id}.mp4")

        if not os.path.isfile(output_file):
            if args.verbose:
                print(f"Downloading video {video_id} from {start_time}s to {start_time + duration}s")

            # Construct the yt-dlp command
            download_cmd = [
                "yt-dlp",
                "-f", "bestvideo+bestaudio",
                "--merge-output-format", "mp4",
                "--external-downloader", "ffmpeg",
                "--external-downloader-args", f"ffmpeg_i:-ss {start_time} -t {duration}",
                "-o", output_file,
                f"https://www.youtube.com/watch?v={video_id}"
            ]

            res = subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                if args.verbose:
                    print(f"Failed to download video {video_id}: {res.stderr.decode()}")
                continue
        else:
            if args.verbose:
                print(f"Skipping download; {output_file} already exists.")

        result = vs.extract_video(video_id, data.loc[i, "x"], data.loc[i, "y"])
        if result == 1:
            if args.verbose:
                print("Video embedding extraction failed; see logs for details.")
            continue

        error = sb.extract_wav(video_id)
        if error == 1:
            if args.verbose:
                print("Audio extraction failed; skipping to next video.")
            continue

    print("Done creating dataset")
    if args.low_memory == "yes":
        delete_audios = f"rm {audios_path}*"
        delete_videos = f"rm {videos_path}*"
        subprocess.run(delete_audios, shell=True)
        subprocess.run(delete_videos, shell=True)

if __name__ == "__main__":
    main()

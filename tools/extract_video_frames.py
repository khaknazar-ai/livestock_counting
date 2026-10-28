"""
tools/extract_video_frames.py

Extract frames from livestock videos for dataset building.

Experiment note: extract representative frames from local or public livestock videos.

Downloads video (if URL given) and extracts 1 frame every N seconds.
Useful for YouTube/Bilibili livestock footage.

Prerequisites:
    pip install yt-dlp

Usage:
    # From local video file:
    python tools/extract_video_frames.py \
        --video path/to/livestock_video.mp4 \
        --out_dir data/raw/video_001 \
        --interval 3

    # From YouTube URL (requires yt-dlp):
    python tools/extract_video_frames.py \
        --url "https://www.youtube.com/watch?v=XXXXX" \
        --out_dir data/raw/youtube_001 \
        --interval 2
"""

import os
import argparse
import subprocess
import cv2


def download_video(url, out_path):
    """Download video using yt-dlp."""
    print(f"Downloading: {url}")
    cmd = [
        'yt-dlp', '-f', 'best[height<=720]',
        '-o', out_path, url
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {out_path}")


def extract_frames(video_path, out_dir, interval_sec=3, prefix="frame"):
    """Extract 1 frame every interval_sec seconds."""
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    skip = int(fps * interval_sec)

    print(f"Video: {fps:.1f} fps, {total} frames, {duration:.0f}s")
    print(f"Extracting 1 frame every {interval_sec}s (skip {skip} frames)")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip == 0:
            name = f"{prefix}_{saved:04d}.jpg"
            path = os.path.join(out_dir, name)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames to {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default=None,
                        help='Path to local video file')
    parser.add_argument('--url', default=None,
                        help='YouTube/Bilibili URL (requires yt-dlp)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for extracted frames')
    parser.add_argument('--interval', type=float, default=3,
                        help='Extract 1 frame every N seconds')
    parser.add_argument('--prefix', default='frame',
                        help='Filename prefix for extracted frames')
    args = parser.parse_args()

    if args.url:
        tmp_video = os.path.join(args.out_dir, '_temp_video.mp4')
        os.makedirs(args.out_dir, exist_ok=True)
        download_video(args.url, tmp_video)
        args.video = tmp_video

    if not args.video:
        print("Error: provide --video or --url")
        return

    extract_frames(args.video, args.out_dir, args.interval, args.prefix)


if __name__ == '__main__':
    main()

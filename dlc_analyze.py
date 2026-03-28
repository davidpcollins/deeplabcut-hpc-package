#!/usr/bin/env python3
"""
dlc_analyze.py — Batch-analyse videos with a trained DLC 3 model.

Supports two modes:
  1) Analyse ALL videos in a directory
  2) Analyse a specific subset via SLURM array indexing

Usage:
    # All .avi videos in a directory
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype .avi

    # Mixed video types
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype .avi .mp4

    # Single video by SLURM_ARRAY_TASK_ID (for array jobs)
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype .avi .mp4 --array_mode
"""

import argparse
import os
import sys
from pathlib import Path


def find_videos(video_dir, videotypes):
    """Recursively find videos matching any of the given extensions."""
    vdir = Path(video_dir)
    videos = set()
    for ext in videotypes:
        # Handle with or without leading dot
        ext = ext if ext.startswith(".") else f".{ext}"
        videos.update(str(v.resolve()) for v in vdir.rglob(f"*{ext}"))
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-analyse videos with a trained DLC 3 model."
    )
    parser.add_argument(
        "config_path", 
        help="Path to DLC project config.yaml"
    )
    parser.add_argument(
        "--video_dir", required=True, 
        help="Directory containing videos"
    )
    parser.add_argument(
        "--videotype", nargs="+", default=[".mp4"],
        help="Video extension(s) to search for (default: .mp4). "
             "Example: --videotype .avi .mp4",
    )
    parser.add_argument(
        "--shuffle", type=int, default=1,
        help="DLC shuffle number to use for analysis (default: 1)",
    )
    parser.add_argument(
        "--trainingsetindex", type=int, default=0,
        help="DLC training set index to use for analysis (default: 0)",
    )
    parser.add_argument(
        "--filter", action="store_true", default=True,
        help="Apply median filter after analysis (default: True)",
    )
    parser.add_argument(
        "--dynamic_cropping", action="store_true", default=False, 
        help="Enable dynamic cropping to increase speed"
    )
    parser.add_argument(
        "--no-filter", dest="filter", action="store_false",
        help="Skip median filtering after analysis (default: False)",
    )
    parser.add_argument(
        "--create_video", action="store_true", default=False,
        help="Create labelled output videos",
    )
    parser.add_argument(
        "--array_mode", action="store_true", default=False,
        help="Process a single video indexed by SLURM_ARRAY_TASK_ID",
    )
    args = parser.parse_args()

    import deeplabcut
    import torch

    print("=" * 60)
    print(f"  DeepLabCut {deeplabcut.__version__}")
    print(f"  CUDA avail {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU        {torch.cuda.get_device_name(0)}")
    print(f"  Config     {args.config_path}")
    print(f"  Video dir  {args.video_dir}")
    print(f"  Video type {args.videotype}")
    print("=" * 60)

    # ── Discover videos ─────────────────────────────────────────────────
    all_videos = find_videos(args.video_dir, args.videotype)
    print(f"Found {len(all_videos)} video(s) total")

    if not all_videos:
        print("ERROR: No videos found. Check --video_dir and --videotype.")
        sys.exit(1)

    # ── Array mode: pick one video ──────────────────────────────────────
    if args.array_mode:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if task_id >= len(all_videos):
            print(f"SLURM_ARRAY_TASK_ID={task_id} exceeds video count ({len(all_videos)}). Exiting.")
            sys.exit(0)
        videos_to_process = [all_videos[task_id]]
        print(f"Array mode: processing video index {task_id} → {videos_to_process[0]}")
    else:
        videos_to_process = all_videos

    # ── Analyse ─────────────────────────────────────────────────────────
    print(f"\nAnalysing {len(videos_to_process)} video(s)...")
    deeplabcut.analyze_videos(
        args.config_path,
        videos_to_process,
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        save_as_csv=True,
    )

    # ── Filter ──────────────────────────────────────────────────────────
    if args.filter:
        print("\nFiltering predictions...")
        deeplabcut.filterpredictions(
            args.config_path,
            videos_to_process,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            filtertype="median", # Can edit this if you want to use a different filter (supported: median, arima, spline)
            windowlength=5,
        )

    # ── Create labelled videos ───────────────────────────────────────────
    if args.create_video:
        print("\nCreating labelled videos...")
        deeplabcut.create_labeled_video(
            args.config_path,
            videos_to_process,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            filtered=args.filter,
            draw_skeleton=True,
            color_by="bodypart",
        )

    print("\n✓ Video analysis complete.")


if __name__ == "__main__":
    main()

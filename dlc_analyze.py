#!/usr/bin/env python3
"""
dlc_analyze.py — Batch-analyse videos with a trained DLC 3 model.

Supports two modes:
  1) Analyse videos in a directory sequentially
  2) Analyse a specific subset via SLURM array indexing

Usage:
    # All .mp4 videos in a directory
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype .mp4

    # Mixed video types
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype ".mp4,.avi"

    # Single video by SLURM_ARRAY_TASK_ID (for array jobs)
    python dlc_analyze.py /path/to/config.yaml --video_dir /data/videos --videotype ".avi,.mp4" --array_mode

To do:
- [ ] Add kwargs for filtering
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
    parser.add_argument("config_path", help="Path to DLC project config.yaml")
    parser.add_argument(
        "--video_dir", required=True, help="Directory containing videos"
    )
    parser.add_argument(
        "--videotype",
        default=".mp4",
        help="Comma-separated string of video extensions, e.g. '.mp4,.avi,.mov' (default: '.mp4') ",
    )
    parser.add_argument(
        "--video_path", default=None, help="Process one explicit video using path"
    )
    parser.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="DLC shuffle number to use for analysis (default: 1)",
    )
    parser.add_argument(
        "--trainingsetindex",
        type=int,
        default=0,
        help="DLC training set index to use for analysis (default: 0)",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        default=False,
        help="Apply median filter after analysis (default: False)",
    )
    parser.add_argument(
        "--filter_type",
        default="median",
        help="Type of filter to apply (default: 'median', other options: 'arima', 'spline')",
    )
    parser.add_argument(
        "--dynamic_cropping",
        action="store_true",
        default=False,
        help="Enable dynamic cropping to increase speed",
    )
    parser.add_argument(
        "--create_video",
        action="store_true",
        default=False,
        help="Create labelled output videos",
    )
    parser.add_argument(
        "--array_mode",
        action="store_true",
        default=False,
        help="Process a single video indexed by SLURM_ARRAY_TASK_ID",
    )
    parser.add_argument(
        "--show_gpu",
        action="store_true",
        default=False,
        help="Show GPU usage information (default: False)",
    )
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        default=False,
        help="Skip analysis step and only do filtering/video creation (default: False)",
    )
    args = parser.parse_args()

    # Keep track of which process is printing (for multi-GPU runs)
    process = os.environ.get("CUDA_VISIBLE_DEVICES")

    # Process videotype into a list of extensions
    videotype = [ext.strip() for ext in args.videotype.split(",") if ext.strip()]
    args.videotype = videotype

    import deeplabcut
    import torch

    print("=" * 60)
    print(f"  DeepLabCut  {deeplabcut.__version__}")
    print(f"  CUDA avail  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU         {torch.cuda.get_device_name(0)}")
        print(f"  Properties: {torch.cuda.get_device_properties(0)}")
        print(f"  CUDA_VISIBLE_DEVICES = {process}")
    print(f"  Config      {args.config_path}")
    print(f"  Video dir   {args.video_dir}")
    print(f"  Video type  {args.videotype}")
    print("=" * 60)

    # ── Discover videos ─────────────────────────────────────────────────
    all_videos = find_videos(args.video_dir, args.videotype)
    if not args.video_path:
        print(f"{process}: Found {len(all_videos)} video(s) total")

    if not all_videos:
        print(f"{process}: ERROR: No videos found. Check --video_dir and --videotype.")
        sys.exit(1)

    # ── Case of passing a single video explicitely───────────────────────
    if args.video_path:
        print(f"{process}: Processing single video: {args.video_path}")
        videos_to_process = [args.video_path]
    # ── Array mode: pick one video ──────────────────────────────────────
    elif args.array_mode:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if task_id >= len(all_videos):
            print(
                f"{process}: SLURM_ARRAY_TASK_ID={task_id} exceeds video count ({len(all_videos)}). Exiting."
            )
            sys.exit(0)
        videos_to_process = [all_videos[task_id]]
        print(
            f"{process}: Array mode: processing video index {task_id} → {videos_to_process[0]}"
        )
    else:
        videos_to_process = all_videos

    # ── Analyse ─────────────────────────────────────────────────────────
    if not args.skip_analysis:
        print(f"\n{process}: Analysing {len(videos_to_process)} video(s)...")
        deeplabcut.analyze_videos(
            args.config_path,
            videos_to_process,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            dynamic=(
                args.dynamic_cropping,
                0.5,
                10,
            ),  # the 2nd-3rd args only matter if #1=True
            save_as_csv=True,
            show_gpu_memory=args.show_gpu,
        )

    # ── Filter ──────────────────────────────────────────────────────────
    if args.filter:
        print(f"\n{process}: Filtering predictions...")
        deeplabcut.filterpredictions(
            args.config_path,
            videos_to_process,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            filtertype=args.filter_type,
        )

    # ── Create labelled videos ───────────────────────────────────────────
    if args.create_video:
        print(f"\n{process}: Creating labelled videos...")
        deeplabcut.create_labeled_video(
            args.config_path,
            videos_to_process,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
            filtered=args.filter,
            draw_skeleton=True,
            color_by="bodypart",
            dotsize=3,  # it seems that create_labeled_video() ignores config.yaml on this
        )

    print(f"\n{process}: ✓ Video analysis complete.")


if __name__ == "__main__":
    main()

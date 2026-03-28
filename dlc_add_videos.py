#!/usr/bin/env python3
"""
dlc_add_videos.py — Add all video files from a directory to a DLC project on cluster storage
Usage:
    python dlc_add_videos.py --config_path /path/to/config.yaml --folder_to_add /path/to/videos --copy_videos --extract_frames
Notes:
    - By default, videos will be copied to the DLC project folder. Use --copy_videos False to reference videos in their original location instead (not recommended for cluster storage).
    - By default, frames will not be extracted. Use --extract_frames to extract frames from the videos when adding them to the project.
    - Supported video formats are determined by the list in the list_video_files() function (default: .mp4, .avi, .mov, .mkv).
    - This script uses the built-in deeplabcut.add_new_videos() function, which handles updating the config.yaml file and copying/extracting as needed.
"""
import argparse
import os
import sys
from pathlib import Path

# Helper function to get list of video files in a folder (including subfolders)
def list_video_files(folder, video_extensions=None):
    if video_extensions is None: # Otherwise you can pass in a custom list of video extensions
        video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    p = Path(folder)
    return [
        str(file.resolve()) for file in p.rglob("*") if file.is_file() and file.suffix.lower() in video_extensions
    ]

def main():
    parser = argparse.ArgumentParser(
        description="Add all video files from a directory to the project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
            help="Path to config.yaml file (default: None)",
    )
    parser.add_argument(
        "--folder_to_add", type=str, default=None,
        help="Path to directory containing video files to add to the project",
    )
    parser.add_argument(
        "--copy_videos", action="store_true", default=True,
        help="Whether to copy video files to the DLC project folder (default: True). If False, DLC will reference videos in their original location.",
    )
    parser.add_argument(
        "--extract_frames", action="store_true", default=False,
        help="Whether to extract frames from video files (default: False)",
    )

    args = parser.parse_args()

    # Check to make sure args are valid
    if args.config_path is None:
        print("Error: --config_path is required.")
        sys.exit(1)
    if args.folder_to_add is None:
        print("Error: --folder_to_add is required.")
        sys.exit(1) 
    if not os.path.isfile(args.config_path):
        print(f"Error: Config file '{args.config_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.folder_to_add):
        print(f"Error: Folder '{args.folder_to_add}' does not exist.")
        sys.exit(1)

    import deeplabcut

    # Get list of video files in the folder
    video_files = list_video_files(args.folder_to_add)
    if not video_files:
        print(f"No video files found in folder '{args.folder_to_add}'.")
        sys.exit(1)

    # Add video files using DLC built-in function
    deeplabcut.add_new_videos(
        args.config_path, # Path to config.yaml file
        video_files, # List of strings of video file paths to add
        copy_videos=args.copy_videos,
        extract_frames=args.extract_frames,
    )

if __name__ == "__main__":
    main()
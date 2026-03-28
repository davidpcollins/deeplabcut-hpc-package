#!/usr/bin/env python3
"""
dlc_head_direction_batch.py — Compute head direction from DLC .h5 files.

Standalone script (no notebook dependencies).  Replicates the core logic
from dlc_helpers.py so it can run on the cluster without importing that
module (though it will use it if available).

Usage:
    python dlc_head_direction_batch.py \
        --h5_dir /data/videos \
        --output_dir /results/head_direction \
        --fps 100
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def compute_head_direction(
    h5_path,
    nose_name="nose",
    left_ear_name="left_ear",
    right_ear_name="right_ear",
    likelihood_threshold=0.6,
    fps=100.0,
):
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]

    def _col(bp, coord):
        return df[(scorer, bp, coord)].values

    nose_x, nose_y, nose_p = _col(nose_name, "x"), _col(nose_name, "y"), _col(nose_name, "likelihood")
    lear_x, lear_y, lear_p = _col(left_ear_name, "x"), _col(left_ear_name, "y"), _col(left_ear_name, "likelihood")
    rear_x, rear_y, rear_p = _col(right_ear_name, "x"), _col(right_ear_name, "y"), _col(right_ear_name, "likelihood")

    ear_mid_x = (lear_x + rear_x) / 2.0
    ear_mid_y = (lear_y + rear_y) / 2.0

    dx = nose_x - ear_mid_x
    dy = nose_y - ear_mid_y

    angle_rad = np.arctan2(-dy, dx)
    angle_deg = np.degrees(angle_rad)

    min_p = np.minimum(np.minimum(nose_p, lear_p), rear_p)
    ok = min_p >= likelihood_threshold
    n_frames = len(nose_x)

    out = pd.DataFrame({
        "frame": np.arange(n_frames),
        "time_s": np.arange(n_frames) / fps,
        "nose_x": nose_x, "nose_y": nose_y,
        "ear_mid_x": ear_mid_x, "ear_mid_y": ear_mid_y,
        "head_dir_deg": np.where(ok, angle_deg, np.nan),
        "head_dir_rad": np.where(ok, angle_rad, np.nan),
        "min_likelihood": min_p,
        "likelihood_ok": ok,
    })

    pct_ok = ok.sum() / n_frames * 100
    print(f"  {n_frames} frames, {pct_ok:.1f}% above threshold")
    return out


def head_direction_summary(hd_df, fps=100.0):
    valid = hd_df.dropna(subset=["head_dir_rad"])
    angles = valid["head_dir_rad"].values

    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    mvl = np.sqrt(mean_cos**2 + mean_sin**2)
    mean_dir_rad = np.arctan2(mean_sin, mean_cos)

    unwrapped = pd.Series(np.unwrap(hd_df["head_dir_rad"].values))
    smoothed = unwrapped.rolling(5, center=True, min_periods=1).mean()
    ang_vel = np.degrees(smoothed.diff() * fps).abs()

    return {
        "n_frames": len(hd_df),
        "n_valid": len(valid),
        "frac_valid": len(valid) / len(hd_df),
        "mean_direction_deg": np.degrees(mean_dir_rad),
        "mean_vector_length": mvl,
        "mean_abs_angular_velocity_deg_s": ang_vel.mean(),
        "median_abs_angular_velocity_deg_s": ang_vel.median(),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch head-direction from DLC .h5 files.")
    parser.add_argument("--h5_dir", required=True, help="Directory with DLC .h5 files")
    parser.add_argument("--output_dir", required=True, help="Where to save CSV results")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=0.6, help="Likelihood threshold")
    parser.add_argument("--pattern", default="*_filtered.h5", help="Glob pattern for h5 files")
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob(args.pattern))
    if not h5_files:
        h5_files = sorted(h5_dir.glob("*DLC*.h5"))
    if not h5_files:
        print(f"ERROR: No .h5 files found in {h5_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} .h5 file(s) in {h5_dir}")

    summaries = []
    for h5_path in h5_files:
        stem = h5_path.stem
        print(f"\n--- {stem} ---")
        try:
            hd = compute_head_direction(
                str(h5_path),
                likelihood_threshold=args.threshold,
                fps=args.fps,
            )
            csv_out = output_dir / f"{stem}_head_direction.csv"
            hd.to_csv(csv_out, index=False)
            print(f"  Saved → {csv_out}")

            s = head_direction_summary(hd, fps=args.fps)
            s["video"] = stem
            summaries.append(s)
        except Exception as e:
            print(f"  [ERROR] {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries).set_index("video")
        summary_csv = output_dir / "summary_stats.csv"
        summary_df.to_csv(summary_csv)
        print(f"\n✓ Summary saved → {summary_csv}")
        print(summary_df.round(3).to_string())

    print(f"\n✓ Processed {len(summaries)}/{len(h5_files)} file(s).")


if __name__ == "__main__":
    main()

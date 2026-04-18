#!/usr/bin/env python3
"""
dlc_train.py — DLC 3 model training pipeline for HPC.

Handles everything after labeling: fixes project paths, creates the
training dataset with SuperAnimal weight initialization, configures
training hyper-parameters, trains, and evaluates.

Workflow on your workstation:
    1. create_new_project()
    2. edit config.yaml (bodyparts, skeleton, engine: pytorch)
    3. extract_frames() + label_frames() + check_labels()
    4. rsync project to cluster

Workflow on the cluster (this script):
    1. Fix project_path in config.yaml       (automatic)
    2. build_weight_init()                   (--setup)
    3. create_training_dataset()             (--setup)
    4. Edit pytorch_config.yaml overrides    (--epochs, --batch_size, etc.)
    5. train_network()
    6. evaluate_network()

Usage:
    # First run (creates training dataset on cluster):
    python dlc_train.py /path/to/config.yaml --setup \
        --superanimal superanimal_topviewmouse \
        --model_name hrnet_w32 \
        --shuffle 1 --epochs 200 --batch_size 8

    # Subsequent runs (dataset already created, just retrain):
    python dlc_train.py /path/to/config.yaml \
        --shuffle 1 --epochs 300

    # Memory replay fine-tuning:
    python dlc_train.py /path/to/config.yaml --setup \
        --superanimal superanimal_topviewmouse \
        --model_name hrnet_w32 \
        --shuffle 3 --epochs 50 --batch_size 64 \
        --detector_epochs 0 --save_epochs 10

To do:
[] Add condition top-down support

"""

import argparse
import sys
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fix_project_path(config_path):
    """
    Update project_path in config.yaml to match the actual location
    of the project on this machine.  Required after rsync-ing a project
    from a different machine.
    """
    project_dir = str(Path(config_path).parent)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    old_path = cfg.get("project_path", "")
    # Normalize for comparison
    old_norm = old_path.replace("\\", "/").rstrip("/")
    new_norm = project_dir.rstrip("/")

    if old_norm != new_norm:
        print("  Fixing project_path:")
        print(f"    Old: {old_path}")
        print(f"    New: {project_dir}")
        cfg["project_path"] = project_dir

        # Also ensure engine is set to pytorch
        if cfg.get("engine") != "pytorch":
            print(f"  Setting engine: pytorch (was: {cfg.get('engine', 'not set')})")
            cfg["engine"] = "pytorch"

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print("  config.yaml updated.\n")
    else:
        print("  project_path already correct.\n")

    return cfg, project_dir


def fix_video_paths(config_path):
    """
    Update video paths in config.yaml to point to videos that actually
    exist in the project's videos/ directory.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_dir = Path(cfg["project_path"])
    videos_dir = project_dir / "videos"
    video_sets = cfg.get("video_sets", {})

    if not video_sets:
        return cfg

    new_video_sets = {}
    changed = False

    for old_video_path, metadata in video_sets.items():
        old_p = Path(old_video_path.replace("\\", "/"))
        video_name = old_p.name

        # Look for the video in the project's videos/ directory
        new_video_path = videos_dir / video_name

        if new_video_path.exists():
            new_key = str(new_video_path)
            if new_key != old_video_path:
                print("  Fixing video path:")
                print(f"    Old: {old_video_path}")
                print(f"    New: {new_key}")
                changed = True
            new_video_sets[new_key] = metadata
        else:
            # Keep old path, user may need to fix manually
            print(f"  WARNING: Video not found at {new_video_path}")
            print(f"           Keeping original: {old_video_path}")
            new_video_sets[old_video_path] = metadata

    if changed:
        cfg["video_sets"] = new_video_sets
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print("  Video paths updated.\n")

    return cfg


def find_pytorch_config(config_path, shuffle=1, trainingsetindex=0):
    """Locate pytorch_config.yaml for a given shuffle."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project_path = Path(cfg["project_path"])
    iteration = cfg.get("iteration", 0)

    # Glob for the shuffle directory (handles varying task name formats)
    candidates = sorted(
        (project_path / "dlc-models-pytorch" / f"iteration-{iteration}").glob(
            f"*shuffle{shuffle}/train/pytorch_config.yaml"
        )
    )
    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        f"pytorch_config.yaml not found for shuffle {shuffle}.\n"
        "Has create_training_dataset() been run? (use --setup)"
    )


def apply_overrides(yaml_path, overrides):
    """Apply dot-notation overrides to a YAML file."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    changed = []
    for key, value in overrides.items():
        if value is None:
            continue
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        old = d.get(parts[-1])
        d[parts[-1]] = value
        changed.append(f"    {key}: {old} → {value}")

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    if changed:
        print(f"  Overrides applied to {Path(yaml_path).name}:")
        print("\n".join(changed))
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Full DLC 3 (PyTorch) training pipeline for HPC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run — set up training dataset on cluster, then train:
  python dlc_train.py /path/to/config.yaml --setup \\
      --superanimal superanimal_topviewmouse --model_name hrnet_w32 \\
      --shuffle 1 --epochs 200 --batch_size 8

  # Retrain (dataset already exists):
  python dlc_train.py /path/to/config.yaml --shuffle 1 --epochs 300
        """,
    )
    parser.add_argument(
        "config_path",
        help="Path to the DLC project config.yaml",
    )

    # ── Setup flags ────────────────────────────────────────────────────
    setup_group = parser.add_argument_group("setup (first run on cluster)")
    setup_group.add_argument(
        "--setup",
        action="store_true",
        default=False,
        help="Create_training_dataset before training",
    )
    setup_group.add_argument(
        "--resume_training",
        action="store_true",
        default=False,
        help="Skip all setup steps and resume with previously trained model. Need to provide snapshot_path and detector_path if true",
    )
    setup_group.add_argument(
        "--num_shuffles",
        type=int,
        default=1,
        help="Number of shuffles to create (default: 1)",
    )
    setup_group.add_argument(
        "--from_shuffle",
        type=int,
        default=None,
        help="(Optional) Shuffle index to copy when creating a new training dataset from an existing split (default: same as --shuffle)",
    )
    setup_group.add_argument(
        "--from_trainsetindex",
        type=int,
        default=0,
        help="(Optional) Trainset index to copy when creating a new training dataset from an existing split (default: 0)",
    )
    setup_group.add_argument(
        "--superanimal",
        type=str,
        default="",
        help="SuperAnimal model name (default: None). Options include 'superanimal_topviewmouse', 'superanimal_quadruped', 'superanimal_humanbody'",
    )
    setup_group.add_argument(
        "--model_name",
        type=str,
        default="hrnet_w32",
        help="Model architecture (default: hrnet_w32)",
    )
    setup_group.add_argument(
        "--net_type",
        type=str,
        default=None,
        help="Net type for create_training_dataset (default: same as --model_name)",
    )
    setup_group.add_argument(
        "--detector_name",
        type=str,
        default="fasterrcnn_mobilenet_v3_large_fpn",
        help="Detector model architecture for top-down models (default: fasterrcnn_mobilenetv3_large_fpn)",
    )
    setup_group.add_argument(
        "--with_decoder",
        action="store_true",
        default=False,
        help="Use transfer learning (with_decoder=False) or keypoint fine tuning (True) \
            when creating training dataset (default: False)",
    )
    setup_group.add_argument(
        "--memory_replay",
        action="store_true",
        default=False,
        help="Whether to prepare for memory replay fine-tuning (default: False). \
            If True, predicts all SuperAnimal bodyparts",
    )

    # ── Training flags ─────────────────────────────────────────────────
    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="Shuffle index to use for training (default: 1)",
    )
    train_group.add_argument(
        "--trainingsetindex",
        type=int,
        default=0,
        help="Training-set index (default: 0)",
    )
    train_group.add_argument(
        "--method",
        type=str,
        default=None,
        help="top-down (td) vs bottom-up (bu); default: same as pytorch_config.yaml",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train_settings.epochs",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override train_settings.batch_size",
    )
    train_group.add_argument(
        "--save_epochs",
        type=int,
        default=None,
        help="Override runner.snapshots.save_epochs",
    )
    train_group.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Override runner.eval_interval",
    )
    train_group.add_argument(
        "--display_iters",
        type=int,
        default=None,
        help="Override train_settings.display_iters",
    )
    train_group.add_argument(
        "--detector_epochs",
        type=int,
        default=None,
        help="Override detector train_settings.epochs (top-down models)",
    )
    train_group.add_argument(
        "--detector_batch_size",
        type=int,
        default=None,
        help="Override detector train_settings.batch_size (top-down models)",
    )
    train_group.add_argument(
        "--keep_deconv_weights",
        action="store_true",
        default=True,
        help="Keep deconvolutional weights when resuming training (default: True)",
    )
    train_group.add_argument(
        "--snapshot_path",
        type=str,
        default=None,
        help="Path to model snapshot to resume training from (required if --resume_training)",
    )
    train_group.add_argument(
        "--detector_path",
        type=str,
        default=None,
        help="Path to detector model to resume training from (required if --resume_training and using top-down model)",
    )
    train_group.add_argument(
        "--dataloader_workers",
        type=int,
        default=None,
        help="Override train_settings.dataloader_workers to speed up data loading (default: 0)",
    )
    train_group.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        default=False,
        help="Override train_settings.dataloader_pin_memory to speed up data loading (default: False)",
    )

    # ── Misc ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="Only run setup (--setup), skip training and evaluation",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        default=False,
        help="Skip evaluation after training",
    )

    args = parser.parse_args()

    # Look for incompatibility
    if args.resume_training:
        if args.setup:
            print("ERROR: --resume_training and --setup cannot both be true.")
            sys.exit(1)
        elif not args.snapshot_path or (args.method == "td" and not args.detector_path):
            print(
                "ERROR: --resume_training requires --snapshot_path (and --detector_path for top-down models)."
            )
            sys.exit(1)

    if args.net_type is None:
        args.net_type = args.model_name
        # DLC throws an error if you use superanimal for weight_init with top_down_model so this is a workaround
        if args.method == "td":
            print(
                "  Detected method=top-down. Using detector_name for training dataset creation."
            )
            args.net_type = "top_down_" + args.net_type

    if args.from_shuffle is None:
        args.from_shuffle = args.shuffle

    # ── Import DLC ──────────────────────────────────────────────────────
    import deeplabcut
    import torch

    print("=" * 65)
    print(f"  DeepLabCut {deeplabcut.__version__}")
    print(f"  PyTorch    {torch.__version__}")
    print(f"  CUDA       {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(
            f" — {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
        )
    else:
        print()
    print(f"  Config     {args.config_path}")
    print(f"  Shuffle    {args.shuffle}")
    print(f"  Setup      {args.setup}")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Fix project_path and video paths in config.yaml
    # ══════════════════════════════════════════════════════════════════
    print("\n[1/5] Fixing project paths...")
    _, project_dir = fix_project_path(args.config_path)
    fix_video_paths(args.config_path)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Create training dataset (only with --setup)
    # ══════════════════════════════════════════════════════════════════
    from deeplabcut.modelzoo import build_weight_init
    from deeplabcut.utils.pseudo_label import keypoint_matching
    from deeplabcut.modelzoo.utils import (
        create_conversion_table,
        read_conversion_table_from_csv,
    )
    import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions

    # cfg = yaml.safe_load(open(args.config_path))
    if args.setup:
        print("[2/5] Setting up training dataset...")
        print(f"  SuperAnimal : {args.superanimal}")
        print(f"  Model       : {args.model_name}")
        print(f"  Net type    : {args.net_type}")

        if (
            not args.superanimal
        ):  # No SuperAnimal — create the training dataset with default imagenet weights
            deeplabcut.create_training_dataset(
                config=args.config_path,
                num_shuffles=args.num_shuffles,
                Shuffles=[args.shuffle],
                net_type=args.net_type,
                detector_type=args.detector_name,
                userfeedback=False,
            )
        else:  # SuperAnimal training requires weight initialization before building the training dataset
            if args.with_decoder:  # Case of fine tuning rather than transfer learning
                print("  Using keypoint fine-tuning (with_decoder=True)")

                # Match the keypoints between the SuperAnimal model and the user's dataset, and save the pseudo labels for analysis/visualization
                keypoint_matching(
                    config_path=args.config_path,
                    superanimal_name=args.superanimal,
                    model_name=args.model_name,
                    detector_name=args.detector_name,
                    copy_images=True,
                )

                # Paths for outputs of keypoint_matching (conversion table, confusion matrix, pseudo predictions)
                conversion_table_path = (
                    Path(project_dir) / "memory_replay" / "conversion_table.csv"
                )

                # Create conversion table from keypoint matching results
                create_conversion_table(
                    config=args.config_path,
                    super_animal=args.superanimal,
                    project_to_super_animal=read_conversion_table_from_csv(
                        conversion_table_path
                    ),
                )
            else:
                print("  Using transfer learning (with_decoder=False)")

            weight_init = build_weight_init(
                cfg=auxiliaryfunctions.read_config(
                    args.config_path
                ),  # this is how they pass config in their notebooks, and it breaks with just string path
                super_animal=args.superanimal,
                model_name=args.model_name,
                detector_name=args.detector_name,
                with_decoder=args.with_decoder,
                memory_replay=args.memory_replay,
            )
            print(f"  Weight init built: {type(weight_init).__name__}")

            deeplabcut.create_training_dataset(
                config=args.config_path,
                num_shuffles=args.num_shuffles,
                Shuffles=[args.shuffle],
                net_type=args.net_type,
                detector_type=args.detector_name,
                augmenter_type="albumentations",
                weight_init=weight_init,
                userfeedback=False,
            )
            print("  Training dataset created.\n")

    else:  # Case of retraining existing dataset and you want to keep the same initial shuffle split
        print(
            "[2/5] Recreating training dataset from existing split...\n"
            f"  from_shuffle={args.from_shuffle}, from_trainsetindex={args.from_trainsetindex}"
        )

        if not args.superanimal:  # No SuperAnimal
            deeplabcut.create_training_dataset_from_existing_split(
                config=args.config_path,
                from_shuffle=args.from_shuffle,
                from_trainsetindex=args.from_trainsetindex,
                num_shuffles=args.num_shuffles,
                shuffles=[args.shuffle],
                net_type=args.net_type,
                detector_type=args.detector_name,
                augmenter_type="albumentations",
                userfeedback=False,
            )
        else:  # SuperAnimal training requires weight initialization before building the training dataset
            if args.with_decoder:  # Case of fine tuning rather than transfer learning
                print("  Using keypoint fine-tuning (with_decoder=True)")

                # Match the keypoints between the SuperAnimal model and the user's dataset, and save the pseudo labels for analysis/visualization
                keypoint_matching(
                    config_path=args.config_path,
                    superanimal_name=args.superanimal,
                    model_name=args.model_name,
                    detector_name=args.detector_name,
                    copy_images=True,
                )

                # Paths for outputs of keypoint_matching (conversion table, confusion matrix, pseudo predictions)
                conversion_table_path = (
                    Path(project_dir) / "memory_replay" / "conversion_table.csv"
                )

                # Create conversion table from keypoint matching results
                create_conversion_table(
                    config=args.config_path,
                    super_animal=args.superanimal,
                    project_to_super_animal=read_conversion_table_from_csv(
                        conversion_table_path
                    ),
                )
            else:
                print("  Using transfer learning (with_decoder=False)")

            weight_init = build_weight_init(
                cfg=auxiliaryfunctions.read_config(args.config_path),
                super_animal=args.superanimal,
                model_name=args.model_name,
                detector_name=args.detector_name,
                with_decoder=args.with_decoder,
                memory_replay=args.memory_replay,
            )
            print(f"  Weight init built: {type(weight_init).__name__}")

            deeplabcut.create_training_dataset_from_existing_split(
                config=args.config_path,
                from_shuffle=args.from_shuffle,
                from_trainsetindex=args.from_trainsetindex,
                num_shuffles=args.num_shuffles,
                shuffles=[args.shuffle],
                net_type=args.net_type,
                detector_type=args.detector_name,
                augmenter_type="albumentations",
                weight_init=weight_init,
                userfeedback=False,
            )

    if args.skip_train:
        print("--skip_train specified. Done.")
        return

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Apply pytorch_config.yaml overrides
    # ══════════════════════════════════════════════════════════════════
    print("[3/5] Configuring training hyper-parameters...")
    try:
        pt_cfg_path = find_pytorch_config(
            args.config_path,
            shuffle=args.shuffle,
            trainingsetindex=args.trainingsetindex,
        )
        print(f"  Pose config: {pt_cfg_path}")

        apply_overrides(
            pt_cfg_path,
            {
                "train_settings.epochs": args.epochs,
                "train_settings.batch_size": args.batch_size,
                "train_settings.display_iters": args.display_iters,
                "runner.snapshots.save_epochs": args.save_epochs,
                "detector.train_settings.epochs": args.detector_epochs,
                "detector.train_settings.batch_size": args.detector_batch_size,
                "runner.eval_interval": args.eval_interval,
                "detector.train_settings.dataloader_workers": args.dataloader_workers,
                "detector.train_settings.dataloader_pin_memory": args.dataloader_pin_memory,
                "train_settings.dataloader_workers": args.dataloader_workers,
                "train_settings.dataloader_pin_memory": args.dataloader_pin_memory,
            },
        )

    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Did you forget --setup?")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Train
    # ══════════════════════════════════════════════════════════════════
    if not args.resume_training:
        print("\n[4/5] Training network...")
    else:
        print("\n[4/5] Resuming training from snapshot...")
    deeplabcut.train_network(
        args.config_path,
        shuffle=args.shuffle,
        trainingsetindex=args.trainingsetindex,
        keepdeconvweights=args.keep_deconv_weights,
        snapshot_path=args.snapshot_path,
        detector_path=args.detector_path,
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate
    # ══════════════════════════════════════════════════════════════════
    if not args.skip_eval:
        print("\n[5/5] Evaluating network...")
        deeplabcut.evaluate_network(
            args.config_path,
            Shuffles=[args.shuffle],
            trainingsetindex=args.trainingsetindex,
            plotting=False,
        )
    else:
        print("\n[5/5] Skipping evaluation (--skip_eval).")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dlc_cache_weights.py — Pre-download model weights for offline HPC use.

DLC 3 downloads weights on first use from:
  1. timm (HuggingFace Hub)  — ImageNet-pretrained backbones (ResNet, HRNet, etc.)
  2. torchvision             — object detector weights (FasterRCNN, SSDLite)
  3. DLC ModelZoo (HF Hub)   — SuperAnimal pose + detector checkpoints

Run this on a login node (with internet), then training jobs on compute
nodes find everything in the local cache.

Usage:
    # Cache what you need for a specific setup:
    python dlc_cache_weights.py --model resnet_50 --detector fasterrcnn_resnet50_fpn_v2

    # Cache for SuperAnimal memory-replay fine-tuning:
    python dlc_cache_weights.py \
        --superanimal superanimal_topviewmouse \
        --sa_model resnet_50 \
        --detector fasterrcnn_mobilenet_v3_large_fpn

    # Cache everything:
    python dlc_cache_weights.py --all

    # Just list what's available:
    python dlc_cache_weights.py --list
"""

import argparse
import sys
import os


# ═══════════════════════════════════════════════════════════════════════
# DLC net_type → timm model name mapping
# ═══════════════════════════════════════════════════════════════════════

BACKBONE_TIMM_MODELS = {
    # ResNets
    "resnet_50": "resnet50_gn",
    "resnet_101": "resnet101",
    "top_down_resnet_50": "resnet50_gn",
    "top_down_resnet_101": "resnet101",
    # HRNets
    "hrnet_w18": "hrnet_w18",
    "hrnet_w32": "hrnet_w32",
    "hrnet_w48": "hrnet_w48",
    "top_down_hrnet_w18": "hrnet_w18",
    "top_down_hrnet_w32": "hrnet_w32",
    "top_down_hrnet_w48": "hrnet_w48",
    # DEKR (HRNet backbone)
    "dekr_w18": "hrnet_w18",
    "dekr_w32": "hrnet_w32",
    "dekr_w48": "hrnet_w48",
    # DLCRNet (custom, no timm download)
    "dlcrnet_stride16_ms5": None,
    "dlcrnet_stride32_ms5": None,
    # BUCTD/CTD (HRNet backbone)
    "ctd_coam_w32": "hrnet_w32",
    "ctd_coam_w48": "hrnet_w48",
    "ctd_coam_w48_human": "hrnet_w48",
    "ctd_prenet_hrnet_w32": "hrnet_w32",
    "ctd_prenet_hrnet_w48": "hrnet_w48",
    # RTMPose variants (CSPNeXt, no timm download)
    "ctd_prenet_rtmpose_s": None,
    "ctd_prenet_rtmpose_m": None,
    "ctd_prenet_rtmpose_x": None,
    "ctd_prenet_rtmpose_x_human": None,
    "rtmpose_s": None,
    "rtmpose_m": None,
    "rtmpose_x": None,
    # AnimalTokenPose (ModelZoo weights, not timm)
    "animal_tokenpose_base": None,
}

# ═══════════════════════════════════════════════════════════════════════
# Detectors: DLC name → torchvision constructor
# ═══════════════════════════════════════════════════════════════════════

DETECTOR_TORCHVISION_MODELS = {
    "ssdlite": "ssdlite320_mobilenet_v3_large",
    "fasterrcnn_mobilenet_v3_large_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_mobilenetv3_large_fpn": "fasterrcnn_mobilenet_v3_large_fpn",
    "fasterrcnn_resnet50_fpn_v2": "fasterrcnn_resnet50_fpn_v2",
}

# ═══════════════════════════════════════════════════════════════════════
# SuperAnimal ModelZoo: HuggingFace repo IDs and available files
#
# DLC downloads specific .pt files from these repos using hf_hub_download.
# The file naming pattern is: {superanimal_name}_{model_or_detector_name}.pt
# Plus legacy files: detector.pt, pose_model.pth
# ═══════════════════════════════════════════════════════════════════════

SUPERANIMAL_HF_REPOS = {
    "superanimal_topviewmouse": {
        "repo_id": "mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse",
        "pose_models": {
            "resnet_50": "superanimal_topviewmouse_resnet_50.pt",
            "hrnet_w32": "superanimal_topviewmouse_hrnet_w32.pt",
        },
        "detectors": {
            "fasterrcnn_mobilenet_v3_large_fpn": "superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt",
            "fasterrcnn_mobilenetv3_large_fpn": "superanimal_topviewmouse_fasterrcnn_mobilenet_v3_large_fpn.pt",
            "fasterrcnn_resnet50_fpn_v2": "superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt",
        },
        "legacy_files": ["detector.pt", "pose_model.pth"],
    },
    "superanimal_quadruped": {
        "repo_id": "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
        "pose_models": {
            "resnet_50": "superanimal_quadruped_resnet_50.pt",
            "hrnet_w32": "superanimal_quadruped_hrnet_w32.pt",
        },
        "detectors": {
            "fasterrcnn_mobilenet_v3_large_fpn": "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt",
            "fasterrcnn_mobilenetv3_large_fpn": "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn.pt",
            "fasterrcnn_resnet50_fpn_v2": "superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt",
        },
        "legacy_files": ["detector.pt", "pose_model.pth"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Download functions
# ═══════════════════════════════════════════════════════════════════════


def cache_timm_backbone(timm_name):
    """Download and cache a timm backbone's pretrained weights."""
    import timm

    print(f"  [timm] {timm_name}...", end=" ", flush=True)
    try:
        model = timm.create_model(timm_name, pretrained=True)
        del model
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def cache_torchvision_detector(tv_name):
    """Download and cache a torchvision detection model."""
    import torchvision.models.detection as det_models

    print(f"  [torchvision] {tv_name}...", end=" ", flush=True)
    try:
        constructors = {
            "ssdlite320_mobilenet_v3_large": det_models.ssdlite320_mobilenet_v3_large,
            "fasterrcnn_mobilenet_v3_large_fpn": det_models.fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_resnet50_fpn_v2": det_models.fasterrcnn_resnet50_fpn_v2,
            "fasterrcnn_resnet50_fpn": det_models.fasterrcnn_resnet50_fpn,
        }
        if tv_name not in constructors:
            print("UNKNOWN")
            return False
        model = constructors[tv_name](weights="DEFAULT")
        del model
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def cache_hf_file(repo_id, filename):
    """Download a single file from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    print(f"  [HF] {repo_id}/{filename}...", end=" ", flush=True)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"OK → {path}")
        return True
    except Exception as e:
        # Some files may not exist in the repo (e.g. newer checkpoint names)
        print(f"FAILED: {e}")
        return False


def cache_superanimal(
    sa_name, model_name=None, detector_name=None, download_legacy=False
):
    """
    Download SuperAnimal weights from HuggingFace.

    Parameters
    ----------
    sa_name : str
        e.g. "superanimal_topviewmouse"
    model_name : str or None
        Pose model to download, e.g. "resnet_50", "hrnet_w32".
        If None, downloads all available pose models.
    detector_name : str or None
        Detector to download, e.g. "fasterrcnn_mobilenet_v3_large_fpn".
        If None, downloads all available detectors.
    download_legacy : bool
        Also download the legacy detector.pt and pose_model.pth files.
    """
    if sa_name not in SUPERANIMAL_HF_REPOS:
        print(f"  WARNING: Unknown SuperAnimal '{sa_name}'")
        print(f"  Available: {list(SUPERANIMAL_HF_REPOS.keys())}")
        return 0, 1

    info = SUPERANIMAL_HF_REPOS[sa_name]
    repo_id = info["repo_id"]
    success, failed = 0, 0

    print(f"\n  --- {sa_name} (repo: {repo_id}) ---")

    # Pose models
    if model_name:
        models_to_get = {model_name: info["pose_models"].get(model_name)}
        if models_to_get[model_name] is None:
            print(
                f"  WARNING: No known pose file for model '{model_name}' in {sa_name}"
            )
            print(f"  Available: {list(info['pose_models'].keys())}")
            # Try the pattern anyway
            models_to_get[model_name] = f"{sa_name}_{model_name}.pt"
    else:
        models_to_get = info["pose_models"]

    for name, filename in models_to_get.items():
        if cache_hf_file(repo_id, filename):
            success += 1
        else:
            failed += 1

    # Detectors
    if detector_name:
        detectors_to_get = {detector_name: info["detectors"].get(detector_name)}
        if detectors_to_get[detector_name] is None:
            print(
                f"  WARNING: No known detector file for '{detector_name}' in {sa_name}"
            )
            print(f"  Available: {list(info['detectors'].keys())}")
            detectors_to_get[detector_name] = f"{sa_name}_{detector_name}.pt"
    else:
        detectors_to_get = info["detectors"]

    # Deduplicate (alternate spellings map to same file)
    seen_files = set()
    for name, filename in detectors_to_get.items():
        if filename in seen_files:
            continue
        seen_files.add(filename)
        if cache_hf_file(repo_id, filename):
            success += 1
        else:
            failed += 1

    # Legacy files
    if download_legacy:
        for filename in info.get("legacy_files", []):
            if cache_hf_file(repo_id, filename):
                success += 1
            else:
                failed += 1

    return success, failed


# ═══════════════════════════════════════════════════════════════════════
# List / help
# ═══════════════════════════════════════════════════════════════════════


def list_models():
    print("\n═══ DLC 3 Pose Backbones (--model) ═══")
    print(f"  {'DLC net_type':<35} {'timm model (downloaded)'}")
    print("  " + "─" * 60)
    for net_type, timm_name in sorted(BACKBONE_TIMM_MODELS.items()):
        dl = timm_name if timm_name else "(no download needed)"
        print(f"  {net_type:<35} {dl}")

    print("\n═══ Object Detectors (--detector) ═══")
    print(f"  {'DLC detector_type':<42} {'torchvision model'}")
    print("  " + "─" * 60)
    for det_type, tv_name in sorted(DETECTOR_TORCHVISION_MODELS.items()):
        print(f"  {det_type:<42} {tv_name}")

    print("\n═══ SuperAnimal ModelZoo (--superanimal + --sa_model) ═══")
    for sa_name, info in SUPERANIMAL_HF_REPOS.items():
        print(f"\n  {sa_name}")
        print(f"    Repo: {info['repo_id']}")
        print("    Pose models (--sa_model):")
        for m, f in info["pose_models"].items():
            print(f"      {m:<30} → {f}")
        print("    Detectors (--sa_detector):")
        seen = set()
        for d, f in info["detectors"].items():
            if f not in seen:
                print(f"      {d:<30} → {f}")
                seen.add(f)

    # Try live list from DLC
    try:
        from deeplabcut.pose_estimation_pytorch import (
            available_models,
            available_detectors,
        )

        print("\n═══ Live from DLC installation ═══")
        print(f"  Pose models: {available_models()}")
        print(f"  Detectors:   {available_detectors()}")
    except Exception:
        pass

    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download DLC model weights for offline HPC use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="*",
        default=None,
        help="DLC backbone net_type(s) to cache, e.g. resnet_50 hrnet_w32",
    )
    parser.add_argument(
        "--detector",
        nargs="*",
        default=None,
        help="Torchvision detector(s) to cache, e.g. fasterrcnn_resnet50_fpn_v2",
    )
    parser.add_argument(
        "--superanimal",
        nargs="*",
        default=None,
        help="SuperAnimal model(s) to cache, e.g. superanimal_topviewmouse",
    )
    parser.add_argument(
        "--sa_model",
        type=str,
        default=None,
        help="Specific SuperAnimal pose model to cache, e.g. resnet_50 or hrnet_w32 "
        "(default: download all available for each --superanimal)",
    )
    parser.add_argument(
        "--sa_detector",
        type=str,
        default=None,
        help="Specific SuperAnimal detector to cache, e.g. fasterrcnn_mobilenet_v3_large_fpn "
        "(default: download all available for each --superanimal)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Also download legacy detector.pt and pose_model.pth files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Cache ALL available models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if not any([args.model, args.detector, args.superanimal, args.all]):
        parser.print_help()
        print("\nERROR: Specify --model, --detector, --superanimal, --all, or --list")
        sys.exit(1)

    hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    torch_cache = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    print("=" * 65)
    print("  DLC Weight Cacher")
    print(f"  HF_HOME    : {hf_cache}")
    print(f"  TORCH_HOME : {torch_cache}")
    print("=" * 65)

    success, failed = 0, 0

    # ── Backbones ──────────────────────────────────────────────────────
    models_to_cache = []
    if args.all:
        models_to_cache = list(BACKBONE_TIMM_MODELS.keys())
    elif args.model:
        models_to_cache = args.model

    if models_to_cache:
        print("\n--- Caching ImageNet backbone weights (timm) ---")
        seen_timm = set()
        for net_type in models_to_cache:
            timm_name = BACKBONE_TIMM_MODELS.get(net_type)
            if timm_name is None:
                if net_type not in BACKBONE_TIMM_MODELS:
                    print(f"  WARNING: Unknown net_type '{net_type}'")
                    failed += 1
                else:
                    print(f"  {net_type}: no timm download needed (skip)")
                continue
            if timm_name in seen_timm:
                print(f"  {net_type} → {timm_name}: already cached (skip)")
                continue
            seen_timm.add(timm_name)
            if cache_timm_backbone(timm_name):
                success += 1
            else:
                failed += 1

    # ── Torchvision Detectors ──────────────────────────────────────────
    detectors_to_cache = []
    if args.all:
        detectors_to_cache = list(DETECTOR_TORCHVISION_MODELS.keys())
    elif args.detector:
        detectors_to_cache = args.detector

    if detectors_to_cache:
        print("\n--- Caching detector weights (torchvision) ---")
        seen_tv = set()
        for det_type in detectors_to_cache:
            tv_name = DETECTOR_TORCHVISION_MODELS.get(det_type)
            if tv_name is None:
                print(f"  WARNING: Unknown detector '{det_type}'")
                failed += 1
                continue
            if tv_name in seen_tv:
                continue
            seen_tv.add(tv_name)
            if cache_torchvision_detector(tv_name):
                success += 1
            else:
                failed += 1

    # ── SuperAnimal ModelZoo ───────────────────────────────────────────
    superanimals_to_cache = []
    if args.all:
        superanimals_to_cache = list(SUPERANIMAL_HF_REPOS.keys())
    elif args.superanimal:
        superanimals_to_cache = args.superanimal

    if superanimals_to_cache:
        print("\n--- Caching SuperAnimal ModelZoo weights (HuggingFace) ---")
        for sa_name in superanimals_to_cache:
            s, f = cache_superanimal(
                sa_name,
                model_name=args.sa_model,
                detector_name=args.sa_detector,
                download_legacy=args.legacy or args.all,
            )
            success += s
            failed += f

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Done. {success} cached, {failed} failed.")
    if failed:
        print("  ⚠ Some downloads failed — check network and retry.")
    else:
        print("  ✓ All weights cached successfully.")
    print("\n  Add these to your SLURM apptainer command:")
    print(f"    --env HF_HOME={hf_cache}")
    print(f"    --env TORCH_HOME={torch_cache}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

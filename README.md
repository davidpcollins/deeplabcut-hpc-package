# DeepLabCut 3 — HPC Deployment Guide

Run DLC training and video analysis on a SLURM cluster with GPU nodes
using Apptainer containers.

---

## Directory contents

```
hpc/
├── dlc3_nomodels.def            # Apptainer container definition - no cached models
├── dlc3_withmodels.def          # Apptainer container definition with all available models cached (larger size)
├── dlc_add_videos.py            # Python: add new videos to project on cluster
├── dlc_cache_weights.py         # Cache the models you want to use on login node (use with dlc3_nomodels.sif image)
├── dlc_train.py                 # Python: train + evaluate
├── dlc_analyze.py               # Python: batch video analysis
├── dlc_head_direction_batch.py  # Python: head-direction from .h5 files
├── slurm_train.sh               # SLURM: training (1 GPU)
├── slurm_analyze.sh             # SLURM: analyse all videos sequentially
├── slurm_analyze_array.sh       # SLURM: analyse videos in parallel (array job)
├── slurm_head_direction.sh      # SLURM: head-direction extraction (CPU-only)
└── README.md                    # this file
```

---

## 1. Build or upload the Apptainer container

Build once on a machine/node with internet access and (fakeroot or sudo):

```bash
# On a build node / login node with internet
module load apptainer          # if needed on your cluster
apptainer build dlc3.sif dlc3.def
```

The resulting `dlc3.sif` (~10 GB) is a read-only, portable image.
Copy it to your cluster's shared filesystem (e.g. `/scratch` or `/projects`).

For pre-built apptainer images: https://ucsf.box.com/s/cspfrac5r0a5beyk9o5fz3td7ohhlpqm

**Quick test:**

```bash
apptainer exec --nv dlc3.sif python -c \
    "import deeplabcut, torch; \
     print(f'DLC {deeplabcut.__version__}'); \
     print(f'CUDA: {torch.cuda.is_available()}')"
```

> If your cluster restricts internet access on compute nodes, build the
> container on a node that has access, or pull the NVIDIA base image
> ahead of time with `apptainer pull docker://nvcr.io/nvidia/pytorch:24.07-py3`.

---

## 2. Prepare your project (on your workstation)

These steps require a display and should be done locally:

1. Create the DLC project (`deeplabcut.create_new_project(...)`)
2. Edit `config.yaml` — set `engine: pytorch`, bodyparts, etc.
3. Extract & label frames (`extract_frames`, `label_frames`)

Then **copy the entire project directory** to the cluster:

```bash
rsync -avz /local/HeadDirection-YourName-2026-03-18/ \
    youruser@cluster:/scratch/youruser/dlc_projects/HeadDirection/
```

---

## 3. Edit paths in the SLURM scripts

Every `.sh` script has an `# ── Paths (EDIT THESE)` section. Update:

| Variable      | What it is                                       |
| ------------- | ------------------------------------------------ |
| `DLC_SIF`     | Full path to `dlc3.sif`                          |
| `CONFIG`      | Full path to your project's `config.yaml`        |
| `VIDEO_DIR`   | Directory containing your `.avi` / `.mp4` videos |
| `SCRIPTS_DIR` | Directory containing the Python runner scripts   |

Also update the `--bind` flag so Apptainer can see your data paths:

```bash
--bind /scratch/user/youruser:/scratch/user/youruser
```

And change `--partition=gpu` to match your cluster's GPU partition name.

---

## 4. Submit jobs

### Training

```bash
sbatch slurm_train.sh
```

This runs `dlc_train.py`, which builds the training dataset and over-
writes the `pytorch_config.yaml` file according to the arguments you provide,
calls `deeplabcut.train_network()` then `deeplabcut.evaluate_network()`.

### Video analysis — sequential

```bash
sbatch slurm_analyze.sh
```

Processes every video in `VIDEO_DIR` one-by-one on a single GPU.

### Video analysis — parallel (SLURM array)

```bash
# First, count your videos
ls /scratch/youruser/videos/*.avi | wc -l
# → e.g., 50

# Edit slurm_analyze_array.sh:  #SBATCH --array=0-49
# Optionally throttle:          #SBATCH --array=0-49%10  (max 10 concurrent)

sbatch slurm_analyze_array.sh
```

Each array task processes one video on its own GPU(s). This is the fastest
approach for large datasets.

### Head-direction extraction (CPU)

After video analysis completes:

```bash
sbatch slurm_head_direction.sh
```

This reads the `.h5` files and computes head-direction angles + summary
statistics. No GPU required.

---

## 5. Chain jobs with dependencies

You can chain the pipeline so each stage starts only after the previous
one succeeds:

```bash
# Step 1: train
TRAIN_JOB=$(sbatch --parsable slurm_train.sh)

# Step 2: analyse (starts after training completes)
ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm_analyze_array.sh)

# Step 3: head direction (starts after all array tasks complete)
sbatch --dependency=afterok:${ANALYZE_JOB} slurm_head_direction.sh
```

---

## 6. Iteration: refine → retrain

After inspecting results, go back to your **workstation** to:

1. Extract outlier frames: `deeplabcut.extract_outlier_frames(...)`
2. Refine labels: `deeplabcut.refine_labels(...)`
3. Merge datasets: `deeplabcut.merge_datasets(...)`

Then rsync the updated project back to the cluster and re-submit
`slurm_train.sh`.

---

## Tips

**Caching all DLC models:** I've included .def and .sif files for versions
of the apptainer image with all available models downloaded and cached or
without. The cached models add ~2-3 GB to the .sif file. If you upload or
build the .sif without models and your cluster's compute nodes do not
have an internet connection, you'll need to use `dlc_cache_weights.py`
to pre-load the model/detector backbones and superanimal weights before
training

**Adding new videos to your project:** Use the `dlc_add_videos.py` and
accompanying .sh file to do this on the cluster if you don't want to
download/re-upload everything

**Adjusting training length:** Edit `pytorch_config.yaml` on the cluster
before submitting the training job. Use the `dlc_helpers.py` functions
or edit the YAML directly:

```bash
# Quick edit from the command line
apptainer exec dlc3.sif python -c "
import yaml
p = '/scratch/.../train/pytorch_config.yaml'
cfg = yaml.safe_load(open(p))
cfg['train_settings']['epochs'] = 500
yaml.dump(cfg, open(p, 'w'), default_flow_style=False, sort_keys=False)
print('Updated epochs to 500')
"
```

**Monitoring:**

```bash
squeue -u $USER                     # check job status
tail -f logs/dlc-train_12345.out    # follow training output
sacct -j 12345 --format=Elapsed,MaxRSS,MaxVMSize,AllocGRES  # resource usage
```

**Storage:** DLC .h5 output files are stored alongside each video. For
a 10-minute video at 100 fps, the output is typically a few MB. Plan
for ~50 MB per video (analysis + filtered + CSV + labelled video).

**Container cache:** Apptainer caches layers in `~/.apptainer/cache`.
On clusters with small home quotas, set `APPTAINER_CACHEDIR` to a
scratch location before building:

```bash
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache
```

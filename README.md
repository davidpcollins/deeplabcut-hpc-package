# DeepLabCut 3 — HPC Guide

Run DLC training and video inference on a HPC cluster with GPU nodes using Slurm
and Apptainer images.

---

## Directory contents

```
dlc3_nomodels.def            # Apptainer container definition - no cached models (need internet access to run)
dlc3_withmodels.def          # Apptainer container definition with all available models cached (larger size)
dlc_add_videos.py            # Python: add new videos to project on cluster
dlc_cache_weights.py         # Cache the models you want to use on login node (use with dlc3_nomodels.sif image)
dlc_train.py                 # Python: train + evaluate network
dlc_analyze.py               # Python: batch video analysis
dlc_head_direction_batch.py  # Python: head-direction from .h5 files
slurm_train.sh               # SLURM: training (1 GPU)
slurm_analyze.sh             # SLURM: analyze all videos sequentially
slurm_analyze_array.sh       # SLURM: analyze videos in parallel (array job)
slurm_multiGPU_array.sh      # SLURM: analyze multiple videos per node using separate GPUs (alternative to array job)
README.md
```

---

## Step 1. Build or upload the Apptainer container

Build once on a node with internet access and (fakeroot or sudo):

```bash
# On a login node with internet
module load apptainer
apptainer build dlc3.sif dlc3_withmodels.def
```

The resulting `dlc3.sif` (~10 GB) is a read-only, portable image.
Copy it to your cluster's shared filesystem (e.g. `/scratch`).

For pre-built apptainer images: https://ucsf.box.com/s/cspfrac5r0a5beyk9o5fz3td7ohhlpqm

**Quick test after building:**

```bash
apptainer exec --nv dlc3.sif python -c \
    "import deeplabcut, torch; \
     print(f'DLC {deeplabcut.__version__}'); \
     print(f'CUDA: {torch.cuda.is_available()}')"
```
Should print the DLC version (e.g. 3.0.0rc13) and True if you have access to GPU

---

## Step 2. Prepare your project (on your workstation)

These steps require a display and should be done locally:

1. Create the DLC project (GUI or `deeplabcut.create_new_project()`)
2. Edit `config.yaml` — set `batch_size`, select bodyparts, etc.
3. Extract & label frames (`extract_frames`, `label_frames`)
4. **copy the entire project directory** to the cluster:
```bash
rsync -avz /local/Task-YourName-2026-03-18/ \
    username@cluster:/scratch/youruser/dlc_projects/Task/
```

---

## 3. Edit paths in the SLURM scripts

Every `.sh` script has a `# ── Paths (EDIT THESE)` section. Update:

| Variable      | What it is                                   |
| ------------- | -------------------------------------------- |
| `DLC_SIF`     | Full path to `dlc3.sif`                      |
| `CONFIG`      | Full path to your project's `config.yaml`    |
| `VIDEO_DIR`   | Directory containing videos to analyze       |
| `SCRIPTS_DIR` | Directory containing this package of scripts |

Also update the `--bind` flag so Apptainer can see your data paths:

```bash
--bind /scratch/user/name:/scratch/user/name  # project drive
--bind /home/user/name:/home/user/name  # code drive
```

And change `--partition=gpu` to match your cluster's appropriate partition name.

---

## Step 4. Submit jobs

### Training

```bash
sbatch slurm_train.sh
```

This runs `dlc_train.py`, which builds the training dataset from pre-labeled images, 
over-writes the `pytorch_config.yaml` file according to the arguments you provide,
calls `deeplabcut.train_network()` then `deeplabcut.evaluate_network()`.

### Video analysis — sequential

```bash
sbatch slurm_analyze.sh
```

Processes every video in `VIDEO_DIR` one-by-one on the reserved GPU.

### Video analysis — parallel (SLURM array)

```bash
# First, count your videos
ls /scratch/youruser/videos/*.mp4 | wc -l
# → e.g., 50

# Edit slurm_analyze_array.sh:  #SBATCH --array=0-49
# Optional throttle:            #SBATCH --array=0-49%10  (max 10 concurrent)

sbatch slurm_analyze_array.sh
```
Each array task processes one video on its own GPU(s). This is the fastest
approach for large datasets.

### Video analysis - parallel with array + multiple GPUs per job:

```bash
sbatch slurm_multigpu_array.sh
```
Works like `slurm_analyze_array.sh` but with the ability to reserve two GPUs per
node and run independent jobs. Helpful if the HPC has restrictions on the number
of simultaneous nodes that can be reserved.

All of these video analysis scripts call `dlc_analyze.py`, which runs video 
inference using the specified model and optionally filters predictions, creates 
labeled videos, and extracts outlier frames

---

## Step 5. Chain jobs with dependencies

You can chain the pipeline so each stage starts only after the previous
one succeeds:

```bash
# Step 1: train
TRAIN_JOB=$(sbatch --parsable slurm_train.sh)

# Step 2: analyse (starts after training completes)
ANALYZE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} slurm_analyze_array.sh)
```

---

## Step 6. Refining and retraining

After inspecting results, go back to your **local workstation** to:

1. Extract outlier frames (if needed): `deeplabcut.extract_outlier_frames()`
2. Refine labels: `deeplabcut.refine_labels()`
3. Merge datasets: `deeplabcut.merge_datasets()`

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

**Adjusting batch size:** Commercial GPUs used on HPC nodes can handle
higher batch sizes than consumer models. Anecdotally, `batch_size=128`
works well and will not give memory issues with Nvidia L40S, while H100
and H200 GPUs can usually handle `batch_size=256` or higher

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

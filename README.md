# Soft-Freezing StyleGAN-XL

> During StyleGAN-XL super-resolution training, **does softly unfreezing part of the stem beat the paper's head-only freezing?** And if it helps, is the gain because *the head simply trains more*, or because the *stem–head interface re-aligns (co-adaptation)*? This project investigates both.

A fork of [StyleGAN-XL](https://github.com/autonomousvision/stylegan_xl) (Sauer et al., SIGGRAPH'22). We implement five **freeze policies** for the super-res stage and go beyond plain FID/KID by diagnosing *where* any improvement comes from, using **layer drift / counterfactual feature injection / LPIPS / feature-map** analyses.

For the original StyleGAN-XL usage (sample generation, inversion, StyleMC, etc.), see the [upstream repository](https://github.com/autonomousvision/stylegan_xl).

---

## 1. Research question & motivation

StyleGAN-XL first trains a low-res **stem**, then stacks a super-resolution **head** on top and grows the resolution progressively. The paper's super-res recipe is **head-only**: mapping + stem synthesis are fully frozen, and only the new head is trained.

We question that assumption:

- **(Q1)** If we also train a few of the top stem layers (i.e. *soft-freeze*), do we beat the head-only baseline?
- **(Q2)** If so, is the gain merely from "training the stem more," or from the **stem–head boundary co-adapting** to each other?
- **(Q3)** How does the effect differ when the stem is **underfit** (low-resource, from-scratch) vs **converged** (the paper's ~87M-kimg pkl)?

Writing `G(z,c) = H(S(z,c))` (S = stem, H = head): the baseline trains `H_b(S_0)` while a variant trains `H_v(S_v)`. A plain baseline↔variant comparison changes both the stem (`S_0→S_v`) and the head (`H_b→H_v`) at once, so the cause is confounded. We therefore use **counterfactual injection** `H_v(S_0)` — keep the variant head fixed and swap the stem feature back to the original — to isolate the contribution of the stem change.

### Freeze policies (variants)

Defined by flags we added to `train.py` (on top of the paper's defaults):

| Variant | Definition | `train.py` flags |
|---|---|---|
| **Baseline** | head-only; mapping + stem synthesis fully frozen | (none, default) |
| **A** (a / a2 / a3) | additionally unfreeze only the last **N** stem synthesis layers (N=1/2/3) | `--unfreeze_last_stem_layers N` |
| **B** | no freezing; train the whole stem at a **reduced LR** | `--soft_freeze --stem_lr_mult 0.1` |
| **C** | B + a **stem-preservation loss** (regularize toward the original stem) | `--soft_freeze --stem_lr_mult 0.1 --preserve_weight λ --preserve_metric mse` |
| **D** | B + **decay the stem LR to 0** over training (linear/cosine) | `--soft_freeze --stem_lr_mult 0.1 --stem_lr_schedule {linear,cosine} [--stem_lr_decay_kimg K]` |

---

## 2. Environment setup

### Requirements
- 64-bit Python 3.8+ / PyTorch (1.9 or 2.x)
- CUDA toolkit 11.1+ (matching your PyTorch build), GCC 7+
- An NVIDIA GPU (everything is written for a single GPU; runners default to `--gpus 1`)

### Conda environment
Pick the env file that matches your machine:

```bash
# Default (PyTorch 1.9.1 / CUDA 11.1) — e.g. A100
conda env create -f environment.yml

# Ada / RTX 4090 (PyTorch 2.0.1 / CUDA 11.8)
conda env create -f environment_4090.yml

conda activate sgxl
```

### CUDA custom ops (JIT build)
The StyleGAN3-family custom CUDA kernels (`bias_act`, `upfirdn2d`, `filtered_lrelu`) are **compiled automatically by `ninja` on first run**. So:
- the conda env must provide `nvcc` (cudatoolkit) and `ninja`, and the system GCC must be compatible with CUDA;
- the first `train.py` / `calc_metrics.py` call is a few minutes slower due to compilation; build artifacts are cached and fast afterwards;
- on build failure, see the StyleGAN3 [troubleshooting docs](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md) and [stylegan_xl#23](https://github.com/autonomousvision/stylegan_xl/issues/23).

### Datasets & pretrained stems
Runners assume this layout:

```
data/          imagenet100_{16,32,64,128,256}.zip, pokemon128.zip, cifar10_{16,32}.zip
pretrained/    imagenet{16,32,64,128}.pkl   (paper's official ImageNet-1k 1000-class stems)
training-runs/ all training output (created automatically)
```

Building dataset zips:
```bash
# General: any image folder -> per-resolution zip (upstream tool)
python dataset_tool.py --source=./data/pokemon --dest=./data/pokemon128.zip \
  --resolution=128x128 --transform=center-crop

# ImageNet-100 256^2 reference zip (for FID), from the val set
python build_imagenet100_256_zip.py

# Helper converters: parquet -> zip / HuggingFace CIFAR-10 -> images
python parquet_to_stylegan_zip.py
python export_cifar10_hf.py
```

---

## 3. Training

A run is one super-res stage trained with one freeze policy, launched directly with `python train.py`. The experiment matrix is **stem state × dataset × resolution (stage) × variant** — build any cell by combining the common flags + the stage values + the variant flags below. Use `tmux` to detach long runs.

### Command template

```bash
python train.py \
  --outdir=training-runs/<run-name> \
  --cfg=stylegan3-t --gpus=1 --batch=32 --mirror=1 --snap=20 --kimg=100 \
  --cond=1 --cls_weight=0.1 --label_dim=1000 --metrics=none \
  --superres --up_factor=2 \
  --data=<DATA> --path_stem=<STEM> --batch-gpu=<BG> --head_layers=<HL> \
  <VARIANT-FLAGS>
```

The flags on the first three lines are common to every super-res run. `--superres --up_factor=2` means each stage doubles the resolution from its stem. (`--metrics=none` skips in-loop FID so training is faster; measure separately — see below. We exclude precision/recall from reporting by convention.)

### Stage values — `<DATA> <STEM> <BG> <HL>`

ImageNet-100 on the **converged** (paper) stems:

| Stage | `--path_stem` (STEM) | `--data` (DATA) | `--batch-gpu` (BG) | `--head_layers` (HL) |
|---|---|---|---|---|
| 16→32  | pretrained/imagenet16.pkl  | data/imagenet100_32.zip  | 8 | 4 |
| 32→64  | pretrained/imagenet32.pkl  | data/imagenet100_64.zip  | 8 | 4 |
| 64→128 | pretrained/imagenet64.pkl  | data/imagenet100_128.zip | 4 | 7 |
| 128→256| pretrained/imagenet128.pkl | data/imagenet100_256.zip | 2 | 7 |

For other datasets, swap `--data` (e.g. `data/pokemon128.zip`, `data/cifar10_32.zip`) and point `--path_stem` at the stem you trained. For the **underfit** (from-scratch) regime you first train a 16² stem and then each super-res stage on the *previous stage's* `network-snapshot.pkl` instead of a `pretrained/*.pkl` — see "From-scratch cascade" below.

### Variant flags — `<VARIANT-FLAGS>`

| Variant | Flags to append |
|---|---|
| **baseline** | *(none — head-only)* |
| **a / a2 / a3** | `--unfreeze_last_stem_layers=1` / `=2` / `=3` |
| **b** | `--soft_freeze --stem_lr_mult=0.1` |
| **c** | `--soft_freeze --stem_lr_mult=0.1 --preserve_weight=1.0 --preserve_metric=mse` |
| **d** | `--soft_freeze --stem_lr_mult=0.1 --stem_lr_schedule=cosine` (or `linear`) `[--stem_lr_decay_kimg=K]` |

### Worked example — ImageNet-100, 32→64, Variant A(N=2)

```bash
python train.py \
  --outdir=training-runs/imagenet100_64_conv_a2 \
  --cfg=stylegan3-t --gpus=1 --batch=32 --mirror=1 --snap=20 --kimg=100 \
  --cond=1 --cls_weight=0.1 --label_dim=1000 --metrics=none \
  --superres --up_factor=2 \
  --data=data/imagenet100_64.zip --path_stem=pretrained/imagenet32.pkl \
  --batch-gpu=8 --head_layers=4 \
  --unfreeze_last_stem_layers=2

# detached, with a log
tmux new-session -d -s in100_64_a2 \
  "cd $(pwd) && python train.py --outdir=training-runs/imagenet100_64_conv_a2 ... 2>&1 | tee training-runs/in100_64_a2.log"
```

### From-scratch cascade (underfit stem)

The underfit regime has no pretrained stem; you train the stem yourself, then chain stages. Stem (no `--superres`):

```bash
python train.py --outdir=training-runs/in100_scratch_stem \
  --cfg=stylegan3-t --gpus=1 --batch=32 --batch-gpu=8 --mirror=1 --snap=20 --kimg=100 \
  --cond=1 --label_dim=1000 --metrics=none \
  --data=data/imagenet100_16.zip
```

Then each super-res stage uses the previous stage's snapshot as `--path_stem`:
```bash
--path_stem=training-runs/in100_scratch_stem/00000-stylegan3-t-imagenet100_16-gpus1-batch32/network-snapshot.pkl
```

### Measuring metrics

```bash
python calc_metrics.py --metrics=fid50k_full,kid50k_full \
  --network=training-runs/imagenet100_64_conv_a2/00000-stylegan3-t-imagenet100_64-gpus1-batch32/network-snapshot.pkl \
  --data=data/imagenet100_64.zip --gpus=1 --mirror=1
```

---

## 4. Analysis

Diagnostics run on the trained snapshot pkls. **Most take no arguments and run from the repo root**; the run paths to analyze are hardcoded near the top of each script (`RUNS` / `RUN_MAP` / `RUN_*`), so check/edit them to match your run names once.

### (a) Layer drift spectrum — *where and how much the stem changed*
Compares the trained `G_ema.synthesis` weights against the starting stem pkl, computing per-layer relative Frobenius drift + a ΔW spectrum.
```bash
python analyze_drift_spectrum.py        # -> analysis_results.json (+ markdown summary)
python analyze_sr256.py                 # appends the sr256 (128->256) block to analysis_results.json
python analyze_pokemon_drift.py         # -> pokemon_drift_results.json
python compare_underfit_vs_converged.py # compares the two jsons: same variant, underfit vs converged
python summarize_analysis.py            # print tables
python summarize_pokemon.py
python two_tables.py
```
Example writeup: [drift_underfit_vs_converged.md](drift_underfit_vs_converged.md) (e.g. converged stems drift ~4–5× less on average than underfit ones).

### (b) Feature-map visualization — *what changed at each layer*
For a fixed (z, class), capture each synthesis layer's output (channel mean) via forward hooks and lay baseline vs variant side by side.
```bash
python feature_map_sr128.py          # baseline vs a3 -> feature_maps_sr128.png
python feature_map_all_stages.py     # every super-res stage -> feature_maps_<stage>.png
python feature_map_sr128_samples.py  # 10 samples x 6 variants -> feature_maps_sr128_*_sNN.png
```

### (c) Counterfactual feature injection — *was the stem change actually needed*
Keep the variant head, but replace only the stem feature with the baseline's → build `H_v(S_0)` and compare against `H_v(S_v)` / baseline. A large quality drop is evidence the gain came from **stem–head co-adaptation**.
```bash
# FID/KID (counterfactual G built by swapping in the baseline stem)
python counterfactual_metrics_sr128.py    # sr128 variants a, c
python counterfactual_metrics_all.py      # all available baseline+variant pairs
python counterfactual_metrics_sr256.py    # sr256
python counterfactual_metrics_pokemon.py  # pokemon128 Variant B
#   -> counterfactual_runs/<label>/metric-*.jsonl

# Qualitative images (normal vs counterfactual at the same latent)
python counterfactual_sr128.py            # -> counterfactual_sr128_*_sNN.png
```

| Output | Contents |
|---|---|
| `analysis_results.json`, `pokemon_drift_results.json` | raw layer drift / ΔW spectrum data |
| `feature_maps_*.png`, `counterfactual_sr128_*.png` | feature-map / qualitative comparison images |
| `counterfactual_runs/*/metric-*.jsonl` | counterfactual FID/KID |

---

## 5. Repository layout

```
train.py                       super-res freeze-variant flags added (A/B/C/D)
analyze_*.py / compare_*.py    layer drift analysis
counterfactual_*.py            counterfactual feature injection (stem swap / hook)
lpips_*.py / feature_map_*.py  LPIPS / feature-map diagnostics
summarize_*.py / two_tables.py result-table generation
build_imagenet100_256_zip.py   data preparation
data/ pretrained/ training-runs/   input zips / stem pkls / output
dnnlib/ torch_utils/ pg_modules/ metrics/ training/   upstream StyleGAN-XL core (mostly unchanged)
```

---

## 6. Credits / License / Citation

This repository forks [StyleGAN-XL](https://github.com/autonomousvision/stylegan_xl) and extends it for a study of super-res freeze policies. The core model/training code belongs to the original authors and builds on the [StyleGAN3](https://github.com/NVlabs/stylegan3) and [Projected GANs](https://github.com/autonomousvision/projected_gan) codebases.

Licensed under the original [LICENSE.txt](LICENSE.txt) (NVIDIA Source Code License).

Cite the original paper:
```bibtex
@InProceedings{Sauer2021ARXIV,
  author    = {Axel Sauer and Katja Schwarz and Andreas Geiger},
  title     = {StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets},
  journal   = {arXiv.org},
  volume    = {abs/2201.00273},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.00273},
}
```

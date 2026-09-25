# SPIRA

**SPIRA** (**Source-Prefix Influence-aware Residual Autoregression**) is a source-conditioned autoregressive framework for **3D medical image translation**.

Our core idea is to view autoregressive translation as a **source–prefix balance** problem. The source provides a fixed, spatially aligned anatomical anchor, while the generated target prefix provides evolving target-side context. Instead of always using the full prefix correction, SPIRA learns how much of that correction should be retained for each target region.

<p align="center">
  <img src="assets/teaser" alt="SPIRA teaser" width="800">
</p>

SPIRA follows three simple ideas:

- 🧭 **Separate source and prefix influence**, by comparing source-only and full-prefix predictions
- ↩️ **Retract harmful or unnecessary corrections**, using a target-aware maximal zero-regret oracle during training
- 🎯 **Distill the oracle decision**, so a lightweight gate can adaptively balance source and prefix information at inference

By selectively retaining target-useful prefix information, SPIRA improves autoregressive 3D medical image translation across multiple modalities and datasets.

## 🔎 Overview

In source-conditioned autoregressive translation, the source volume already provides rich patient-specific anatomy, while the target prefix contributes additional target-side information. However, the usefulness of this prefix correction varies across regions, and a large prediction change does not necessarily mean that the correction is useful.

SPIRA explicitly models this difference and performs **token-wise selective prefix retraction**.

<p align="center">
  <img src="assets/framework" alt="SPIRA framework" width="800">
</p>

## 💡 Key Ideas

<table>
  <thead>
    <tr>
      <th align="left" width="34%">Component</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🧩 <b>Source–Prefix Decomposition</b></td>
      <td>Uses the same autoregressive predictor with and without the visible target prefix. Their difference defines the current prefix correction beyond the source-only prediction.</td>
    </tr>
    <tr>
      <td>↩️ <b>Maximal Zero-Regret Retraction</b></td>
      <td>During training, a ground-truth oracle finds the largest portion of the prefix correction that can be removed without increasing the current target loss.</td>
    </tr>
    <tr>
      <td>🎯 <b>Correction-Aware Gate</b></td>
      <td>A lightweight token-wise gate distills the oracle predictive distribution and predicts the retraction amount without access to the target at inference.</td>
    </tr>
  </tbody>
</table>

Together, these components let SPIRA adapt the source–prefix balance to local prediction needs instead of applying one fixed prefix strength everywhere.

## ✨ Features

<table>
  <thead>
    <tr>
      <th align="left" width="34%">Feature</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🧠 <b>3D blockwise autoregression</b></td>
      <td>Generates volumetric latent blocks sequentially with a source-conditioned autoregressive Transformer.</td>
    </tr>
    <tr>
      <td>📍 <b>Token-wise control</b></td>
      <td>Predicts local retraction amounts instead of sharing one global source–prefix balance across the volume.</td>
    </tr>
    <tr>
      <td>⚖️ <b>Utility-aware retraction</b></td>
      <td>Distinguishes how much a prefix correction changes the prediction from whether that change helps the target.</td>
    </tr>
    <tr>
      <td>🚫 <b>GT-free inference</b></td>
      <td>The target-aware oracle is used only during training; inference relies only on source and generated-prefix features.</td>
    </tr>
    <tr>
      <td>🧊 <b>VidTok-FSQ latent space</b></td>
      <td>Uses a frozen non-causal VidTok-FSQ tokenizer with cached discrete 3D latent representations.</td>
    </tr>
    <tr>
      <td>🪟 <b>Whole-volume generation</b></td>
      <td>Supports full-volume or sliding-window autoregressive inference with KV caching.</td>
    </tr>
  </tbody>
</table>

## 📊 Experimental Scope

SPIRA is evaluated on **8 translation tasks across 3 datasets**.

**BraTS 2024**
- T1n → T1c
- T1c → T1n
- T2w → T2f
- T2f → T1c

**SynthRAD 2025**
- MRI → CT
- CBCT → CT

**AutoPET**
- CT → PET
- PET → CT

The experiments evaluate translation quality using metrics including **PSNR, SSIM, LPIPS, and FID**, together with ROI reconstruction, statistical analysis, and downstream segmentation evaluation.

## 🗂️ Data Preparation

Download the corresponding public datasets and prepare separate training and validation JSON datalists. Each entry describes one modality of one case, for example:

```json
[
  {
    "case_id": "case_001",
    "cohort": "cohort_a",
    "modality": "t1n",
    "image": "case_001/t1n.nii.gz"
  },
  {
    "case_id": "case_001",
    "cohort": "cohort_a",
    "modality": "t1c",
    "image": "case_001/t1c.nii.gz"
  }
]
```

All modalities from the same case should be spatially registered to the same grid. Update the corresponding YAML files in `configs/` with the dataset paths, modality pair, and tokenizer checkpoint.

## 🛠️ Usage

### Installation

Use Python 3.11 and install a compatible PyTorch build, then run:

```bash
python -m pip install -r requirements.txt
```

Latent generation and image decoding additionally require the public [VidTok](https://github.com/microsoft/VidTok) implementation. Place the VidTok source and checkpoint according to the paths specified in the latent configuration.

### 1. Generate latent caches

For BraTS 2024:

```bash
python spira_ar/tools/precompute_vidtok_latents_overlap.py \
  --dataset brats24 \
  --config configs/latent_brats24.yaml \
  --split train
```

```bash
python spira_ar/tools/precompute_vidtok_latents_overlap.py \
  --dataset brats24 \
  --config configs/latent_brats24.yaml \
  --split val
```

For SynthRAD or AutoPET, replace the dataset name and latent configuration accordingly.

### 2. Train SPIRA

```bash
python spira_ar/train_spira.py --config configs/brats24.yaml
```

### 3. Full-volume inference

```bash
python spira_ar/infer_spira.py \
  --config configs/brats24.yaml \
  --ckpt outputs/brats24/t1n_to_t1c/last.pt \
  --use-ema \
  --output-dir outputs/brats24/predictions
```

# Denoising and Similarity-Guided Fusion Network (DSFN)

PyTorch implementation of the Denoising and Similarity-Guided Fusion Network
(DSFN) for fault diagnosis from multivariate UAV sensor time series. The model operates on
9-channel, 1,024-step windows from RflyMAD and predicts six classes: Normal,
Accel, GPS, Gyro, Mag, and Baro.

This repository contains the research code used to train the model, prepare the
window-level split reported in the manuscript, and benchmark dynamic inference. The
RflyMAD dataset and trained checkpoints are not redistributed.

## Method overview

The nine input channels are divided into three branches (Gyro, Accel, and
Velocity). Each branch uses a temporal CNN and Transformer layers to model
local patterns and long-range dependencies. The complete model contains a
gated feature-level denoising (GTD) module, dynamic-depth selection, and a
cross-modal similarity-guided reliability-consistency fusion (CRCF) module.

## Repository contents

- `train_dsfn.py`: model definition, training, evaluation, ablations, and plots.
- `prepare_rflymad.py`: extraction, windowing, and stratified 80/10/10 splitting.
- `benchmark_dynamic_efficiency.py`: parameters, FLOPs, average depth, skip rate,
  latency, and optional robustness measurements.
- `data/README.md`: dataset download and preparation notes.

## Installation

Python 3.9 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare RflyMAD

Download `Real-Sensors.zip` and `Real-No_Fault.zip` from the
[official RflyMAD page](https://rfly-openha.github.io/documents/4_resources/dataset.html),
then run:

```bash
python prepare_rflymad.py \
  --sensor-zip /path/to/Real-Sensors.zip \
  --normal-zip /path/to/Real-No_Fault.zip \
  --output-dir data/processed \
  --window-size 1024 \
  --stride 512 \
  --split-mode window \
  --split-seed 42
```

This command reproduces the manuscript protocol: 1,024-step windows with a
stride of 512, followed by a stratified window-level 80%/10%/10% split. The
optional `--split-mode flight` setting is provided for separate grouped-split
studies, but it is not the protocol used for the reported manuscript results.

## Train

Training options are supplied through `SGLA_*` environment variables. Example
for Linux/macOS:

```bash
SGLA_DATA_ROOT=data/processed \
SGLA_SPLIT_DIR=data/processed/fixed_split_80_10_10_seed42 \
SGLA_SAVE_ROOT=runs \
SGLA_EXP_NAME=Denoise_RCDFusion \
SGLA_SEED=42 \
SGLA_USE_POSITIONAL_ENCODING=0 \
python train_dsfn.py
```

Windows PowerShell:

```powershell
$env:SGLA_DATA_ROOT = "data/processed"
$env:SGLA_SPLIT_DIR = "data/processed/fixed_split_80_10_10_seed42"
$env:SGLA_SAVE_ROOT = "runs"
$env:SGLA_EXP_NAME = "Denoise_RCDFusion"
$env:SGLA_SEED = "42"
$env:SGLA_USE_POSITIONAL_ENCODING = "0"
python train_dsfn.py
```

Repeat the experiment with seeds `42`, `2025`, and `3407` and report the mean
and standard deviation. Checkpoints, tables, and figures are written under the
configured run directory.

## Dynamic-inference benchmark

```bash
python benchmark_dynamic_efficiency.py \
  --checkpoint runs/Denoise_RCDFusion_seed42/best_model.pth \
  --data-root data/processed \
  --split-dir data/processed/fixed_split_80_10_10_seed42 \
  --output-dir results/efficiency \
  --device cuda
```

Use the same hardware, batch size, warm-up iterations, and timing iterations
when comparing models. GPU and CPU latency should be reported separately.

## Reproducibility notes

- Input shape: `[batch, 9, 1024]`.
- Classes: Normal, Accel, GPS, Gyro, Mag, and Baro.
- Reported split: stratified window-level 80%/10%/10% (7,577/947/946 samples).
- Window length and stride: 1,024 and 512 samples, respectively.
- Recommended random seeds: 42, 2025, and 3407.
- Positional encoding is disabled in the reported manuscript experiments.

## Citation

Please cite the associated paper when it becomes available. The citation entry
will be added after publication.

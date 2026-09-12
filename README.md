# DSFN: A Robust UAV Sensor Fault Diagnosis Network

Implementation and reproduction materials for the manuscript **“A Robust UAV
Sensor Fault Diagnosis Network Based on Feature-Level Denoising and
Reliability-Consistency Fusion.”**

DSFN diagnoses six UAV sensor states (Normal, Accel, GPS, Gyro, Mag and Baro)
from nine-channel time-series inputs. The network combines modality-specific
temporal CNN/Transformer encoders, gated feature-level denoising, dynamic
feature selection and reliability-consistency fusion.

## Repository contents

- `train_dsfn.py`: DSFN model definition and checkpoint evaluation.
- `paper_training.py`: training entry point for the manuscript configurations.
- `paper_experiments/reference_trainer.py`: training implementation used for
  the reported experiments.
- `paper_experiments/baseline_models.py`: ResNet1D, TCN, InceptionTime,
  LSTM-FCN, DRSN and MRA-CNN implementations.
- `paper_experiments/run_*_suite.py`: clean, baseline, Gaussian-noise and
  modality-masking evaluations.
- `reproduce.py`: unified training and evaluation command.
- `prepare_paper_data.py`: validates and imports the processed arrays.

## Environment

The reported environment is Python 3.8.20, PyTorch 2.4.1, CUDA 11.8 and Ubuntu
22.04.5. Install the CUDA build of PyTorch first, followed by the pinned
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==24.3.1 typing_extensions==4.12.2
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Data preparation

The experiments use the processed `train_data.npy` and `train_labels.npy`
derived from the Real-Sensors and Real-No Fault subsets of
[RflyMAD](https://rfly-openha.github.io/documents/4_resources/dataset.html).
Import the processed arrays without changing their names or order:

```bash
python prepare_paper_data.py --source-dir /path/to/processed/dataset
```

The command validates the arrays and installs them under `data/paper/`. The
small data manifest and fixed experimental protocol files required by the
training scripts are included in the repository. Large arrays, checkpoints and
generated results are excluded from Git.

Model input has shape `[batch, 9, 1024]`. Channels are grouped into Gyroscope,
Acceleration and Velocity modalities, with three channels per modality.

## Training

Train the complete DSFN with the three manuscript seeds:

```bash
python reproduce.py train --suite dsfn --seeds 42,2025,3407 --device cuda
```

Train the ablations or six horizontal comparison models:

```bash
python reproduce.py train --suite ablations --seeds 42,2025,3407 --device cuda
python reproduce.py train --suite baselines --seeds 42,2025,3407 --device cuda
```

Run every manuscript training configuration:

```bash
python reproduce.py train --suite all --seeds 42,2025,3407 --device cuda
```

By default, checkpoints are written under `runs/`. Use `--checkpoint-root` to
select another output directory.

## Evaluation

Evaluate DSFN and all horizontal comparison models on the clean, noise and
modality-masking protocols:

```bash
python reproduce.py evaluate \
  --suite all \
  --checkpoint-root runs \
  --output-dir results/paper \
  --seeds 42,2025,3407 \
  --device cuda
```

Evaluation outputs contain per-seed predictions and metrics together with the
aggregated values used in the manuscript tables. Gaussian-noise evaluation
perturbs one normalized modality at a time at 0 dB and -5 dB. Modality masking
sets the selected normalized modality to zero while leaving the other two
unchanged.

## Manuscript configurations

| Experiment | Public configuration |
|---|---|
| Static average fusion | `Paper_A0_StaticAvg` |
| Dynamic depth | `Paper_A1_DynamicDepth` |
| Dynamic depth with similarity loss | `Paper_A2_DynamicDepthLoss` |
| Feature-level denoising | `Paper_A3_GTD` |
| Complete DSFN | `Paper_A4_DSFN` |
| Reliability only | `Paper_B1_Reliability` |
| Reliability and consistency | `Paper_B2_ReliabilityConsistency` |
| Reliability and path stability | `Paper_B3_ReliabilityDynamicPath` |

The complete model uses three base Transformer layers and two dynamic layers
per branch. Gyroscope and acceleration patch lengths are 16; velocity patch
length is 32. The reported model does not add positional encoding.

Common training settings are AdamW, learning rate `3e-4`, weight decay `1e-4`,
batch size 64 and early-stopping patience 20. DSFN uses a maximum of 150 epochs
and EMA decay 0.995; horizontal models use a maximum of 120 epochs. Report the
mean and sample standard deviation across seeds 42, 2025 and 3407.

## Citation

Please cite the associated paper when it becomes available. The complete
citation will be added after publication.

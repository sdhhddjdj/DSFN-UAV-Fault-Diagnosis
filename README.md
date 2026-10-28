# DSFN: UAV Fault Diagnosis under Sensor Noise and Modality Loss

PyTorch implementation of **A Robust UAV Sensor Fault Diagnosis Network Based on Feature-Level Denoising and Reliability–Consistency Fusion**.

DSFN uses fixed-depth multi-scale residual dilated CNN encoders, gated feature-level denoising (GTD), and cross-modal reliability–consistency fusion (CRCF). It covers six-class sensor-fault diagnosis on RflyMAD and three-class propeller-fault diagnosis on UAV-FD.

## Model

- Three branches process Gyro, Accel, and Velocity inputs of shape `[B,9,1024]`.
- Each CNN has a convolutional stem and four residual blocks with dilation rates 1, 2, 4, and 8. Patch lengths are 16, 16, and 32, with 128-dimensional token features.
- CRCF scores each modality using its pooled feature, auxiliary prediction confidence, and cross-modal cosine consistency: **130 → 64 → 1**.
- DSFN trains the CNN/fusion network for 80 epochs, then freezes it while training GTD for 40 epochs. Inference correction scales are `(0,0.25,1)`.
- An unavailable modality is excluded from scoring, available weights are renormalized, and GTD is bypassed in **all** branches for that sample.
- Seven comparison models train for 120 epochs. Centaur uses a separate 100-epoch denoising-autoencoder stage and 120-epoch classifier stage.

## Environment

The manuscript experiments used Python 3.8.20, PyTorch 2.4.1, CUDA 11.8, and Ubuntu 22.04.5. For that environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU-only use, install the CPU PyTorch build instead. CPU implementation checks used PyTorch 2.5.1+cpu; see [verification details](docs/IMPLEMENTATION.md).

## Data

Obtain data from the original providers:

- [RflyMAD](https://rfly-openha.github.io/documents/4_resources/dataset.html)
- [UAV-FD, Zenodo record 7648996](https://doi.org/10.5281/zenodo.7648996)

[Data preparation](data/README.md) documents the required channel order and explicit flight/group manifest. Partition complete flight cases and logs **before** extracting windows. Channel normalization and class weights use training windows only.

| Dataset | Classes | Train / validation / test windows | Train / validation / test stride |
| --- | ---: | --- | --- |
| RflyMAD | 6 | 3,207 / 203 / 210 | 512 / 1,024 / 1,024 |
| UAV-FD | 3 | 555 / 123 / 68 | 512 / 512 / 1,024 |

**RflyMAD:** Related flight cases and their logs are grouped before window extraction. The fixed training, validation, and test partitions contain **178/22/22 flight-case groups** and **196/24/25 logs**, respectively. A case and its associated logs stay in one partition. All models and five training seeds use these same partitions.

**UAV-FD:** The 18 flights are partitioned into **12/3/3 training, validation, and test flights**. Fault recordings at motor positions 1–4, 5, and 6 are assigned to these partitions, with normal recordings assigned correspondingly. The [flight assignment](data/manifests/uavfd_flights.csv) is included.

For RflyMAD, supply processed flight files and a small CSV manifest containing `log_id`, `group_id`, and `split`; see [data preparation](data/README.md). The per-log RflyMAD assignment is not bundled. Exact numerical replication requires the same flight assignments and preprocessing as the paper.

```bash
python prepare_paper_data.py --dataset rflymad   --flights-dir /path/to/canonical_flights   --manifest /path/to/flight_manifest.csv --output-dir data/rflymad

python prepare_paper_data.py --output-dir data/rflymad --verify-only
```

For the official UAV-FD MAT files, the original synchronization and flight-event preparation are also provided:

```bash
python prepare_uavfd.py --raw-dir /path/to/official_18_mat_files --output-dir data/uavfd
```

Add `--download` to fetch missing files from Zenodo. The converter preserves flight-segment boundaries, creates the **555/123/68-window** partitions, and records the Gaussian-noise replay metadata.

The generic importer accepts synchronized, aligned flight signals in the documented channel order. Raw data and fitted checkpoints are not bundled.

## Train and evaluate

Train DSFN, or use `--family all` for DSFN and all eight comparison models:

```bash
python reproduce.py train --dataset rflymad --data data/rflymad   --family dsfn --seeds 42,2025,3407,1234,5678 --device cuda --output runs

python reproduce.py evaluate --dataset rflymad --data data/rflymad   --family dsfn --seeds 42,2025,3407,1234,5678 --device cuda --output runs

python reproduce.py summarize --dataset rflymad --family dsfn   --seeds 42,2025,3407,1234,5678 --output runs
```

For UAV-FD, replace `rflymad` with `uavfd` and supply its prepared data directory. Available families are `dsfn`, `resnet1d`, `tcn`, `inceptiontime`, `lstm_fcn`, `drsn`, `mra_cnn`, `cnn_dbf`, and `centaur`.

Evaluation covers clean inputs, six SNR levels (20, 10, 5, 0, −5, −10 dB) applied to each of three modalities, and three modality-masking conditions. It exports condition-level metrics, logits, confusion matrices, seed-level regime means, and DSFN branch confidence/consistency/fusion weights and correctness AUROCs.

Base/classifier checkpoints are selected by mean macro-F1 over the 18 **validation** noise conditions, with the earliest epoch retained on ties. GTD uses the fixed final 40-epoch EMA. Test data are used only by the explicit evaluation command. Existing runs are preserved; use a new output directory for a fresh run. Pass `--noise-bank` to replay an archived standard-Gaussian bank for the selected split.

`Noise Mean`, `Low-SNR Mean`, and `Masking Mean` average 18, 9 (SNR ≤ 0 dB), and 3 conditions respectively **within each seed**. Final tables use the mean and sample standard deviation (`ddof=1`) of the five seed-level values. See [the complete protocol](docs/PROTOCOL.md).

## Manuscript results

The table reports the paper's five-seed mean ± sample standard deviation (%):

| Dataset | Clean accuracy | Noise Mean accuracy | Masking Mean accuracy | Masking Mean macro-F1 |
| --- | --- | --- | --- | --- |
| RflyMAD | 76.19 ± 1.99 | 73.33 ± 1.30 | 69.11 ± 0.53 | 61.83 ± 0.74 |
| UAV-FD | 77.35 ± 13.90 | 77.66 ± 11.35 | 72.65 ± 10.54 | 71.64 ± 11.23 |

[Full nine-model tables and CSVs](docs/RESULTS.md) include both metrics in all four regimes. On UAV-FD, CNN-DBF leads under clean/noisy/low-SNR inputs; DSFN leads the mean metrics under modality masking. The larger UAV-FD standard deviations should be retained when interpreting its results.

Verification includes model consistency and synthetic train–save–load–evaluate tests. Full training on the real datasets has not been repeated using this package; see [verification details](docs/IMPLEMENTATION.md). The UAV-FD converter supplies noise-replay metadata; for other prepared data, `--noise-bank` accepts a matching saved Gaussian bank.

## Code map

- `dsfn/models.py`: DSFN, CRCF, GTD, and model/loss builders.
- `dsfn/baseline_models.py`, `dsfn/cnn_dbf.py`, `dsfn/centaur.py`: comparison architectures.
- `dsfn/training.py`, `dsfn/centaur_training.py`: staged training and checkpoint selection.
- `dsfn/data.py`, `prepare_paper_data.py`: grouped preparation and training-only statistics.
- `dsfn/protocol.py`, `dsfn/evaluate.py`: shared conditions and metric export.
- `configs/`: final manuscript configurations.
- `paper_results/`: reported reference tables, without sensor samples.

```bash
python -m unittest discover -s tests -v
```

## Attribution and citation

Please cite the associated manuscript when available; the publication citation will be added after acceptance. The CNN-DBF and Centaur adaptations retain their upstream attribution and separate licenses in [third_party](third_party/README.md). The original datasets remain subject to their providers' terms.

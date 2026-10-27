# DSFN: UAV Fault Diagnosis under Sensor Noise and Modality Loss

PyTorch implementation for the revised manuscript **A Robust UAV Sensor Fault Diagnosis Network Based on Feature-Level Denoising and Reliability–Consistency Fusion** (MST-139726).

The revised DSFN uses fixed-depth multi-scale residual dilated CNN encoders, gated feature-level denoising (GTD), and cross-modal reliability–consistency fusion (CRCF). It covers six-class sensor-fault diagnosis on RflyMAD and three-class propeller-fault diagnosis on UAV-FD.

## Revised implementation

- Three branches process Gyro, Accel, and Velocity inputs of shape `[B,9,1024]`.
- Each CNN has a convolutional stem and four residual blocks with dilation rates 1, 2, 4, and 8. Patch lengths are 16, 16, and 32, with 128-dimensional token features.
- CRCF scores each modality using its pooled feature, auxiliary prediction confidence, and cross-modal cosine consistency: **130 → 64 → 1**.
- DSFN trains the CNN/fusion network for 80 epochs, then freezes it while training GTD for 40 epochs. Inference correction scales are `(0,0.25,1)`.
- An unavailable modality is excluded from scoring, available weights are renormalized, and GTD is bypassed in **all** branches for that sample.
- Seven comparison models train for 120 epochs. Centaur uses a separate 100-epoch denoising-autoencoder stage and 120-epoch classifier stage.

The previous Transformer encoder, dynamic-layer selector, entropy-based path-stability term, and early-exit mechanism are not part of the revised model. The original implementation remains in Git history. Old window-level split files are retired.

## Environment

The manuscript experiments used Python 3.8.20, PyTorch 2.4.1, CUDA 11.8, and Ubuntu 22.04.5. For that environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU-only use, install the CPU PyTorch build instead. The portable export was tested locally with PyTorch 2.5.1+cpu; see [implementation checks](docs/IMPLEMENTATION.md).

## Data

Obtain data from the original providers:

- [RflyMAD](https://rfly-openha.github.io/documents/4_resources/dataset.html)
- [UAV-FD, Zenodo record 7648996](https://doi.org/10.5281/zenodo.7648996)

[Data preparation](data/README.md) documents the required channel order and explicit flight/group manifest. Partition complete flight cases and logs **before** extracting windows. Channel normalization and class weights use training windows only.

| Dataset | Classes | Train / validation / test windows | Train / validation / test stride |
| --- | ---: | --- | --- |
| RflyMAD | 6 | 3,207 / 203 / 210 | 512 / 1,024 / 1,024 |
| UAV-FD | 3 | 555 / 123 / 68 | 512 / 512 / 1,024 |

UAV-FD uses held-out fault-motor-position/log groups, not geographic locations. Its final test set contains **68 nonoverlapping windows**, replacing the earlier 134-window evaluation. The [18-flight assignment](data/manifests/uavfd_flights.csv) is included. RflyMAD's exact case/log assignment and processed arrays are not included in this export; obtain them alongside the processed logs for exact table replication. Matching window counts alone does not establish an identical split.

```bash
python prepare_paper_data.py --dataset rflymad   --flights-dir /path/to/canonical_flights   --manifest /path/to/flight_manifest.csv --output-dir data/rflymad

python prepare_paper_data.py --output-dir data/rflymad --verify-only
```

For the official UAV-FD MAT files, the original synchronization and flight-event preparation are also provided:

```bash
python prepare_uavfd.py --raw-dir /path/to/official_18_mat_files --output-dir data/uavfd
```

Add `--download` only when the script should fetch the files from Zenodo. This converter keeps separate flight segments and the final nonoverlapping-test selection, then checks 555/123/68 windows. It also records the original 134-window Gaussian-bank positions so the evaluator can replay the exact 68-window noise subset.

The generic canonical importer does not replace the original raw-log synchronization/conversion. Do not reuse the old shuffled `train_data.npy` split as the revised grouped protocol. Raw data and fitted checkpoints are not bundled.

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

These are reported five-seed results (%), not new training results from the portable export:

| Dataset | Clean accuracy | Noise Mean accuracy | Masking Mean accuracy | Masking Mean macro-F1 |
| --- | --- | --- | --- | --- |
| RflyMAD | 76.19 ± 1.99 | 73.33 ± 1.30 | 69.11 ± 0.53 | 61.83 ± 0.74 |
| UAV-FD | 77.35 ± 13.90 | 77.66 ± 11.35 | 72.65 ± 10.54 | 71.64 ± 11.23 |

[Full nine-model tables and CSVs](docs/RESULTS.md) include both metrics in all four regimes. On UAV-FD, CNN-DBF leads under clean/noisy/low-SNR inputs; DSFN leads the mean metrics under modality masking. The larger UAV-FD standard deviations should be retained when interpreting its results.

The architecture and missing-modality inference were compared with the archived experiment source. The full training experiments have **not** been rerun in this exported repository. Exact metric replay also requires the matching preprocessing, split assignments, initialization/checkpoint files, and Gaussian banks. The official UAV-FD converter records the original test-bank positions for automatic subset replay; generic imported data without that metadata uses a newly generated bank unless `--noise-bank` is provided. See [implementation verification](docs/IMPLEMENTATION.md).

## Code map

- `dsfn/models.py`: revised DSFN, CRCF, GTD, and model/loss builders.
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

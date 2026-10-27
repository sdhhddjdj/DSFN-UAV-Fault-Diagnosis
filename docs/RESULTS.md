# Manuscript-reported results

These are the five-seed results reported in the revised manuscript. They were transcribed directly from manuscript Tables 2 and 4 (RflyMAD) and Table 8 (UAV-FD). **They are not new validation results from this repository port.** Running the code in this repository does not by itself establish that these numbers have been reproduced.

Values are mean ± sample standard deviation across training seeds `42, 2025, 3407, 1234, 5678`, expressed in percent. The published values are retained at their reported two-decimal precision. For aggregate regimes, conditions are averaged within each seed first; the mean and sample standard deviation (`ddof=1`) are then computed across the five seed-level values.

Clean contains one condition; Noise Mean averages 18 conditions (three individually corrupted modalities × six SNR levels); Low-SNR Mean averages the nine conditions at 0, −5, and −10 dB; Masking Mean averages three single-modality masking conditions. See [the experimental protocol](PROTOCOL.md) for dataset partitions and evaluation definitions.

## RflyMAD

Source: manuscript Tables 2 and 4. Six sensor-fault classes; 210 non-overlapping test windows.

### Accuracy (%)

| Model | Clean | Noise Mean | Low-SNR Mean | Masking Mean |
| --- | ---: | ---: | ---: | ---: |
| DSFN | 76.19 ± 1.99 | 73.33 ± 1.30 | 71.98 ± 0.99 | 69.11 ± 0.53 |
| ResNet1D | 72.19 ± 0.87 | 69.42 ± 0.96 | 66.80 ± 1.24 | 52.73 ± 0.72 |
| TCN | 71.14 ± 2.25 | 68.27 ± 2.03 | 65.74 ± 2.12 | 54.06 ± 1.94 |
| InceptionTime | 71.81 ± 1.59 | 70.17 ± 1.55 | 68.79 ± 1.69 | 53.52 ± 1.01 |
| LSTM-FCN | 71.33 ± 1.23 | 67.48 ± 0.72 | 64.14 ± 0.63 | 51.30 ± 1.65 |
| DRSN | 66.10 ± 1.09 | 63.33 ± 1.38 | 60.19 ± 1.50 | 52.29 ± 1.09 |
| MRA-CNN | 64.19 ± 2.11 | 61.76 ± 2.14 | 59.42 ± 2.31 | 50.92 ± 1.35 |
| CNN-DBF | 70.95 ± 1.78 | 68.52 ± 1.69 | 67.21 ± 1.50 | 47.33 ± 4.57 |
| Centaur | 30.48 ± 5.61 | 30.32 ± 5.56 | 30.26 ± 5.54 | 24.22 ± 3.56 |

### Macro-F1 (%)

| Model | Clean | Noise Mean | Low-SNR Mean | Masking Mean |
| --- | ---: | ---: | ---: | ---: |
| DSFN | 70.12 ± 3.00 | 66.70 ± 1.82 | 65.16 ± 1.28 | 61.83 ± 0.74 |
| ResNet1D | 64.27 ± 0.88 | 61.76 ± 0.67 | 59.41 ± 0.83 | 42.35 ± 1.24 |
| TCN | 62.82 ± 1.93 | 60.36 ± 1.76 | 58.23 ± 2.03 | 43.24 ± 2.14 |
| InceptionTime | 63.24 ± 1.85 | 61.93 ± 1.95 | 60.81 ± 2.06 | 43.11 ± 1.40 |
| LSTM-FCN | 64.72 ± 1.91 | 60.75 ± 1.26 | 57.34 ± 1.15 | 42.56 ± 1.16 |
| DRSN | 58.76 ± 0.70 | 56.29 ± 1.25 | 53.51 ± 1.34 | 41.41 ± 1.20 |
| MRA-CNN | 56.03 ± 2.17 | 53.94 ± 2.38 | 51.88 ± 2.77 | 39.86 ± 2.64 |
| CNN-DBF | 65.09 ± 1.65 | 62.55 ± 1.54 | 61.33 ± 1.26 | 42.98 ± 4.58 |
| Centaur | 26.39 ± 4.15 | 25.94 ± 3.92 | 25.58 ± 3.81 | 18.41 ± 2.51 |

Machine-readable values: [`rflymad_summary.csv`](../paper_results/rflymad_summary.csv).

## UAV-FD

Source: manuscript Table 8. Three propeller-damage classes; the final non-overlapping test subset contains 68 windows from three held-out flight logs.

### Accuracy (%)

| Model | Clean | Noise Mean | Low-SNR Mean | Masking Mean |
| --- | ---: | ---: | ---: | ---: |
| DSFN | 77.35 ± 13.90 | 77.66 ± 11.35 | 77.32 ± 9.71 | 72.65 ± 10.54 |
| ResNet1D | 42.94 ± 3.19 | 42.86 ± 2.08 | 43.24 ± 1.99 | 38.63 ± 2.76 |
| TCN | 43.82 ± 2.83 | 44.10 ± 0.99 | 44.08 ± 1.63 | 40.69 ± 2.79 |
| InceptionTime | 61.76 ± 5.79 | 56.67 ± 5.01 | 53.24 ± 4.52 | 48.92 ± 3.22 |
| LSTM-FCN | 44.12 ± 3.89 | 46.23 ± 4.06 | 47.68 ± 4.59 | 39.41 ± 2.91 |
| DRSN | 43.53 ± 5.94 | 44.26 ± 5.07 | 44.77 ± 4.66 | 37.35 ± 4.00 |
| MRA-CNN | 52.06 ± 4.83 | 52.73 ± 3.44 | 53.20 ± 2.30 | 40.98 ± 3.71 |
| CNN-DBF | 95.59 ± 1.80 | 94.82 ± 1.53 | 94.41 ± 1.65 | 67.16 ± 5.15 |
| Centaur | 37.94 ± 2.63 | 38.45 ± 3.58 | 39.12 ± 3.84 | 41.67 ± 1.83 |

### Macro-F1 (%)

| Model | Clean | Noise Mean | Low-SNR Mean | Masking Mean |
| --- | ---: | ---: | ---: | ---: |
| DSFN | 77.36 ± 14.46 | 77.49 ± 11.73 | 77.10 ± 10.04 | 71.64 ± 11.23 |
| ResNet1D | 42.14 ± 3.72 | 42.22 ± 2.29 | 42.79 ± 2.04 | 37.03 ± 2.56 |
| TCN | 42.20 ± 2.97 | 42.66 ± 0.96 | 42.90 ± 1.46 | 39.54 ± 2.86 |
| InceptionTime | 61.34 ± 6.05 | 55.82 ± 5.18 | 52.17 ± 4.52 | 45.75 ± 3.73 |
| LSTM-FCN | 42.22 ± 4.13 | 44.46 ± 3.80 | 46.04 ± 3.99 | 38.34 ± 2.90 |
| DRSN | 42.04 ± 6.48 | 43.12 ± 5.05 | 43.83 ± 4.33 | 35.28 ± 4.36 |
| MRA-CNN | 48.66 ± 4.25 | 48.84 ± 3.47 | 48.71 ± 3.14 | 38.41 ± 3.83 |
| CNN-DBF | 95.76 ± 1.81 | 95.05 ± 1.45 | 94.68 ± 1.53 | 66.25 ± 6.17 |
| Centaur | 35.68 ± 3.42 | 36.11 ± 3.85 | 36.65 ± 4.02 | 37.72 ± 3.58 |

Machine-readable values: [`uavfd_summary.csv`](../paper_results/uavfd_summary.csv).

## Interpretation and provenance

On RflyMAD, DSFN has the highest reported means for both metrics in all four aggregate regimes. On UAV-FD, CNN-DBF has the highest reported means under clean inputs, noise, and low SNR; DSFN has the highest reported means under modality masking. These are comparisons of reported means, not claims of statistical significance.

The two datasets have different label spaces and should not be pooled into one classification score. UAV-FD uses a fixed 12/3/3-flight training/validation/test partition, and the relatively large DSFN seed standard deviations should remain visible when reporting its means.

The CSV schema is `dataset,model,regime,metric,mean_percent,std_percent`. Each CSV has 72 rows: nine models × four regimes × two metrics. Earlier BASiC results, the earlier 134-window UAV-FD test evaluation, and superseded 80-epoch comparison runs are not included in these tables.

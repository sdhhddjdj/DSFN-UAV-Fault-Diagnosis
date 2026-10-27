# Revised DSFN experimental protocol

This document describes the final CNN–GTD–CRCF model and the fixed dataset partitions used in the revised manuscript. The companion JSON files record the common experiment settings; they do not contain sensor samples or replace the original flight/log split manifests.

## Model

The input is a nine-channel window of 1,024 time steps, ordered as three gyroscope channels, three accelerometer channels, and three velocity channels. Each modality has an independent CNN branch with the same architecture:

- A kernel-3 convolutional stem maps 3 channels to 32 channels, followed by batch normalization and ReLU.
- Four residual blocks use two kernel-5 dilated convolutions per block, with dilation rates 1, 2, 4, and 8. The temporal CNN has a theoretical receptive field of 123 steps.
- Patch projections use kernel size and stride 16 for Gyro and Accel, and 32 for Vel. They produce 64, 64, and 32 tokens, respectively, with a common feature dimension of 128.
- GTD applies gated residual corrections to token features. Its local correction uses layer normalization, a kernel-5 depthwise convolution, and a kernel-1 pointwise convolution. The channel gate uses a 128→32→128 network. The final correction projection is initialized to zero.
- CRCF scores each available modality from its pooled 128-dimensional feature, maximum auxiliary-classifier softmax probability, and mean mapped cosine similarity to the other available modalities. A shared 130→64→1 network produces scores; softmax normalizes the fusion weights over available modalities. A linear classifier operates on the weighted pooled representation.

The revised model uses fixed-depth CNN encoders. Dynamic Transformer selection, entropy-based path stability, an inference threshold, and early exit are not part of this model. The auxiliary softmax confidence is a confidence proxy, rather than a calibrated probability of correctness.

## Datasets and partitions

Flight cases/logs are assigned to partitions before selecting windows. Training, validation, and test data are disjoint at the relevant flight/log level. The five seeds repeat training on these same fixed partitions; they are not five independently resampled splits.

| Dataset | Classes | Training | Validation | Test |
| --- | --- | --- | --- | --- |
| RflyMAD | Normal, Accel, GPS, Gyro, Mag, Baro | 3,207 windows; 178 flight-case groups; 196 logs | 203 windows; 22 groups; 24 logs | 210 windows; 22 groups; 25 logs |
| UAV-FD | Normal, Damage5, Damage10 | 555 windows; 12 flights | 123 windows; 3 flights | 68 windows; 3 flights |

RflyMAD class counts, in the order above, are `[1316, 271, 368, 343, 409, 500]` for training, `[82, 20, 25, 24, 26, 26]` for validation, and `[82, 23, 20, 25, 34, 26]` for testing.

UAV-FD is a separate real-flight propeller-damage classification task. It does not use the six RflyMAD sensor-fault labels. Its 18 logs comprise six normal flights and two damage severities at each of six motor positions. Fault recordings for motor positions 1–4 are assigned to training, position 5 to validation, and position 6 to testing. Normal recordings 1–4, 5, and 6 are assigned correspondingly. Both damage severities of a motor position stay in the same partition. This is a held-out motor-position and flight-log partition, not a split by geographic location. Class counts are `[158, 174, 223]`, `[35, 31, 57]`, and `[21, 17, 30]` for training, validation, and testing.

### Windowing and preprocessing

| Setting | RflyMAD | UAV-FD |
| --- | --- | --- |
| Window length | 1,024 steps | 1,024 steps |
| Training stride | 512 | 512 |
| Validation stride | 1,024 | 512 |
| Test stride | 1,024 | 1,024 |

Validation windows therefore overlap within the UAV-FD validation logs, while validation windows on RflyMAD and test windows on both datasets do not overlap. This within-partition overlap does not place a flight log in more than one partition.

UAV-FD angular velocity, acceleration, and estimated velocity are synchronized from their own timestamps onto a common 350-Hz grid using zero-order hold. The three redundant IMU streams and two velocity estimates are averaged after synchronization. Valid airborne intervals and valid signal regions are selected before window extraction; a window does not cross a flight segment or an invalid region. The retained channels are `Gyro_X, Gyro_Y, Gyro_Z, Accel_X, Accel_Y, Accel_Z, Vel_N, Vel_E, Vel_D`.

The final UAV-FD test set is a deterministic non-overlapping subset of the earlier 134-window, stride-512 test bank. Within each flight/segment and contiguous valid window chain, chronological positions 0, 2, 4, … are retained, restarting after gaps. This yields 68 windows. The final results reuse the trained checkpoints and original noise realization for each retained sample; there is no retraining, checkpoint reselection, or performance-based sample selection in this step.

Channel-normalization statistics and class weights are fitted using only the training partition. DSFN uses per-channel mean and population standard deviation. Centaur additionally retains its native training-fitted per-channel min–max transformation. Models are trained independently on each dataset; fitted weights are not transferred from RflyMAD to UAV-FD. The RflyMAD sample rate is not restated in the accompanying configuration until a canonical preprocessing specification is supplied; its `sample_rate_hz` is `null` rather than an assumed frequency.

## Training and model selection

The five training seeds are `42, 2025, 3407, 1234, 5678`.

DSFN uses 80 epochs of base-network training followed by 40 epochs of GTD training. Base training jointly optimizes the CNNs, patch projections, auxiliary classifiers, CRCF, and fused classifier without GTD. It uses AdamW, batch size 64, learning rate `3e-4`, weight decay `1e-4`, cosine decay to `3e-6`, and exponential moving average decay `0.995`. The base objective is class-weighted fused cross-entropy plus `0.1` times the mean auxiliary cross-entropy and `0.1` times the batch-mean fusion-weight balancing loss.

The base checkpoint is selected by the highest mean macro-F1 over the 18 validation noise conditions; the earliest checkpoint is retained in a tie. GTD training freezes all base-network parameters and batch-normalization running statistics. It uses paired clean and artificially corrupted training windows, with GTD scales `(1, 1, 1)`. GTD evaluation uses its fixed epoch-40 averaged weights.

The inference correction scales are fixed at `(Gyro, Accel, Vel) = (0, 0.25, 1)`, selected using RflyMAD validation performance. These same values are used on UAV-FD, with no UAV-FD test-set tuning. They are residual-correction scales, not modality-fusion weights.

The seven comparison models—ResNet1D, TCN, InceptionTime, LSTM-FCN, DRSN, MRA-CNN, and CNN-DBF—are trained for 120 epochs. The final 120-epoch runs start from initialization with a complete 120-epoch schedule; they are not a continuation of selected epoch-80 checkpoints. Centaur uses 100 epochs of denoising-autoencoder training and 120 epochs of classifier training. For the RflyMAD classifier extension, its already trained 100-epoch autoencoder is reused and the classifier is restarted from its original initialization. UAV-FD comparison runs are initialized independently on UAV-FD.

The JSON learning-rate/weight-decay/scheduler settings describe DSFN and the seven standard comparison models. They should not be interpreted as an override of Centaur's separate native two-stage optimizer and normalization recipe.

## Test conditions and missing-modality behavior

Each model is evaluated under 22 conditions: one clean condition, 18 single-modality noise conditions, and three single-modality masking conditions.

- Noise is zero-mean Gaussian noise added separately to normalized Gyro, Accel, or Vel channels at SNRs `20, 10, 5, 0, -5, -10 dB`. The two other modalities remain unchanged. For each sample, signal power is the target modality's mean squared amplitude across its three channels and all time steps. Noise standard deviation is `sqrt(signal_power / 10**(SNR/10))`. The same condition/sample noise bank is shared across comparison models.
- Masking sets all normalized channels of exactly one modality to zero. The other two modalities remain unchanged. DSFN excludes the unavailable branch from CRCF scoring, assigns it zero fusion weight, and renormalizes the weights over the remaining branches. If any complete modality is absent, **GTD is bypassed for all branches**, preserving the available branches' base features.

The primary metrics are classification accuracy and macro-F1. Accuracy and each class's F1 are computed on the complete selected test set for each condition and seed. Macro-F1 is the unweighted mean of class F1 scores.

| Reported regime | Conditions averaged within each seed |
| --- | --- |
| Clean | One clean condition |
| Noise Mean | All 18 modality–SNR conditions |
| Low-SNR Mean | Nine conditions: three modalities × SNRs 0, −5, −10 dB |
| Masking Mean | Three single-modality masking conditions |

Conditions are averaged within each seed first. The reported value is then the mean and **sample standard deviation (`ddof=1`) across the five seed-level values**, in percent. This standard deviation is not the spread across SNRs or across individual test windows. Low-SNR Mean is a subset of Noise Mean, not an additional experiment set.

The Figure 5 correctness AUROC is a separate diagnostic: the positive label is whether a modality's auxiliary classifier predicted correctly, and the score is that modality's confidence or cosine consistency. It is computed per seed, branch, and noise condition. It is not multiclass classification AUROC and does not replace accuracy or macro-F1.

## Scope of the reported evidence

RflyMAD and UAV-FD use different label spaces and are evaluated separately. UAV-FD contains only 18 flight logs and one fixed motor-position split; the results are not six-fold cross-validation or zero-shot cross-dataset transfer. UAV-FD acquisition batches also differ between the two damage severities, which limits separation of severity and acquisition effects. The repository's final comparison should use the 68-window UAV-FD test evaluation and the final 120-epoch baseline results, rather than earlier 134-window/80-epoch or BASiC results.

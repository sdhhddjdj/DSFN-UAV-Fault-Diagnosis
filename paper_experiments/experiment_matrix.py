"""Shared definitions of the manuscript's comparison suite.

The configuration of every ablation row lives in ``train_dsfn.ABLATION_CONFIGS``
and is selected with ``DSFN_EXP_NAME``; this module only holds the values that
are shared by the experiment scripts: the reported seeds, the horizontal
comparison models and the robustness protocol of Section 4.1.3.
"""

from __future__ import annotations


# Seeds used for every reported mean +/- standard deviation.
DEFAULT_SEEDS = [3407, 42, 2025]

# The complete model reported in the manuscript (Table 1, row A4).
MANUSCRIPT_MODEL = "Paper_A4_DSFN"

# Section 4.1.3 protocol. Gaussian noise is added to one modality at a time
# (Gyro, Accel, Vel) at 0 dB and -5 dB, giving the six modality-SNR conditions
# aggregated in Table 4. Masking sets every channel of the selected modality to
# zero (Table 5). No model is retrained for these evaluations.
PAPER_NOISE_MODALITIES = ["gyro", "accel", "vel"]
PAPER_LOW_SNR_DB = [0.0, -5.0]
PAPER_MASK_MODALITIES = ["gyro", "accel", "vel"]

# Tables 3-5: the six comparison models, in the order they appear in the paper.
HORIZONTAL_BASELINES = [
    {
        "name": "resnet1d",
        "table_name": "ResNet1D",
        "family": "residual learning",
    },
    {
        "name": "tcn",
        "table_name": "TCN",
        "family": "temporal convolution",
    },
    {
        "name": "inceptiontime",
        "table_name": "InceptionTime",
        "family": "multiscale convolution",
    },
    {
        "name": "lstm_fcn",
        "table_name": "LSTM-FCN",
        "family": "hybrid recurrent-convolutional modeling",
    },
    {
        "name": "drsn",
        "table_name": "DRSN",
        "family": "adaptive soft-thresholding",
    },
    {
        "name": "mra_cnn",
        "table_name": "MRA-CNN",
        "family": "multiscale residual attention",
    },
]

"""DSFN: Denoising and Similarity-Guided Fusion Network for UAV sensor fault diagnosis.

Reference implementation of the model described in the manuscript: three
modality-specific CNN-Transformer encoders with a dynamic-depth selector, a
gated feature-level denoising (GTD) module, and a cross-modal
reliability-consistency fusion (CRCF) module.

Input: 9-channel, 1,024-step windows (Gyro, Accel and Vel, three channels each).
Output: six classes (Normal, Accel, GPS, Gyro, Mag, Baro).

Run ``python train_dsfn.py`` to train the model reported in the manuscript;
see README.md for the environment variables and the ablation configurations.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 强制使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import copy
import time
import math
from sklearn.metrics import classification_report
device = "cuda" if torch.cuda.is_available() else "cpu"
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

# Manuscript coefficients: similarity-scaling coefficient T_s (Eq. (12)),
# similarity-stability weight lambda_s (Eq. (15)) and the weight lambda_sim of
# the similarity-guided loss in the total objective (Eq. (28)).
SIMILARITY_SCALE = 0.1
SIMILARITY_STABILITY_WEIGHT = 0.5
SIMILARITY_LOSS_WEIGHT = 0.01


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_bool(value):
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _env_float_list(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip() == "":
        return []
    if value.strip().lower() in {"none", "null", "off", "false"}:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]
# ==========================================
# 1. Configuration
# ==========================================
# The defaults below are the configuration reported in the manuscript. Every
# value can be overridden through the matching DSFN_* environment variable,
# and the experiment definitions in ABLATION_CONFIGS select the individual
# ablation rows of Tables 1 and 2.
CONFIG = {
    "batch_size": _env_int("DSFN_BATCH_SIZE", 64),
    "epochs": _env_int("DSFN_EPOCHS", 150),
    "patience": _env_int("DSFN_PATIENCE", 20),
    "lr": _env_float("DSFN_LR", 3e-4),
    "weight_decay": _env_float("DSFN_WEIGHT_DECAY", 1e-4),
    "num_classes": 6,
    "data_root": os.environ.get("DSFN_DATA_ROOT", "dataset"),

    # Experiment name selects one entry of ABLATION_CONFIGS. The default is the
    # complete model (Table 1, row A4).
    "exp_name": "Paper_A4_DSFN",

    # Model structure
    "use_temporal_cnn": True,
    "use_gate_dynamic_weight": True,
    "use_dynamic_depth": True,
    "use_similarity_loss": True,
    "use_prefusion_denoise": True,
    "use_outer_gate": True,
    "use_crcf": True,
    "transformer_depth": _env_int("DSFN_TRANSFORMER_DEPTH", 5),
    "transformer_split_layer": _env_int("DSFN_TRANSFORMER_SPLIT_LAYER", 3),
    "gate_threshold": 0.25,
    "crcf_aux_weight": _env_float("DSFN_CRCF_AUX_WEIGHT", 0.05),
    "crcf_balance_weight": _env_float("DSFN_CRCF_BALANCE_WEIGHT", 0.01),
    "crcf_use_reliability": _env_bool("DSFN_CRCF_USE_RELIABILITY", True),
    "crcf_use_consistency": _env_bool("DSFN_CRCF_USE_CONSISTENCY", True),
    "crcf_use_certainty": _env_bool("DSFN_CRCF_USE_CERTAINTY", True),

    # Training strategy
    "use_aug": _env_bool("DSFN_USE_AUG", True),
    "aug_noise_std": _env_float("DSFN_AUG_NOISE_STD", 0.005),
    "aug_drift_prob": _env_float("DSFN_AUG_DRIFT_PROB", 0.5),
    "aug_scale_min": _env_float("DSFN_AUG_SCALE_MIN", 0.98),
    "aug_scale_max": _env_float("DSFN_AUG_SCALE_MAX", 1.02),
    "aug_time_shift_max": _env_int("DSFN_AUG_TIME_SHIFT_MAX", 0),
    "use_class_weight": _env_bool("DSFN_USE_CLASS_WEIGHT", True),
    "label_smoothing": _env_float("DSFN_LABEL_SMOOTHING", 0.0),
    "use_channel_norm": True,
    "use_ema": _env_bool("DSFN_USE_EMA", True),
    "ema_decay": _env_float("DSFN_EMA_DECAY", 0.995),
    "model_dropout": _env_float("DSFN_MODEL_DROPOUT", 0.2),

    # Evaluation protocol (Section 4.1.2): all candidate dynamic layers are
    # executed and their outputs are aggregated with the selector probabilities.
    "eval_force_full": _env_bool("DSFN_EVAL_FORCE_FULL", True),
    "eval_soft_select": _env_bool("DSFN_EVAL_SOFT_SELECT", True),

    # Fixed window-level split
    "split_dir": os.environ.get(
        "DSFN_SPLIT_DIR", os.path.join("dataset", "fixed_split_80_10_10")
    ),
    "split_seed": _env_int("DSFN_SPLIT_SEED", 42),
    "seed": _env_int("DSFN_SEED", 3407),

    # Runtime
    "fps_num_batches": _env_int("DSFN_FPS_NUM_BATCHES", 16),
    "fps_batch_size": _env_int("DSFN_FPS_BATCH_SIZE", 1),
    "save_root": os.environ.get("DSFN_SAVE_ROOT", "runs_dsfn_80_10_10"),
    "init_weights": os.environ.get("DSFN_INIT_WEIGHTS", ""),
}
CONFIG["run_final_reports"] = _env_bool("DSFN_RUN_FINAL_REPORTS", True)
CONFIG["clean_test_only"] = _env_bool("DSFN_CLEAN_TEST_ONLY", False)
CONFIG["save_dir"] = os.path.join(
    CONFIG["save_root"],
    f"{CONFIG['exp_name']}_seed{CONFIG['seed']}",
)
os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ==========================================
# 2. Manuscript experiment definitions
# ==========================================
# Table 1 (incremental ablation) and Table 2 (evidence components of CRCF).
MANUSCRIPT_ABLATION_CONFIGS = {
    # A0: CNN-Transformer backbone with static average fusion.
    "Paper_A0_StaticAvg": {
        "use_temporal_cnn": True,
        "use_gate_dynamic_weight": False,
        "use_dynamic_depth": False,
        "use_similarity_loss": False,
        "use_prefusion_denoise": False,
        "use_outer_gate": False,
        "use_crcf": False,
        "label_smoothing": 0.0,
    },
    # A1: adds the dynamic-depth selector.
    "Paper_A1_DynamicDepth": {
        "use_temporal_cnn": True,
        "use_gate_dynamic_weight": False,
        "use_dynamic_depth": True,
        "use_similarity_loss": False,
        "use_prefusion_denoise": False,
        "use_outer_gate": False,
        "use_crcf": False,
        "label_smoothing": 0.0,
    },
    # A2: adds the adjacent-layer similarity-guided loss.
    "Paper_A2_DynamicDepthLoss": {
        "use_temporal_cnn": True,
        "use_gate_dynamic_weight": False,
        "use_dynamic_depth": True,
        "use_similarity_loss": True,
        "use_prefusion_denoise": False,
        "use_outer_gate": False,
        "use_crcf": False,
        "label_smoothing": 0.0,
    },
    # A3: adds the gated feature-level denoising (GTD) module.
    "Paper_A3_GTD": {
        "use_temporal_cnn": True,
        "use_gate_dynamic_weight": True,
        "use_dynamic_depth": True,
        "use_similarity_loss": True,
        "use_prefusion_denoise": True,
        "use_outer_gate": True,
        "use_crcf": False,
        "label_smoothing": 0.0,
    },
}

# A4: the complete model, adding reliability-consistency fusion (CRCF).
MANUSCRIPT_ABLATION_CONFIGS["Paper_A4_DSFN"] = {
    **MANUSCRIPT_ABLATION_CONFIGS["Paper_A3_GTD"],
    "use_crcf": True,
    "crcf_aux_weight": 0.05,
    "crcf_balance_weight": 0.01,
    "crcf_use_reliability": True,
    "crcf_use_consistency": True,
    "crcf_use_certainty": True,
}

# Table 2: evidence components used by CRCF. B0 and B4 reuse the A3 / A4 runs.
MANUSCRIPT_ABLATION_CONFIGS.update(
    {
        "Paper_B0_AvgFusion": {
            **MANUSCRIPT_ABLATION_CONFIGS["Paper_A3_GTD"],
        },
        "Paper_B1_Reliability": {
            **MANUSCRIPT_ABLATION_CONFIGS["Paper_A4_DSFN"],
            "crcf_use_consistency": False,
            "crcf_use_certainty": False,
        },
        "Paper_B2_ReliabilityConsistency": {
            **MANUSCRIPT_ABLATION_CONFIGS["Paper_A4_DSFN"],
            "crcf_use_certainty": False,
        },
        "Paper_B3_ReliabilityDynamicPath": {
            **MANUSCRIPT_ABLATION_CONFIGS["Paper_A4_DSFN"],
            "crcf_use_consistency": False,
        },
        "Paper_B4_CRCF": {
            **MANUSCRIPT_ABLATION_CONFIGS["Paper_A4_DSFN"],
        },
    }
)

ABLATION_CONFIGS = MANUSCRIPT_ABLATION_CONFIGS


def apply_ablation_config(exp_name):
    """Apply one manuscript configuration and point the run directory at it."""
    if exp_name not in ABLATION_CONFIGS:
        raise KeyError(
            f"Unknown experiment '{exp_name}'. "
            f"Available: {sorted(ABLATION_CONFIGS)}"
        )
    CONFIG.update(ABLATION_CONFIGS[exp_name])
    CONFIG["exp_name"] = exp_name
    CONFIG["save_dir"] = os.path.join(
        CONFIG["save_root"],
        f"{exp_name}_seed{CONFIG['seed']}",
    )
    os.makedirs(CONFIG["save_dir"], exist_ok=True)


def apply_runtime_overrides():
    CONFIG["exp_name"] = os.environ.get("DSFN_EXP_NAME", CONFIG["exp_name"])
    CONFIG["data_root"] = os.environ.get("DSFN_DATA_ROOT", CONFIG["data_root"])
    CONFIG["split_dir"] = os.environ.get("DSFN_SPLIT_DIR", CONFIG["split_dir"])
    CONFIG["save_root"] = os.environ.get("DSFN_SAVE_ROOT", CONFIG["save_root"])
    CONFIG["init_weights"] = os.environ.get("DSFN_INIT_WEIGHTS", CONFIG["init_weights"])
    CONFIG["seed"] = _env_int("DSFN_SEED", CONFIG["seed"])
    CONFIG["split_seed"] = _env_int("DSFN_SPLIT_SEED", CONFIG["split_seed"])
    CONFIG["epochs"] = _env_int("DSFN_EPOCHS", CONFIG["epochs"])
    CONFIG["patience"] = _env_int("DSFN_PATIENCE", CONFIG["patience"])
    CONFIG["batch_size"] = _env_int("DSFN_BATCH_SIZE", CONFIG["batch_size"])
    CONFIG["lr"] = _env_float("DSFN_LR", CONFIG["lr"])
    CONFIG["weight_decay"] = _env_float("DSFN_WEIGHT_DECAY", CONFIG["weight_decay"])
    CONFIG["fps_num_batches"] = _env_int("DSFN_FPS_NUM_BATCHES", CONFIG["fps_num_batches"])
    CONFIG["fps_batch_size"] = _env_int("DSFN_FPS_BATCH_SIZE", CONFIG["fps_batch_size"])
    CONFIG["label_smoothing"] = _env_float("DSFN_LABEL_SMOOTHING", CONFIG["label_smoothing"])
    CONFIG["crcf_aux_weight"] = _env_float("DSFN_CRCF_AUX_WEIGHT", CONFIG["crcf_aux_weight"])
    CONFIG["crcf_balance_weight"] = _env_float(
        "DSFN_CRCF_BALANCE_WEIGHT", CONFIG["crcf_balance_weight"]
    )
    CONFIG["crcf_use_reliability"] = _env_bool(
        "DSFN_CRCF_USE_RELIABILITY", CONFIG["crcf_use_reliability"]
    )
    CONFIG["crcf_use_consistency"] = _env_bool(
        "DSFN_CRCF_USE_CONSISTENCY", CONFIG["crcf_use_consistency"]
    )
    CONFIG["crcf_use_certainty"] = _env_bool(
        "DSFN_CRCF_USE_CERTAINTY", CONFIG["crcf_use_certainty"]
    )
    CONFIG["ema_decay"] = _env_float("DSFN_EMA_DECAY", CONFIG["ema_decay"])
    CONFIG["model_dropout"] = _env_float("DSFN_MODEL_DROPOUT", CONFIG["model_dropout"])
    CONFIG["transformer_depth"] = _env_int(
        "DSFN_TRANSFORMER_DEPTH", CONFIG["transformer_depth"]
    )
    CONFIG["transformer_split_layer"] = _env_int(
        "DSFN_TRANSFORMER_SPLIT_LAYER", CONFIG["transformer_split_layer"]
    )
    CONFIG["use_ema"] = _env_bool("DSFN_USE_EMA", CONFIG["use_ema"])
    CONFIG["use_aug"] = _env_bool("DSFN_USE_AUG", CONFIG["use_aug"])
    CONFIG["use_class_weight"] = _env_bool(
        "DSFN_USE_CLASS_WEIGHT", CONFIG["use_class_weight"]
    )
    CONFIG["eval_force_full"] = _env_bool(
        "DSFN_EVAL_FORCE_FULL", CONFIG["eval_force_full"]
    )
    CONFIG["eval_soft_select"] = _env_bool(
        "DSFN_EVAL_SOFT_SELECT", CONFIG["eval_soft_select"]
    )
    CONFIG["aug_noise_std"] = _env_float("DSFN_AUG_NOISE_STD", CONFIG["aug_noise_std"])
    CONFIG["aug_drift_prob"] = _env_float("DSFN_AUG_DRIFT_PROB", CONFIG["aug_drift_prob"])
    CONFIG["aug_scale_min"] = _env_float("DSFN_AUG_SCALE_MIN", CONFIG["aug_scale_min"])
    CONFIG["aug_scale_max"] = _env_float("DSFN_AUG_SCALE_MAX", CONFIG["aug_scale_max"])
    CONFIG["aug_time_shift_max"] = _env_int(
        "DSFN_AUG_TIME_SHIFT_MAX", CONFIG["aug_time_shift_max"]
    )
    CONFIG["run_final_reports"] = _env_bool(
        "DSFN_RUN_FINAL_REPORTS", CONFIG["run_final_reports"]
    )
    CONFIG["clean_test_only"] = _env_bool(
        "DSFN_CLEAN_TEST_ONLY", CONFIG["clean_test_only"]
    )
    CONFIG["save_dir"] = os.path.join(
        CONFIG["save_root"],
        f"{CONFIG['exp_name']}_seed{CONFIG['seed']}",
    )
    os.makedirs(CONFIG["save_dir"], exist_ok=True)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
############################################
# Dataset
############################################
def apply_drift(raw_signal, channel_indices):
    # raw_signal: [9, 1024]
    # 模拟传感器随着时间缓慢偏移
    for ch in channel_indices:
        drift = np.linspace(0, np.random.uniform(-0.5, 0.5), 1024)
        raw_signal[ch, :] += drift
    return raw_signal

def compute_channel_stats(data, indices):
    means = []
    stds = []
    subset = data[indices]

    for ch in range(subset.shape[1]):
        values = subset[:, ch, :].reshape(-1)
        values = values[np.isfinite(values)]

        if values.size == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(values.mean())
            std = float(values.std())
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0

        means.append(mean)
        stds.append(std)

    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)

def compute_class_weights(labels, indices, num_classes):
    counts = np.bincount(labels[indices].astype(int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.astype(np.float32)

class SensorDataset(Dataset):
    def __init__(self, root, is_train=False, channel_mean=None, channel_std=None):
        self.data = np.load(os.path.join(root, "train_data.npy"))
        self.labels = np.load(os.path.join(root, "train_labels.npy"))
        self.is_train = is_train  # 记录是否为训练模式

        if self.data.shape[1] == 1024:
            self.data = self.data.transpose(0, 2, 1)
        self.data = self.data.astype(np.float32, copy=False)
        self.channel_mean = None
        self.channel_std = None
        if channel_mean is not None:
            self.channel_mean = np.asarray(channel_mean, dtype=np.float32).reshape(-1, 1)
        if channel_std is not None:
            channel_std = np.asarray(channel_std, dtype=np.float32)
            channel_std = np.where(channel_std < 1e-6, 1.0, channel_std)
            self.channel_std = channel_std.reshape(-1, 1)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[idx].copy()
        if self.channel_mean is not None:
            raw = np.where(np.isfinite(raw), raw, self.channel_mean)
        else:
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        # copy 防止修改原始内存

        if self.is_train and CONFIG.get("use_aug", True):
            noise_std = float(CONFIG.get("aug_noise_std", 0.005))
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, raw.shape)
                raw += noise

            if np.random.random() < float(CONFIG.get("aug_drift_prob", 0.5)):
                raw = apply_drift(raw, [6, 7, 8])

            scale_min = float(CONFIG.get("aug_scale_min", 0.98))
            scale_max = float(CONFIG.get("aug_scale_max", 1.02))
            scaling_factor = np.random.uniform(scale_min, scale_max)
            raw *= scaling_factor

            shift_max = int(CONFIG.get("aug_time_shift_max", 0) or 0)
            if shift_max > 0:
                shift = np.random.randint(-shift_max, shift_max + 1)
                if shift != 0:
                    raw = np.roll(raw, shift=shift, axis=-1)

        if CONFIG.get("use_channel_norm", True) and self.channel_mean is not None and self.channel_std is not None:
            raw = (raw - self.channel_mean) / self.channel_std

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        label = torch.tensor(self.labels[idx]).long()
        gyro = torch.tensor(raw[0:3]).float()
        accel = torch.tensor(raw[3:6]).float()
        vel = torch.tensor(raw[6:9]).float()

        return gyro, accel, vel, label

class PatchEmbedding(nn.Module):
    """Non-overlapping patch embedding producing the token sequence (Eq. (7))."""

    def __init__(self, in_ch=3, embed_dim=128, patch=16):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.conv(x)
        return x.permute(0, 2, 1)
############################################
# Temporal CNN (Section 3.1.1)
############################################
class TemporalCNN(nn.Module):
    def __init__(self, in_ch=3, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # 🚀 引入扩张卷积 (dilation=4)，感受野直接扩大 4 倍
            nn.Conv1d(16, out_ch, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
############################################
# Dynamic-depth Transformer encoder (Sections 3.1.2-3.1.3)
############################################
class DynamicDepthTransformerEncoder(nn.Module):
    def __init__(self, dim=128, heads=4, depth=5, split_layer=3, dropout=0.4):
        super().__init__()
        self.split_layer = split_layer
        self.total_depth = depth
        self.num_dynamic = depth - split_layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,dropout=dropout)

        # 1. 基础层（固定计算）
        self.base_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(split_layer)])
        # 2. 动态层（可跳过）
        self.dynamic_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(depth - split_layer)])

        # 3. Dynamic-layer selector (Eq. (10))
        self.selector = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.dynamic_layers))
        )

    def run_base(self, x, start_layer=0, end_layer=None):
        """Run a selectable slice of the fixed base layers."""
        if end_layer is None:
            end_layer = len(self.base_layers)
        start_layer = max(0, min(int(start_layer), len(self.base_layers)))
        end_layer = max(start_layer, min(int(end_layer), len(self.base_layers)))
        for layer in self.base_layers[start_layer:end_layer]:
            x = layer(x)
        return x

    def forward_dynamic(self, x, force_full=False, collect_intermediates=True):
        """
        Dynamic-depth path (Eq. (8)-(10)).

        Training always executes every candidate dynamic layer and merges them
        with the selector probabilities. Evaluation uses the same soft-merge
        path when ``force_full`` is set, which is the protocol reported in the
        manuscript, and otherwise follows the most likely exit depth.
        """
        base_feat = x
        num_dynamic = len(self.dynamic_layers)

        if not CONFIG.get("use_dynamic_depth", True):
            layer_outs = []
            curr_x = x
            for layer in self.dynamic_layers:
                curr_x = layer(curr_x)
                if collect_intermediates:
                    layer_outs.append(curr_x)
            return curr_x, None, base_feat, layer_outs

        cls_token_base = x.mean(dim=1)
        select_logits = self.selector(cls_token_base)
        select_probs = F.softmax(select_logits, dim=-1)

        layer_outs = []
        curr_x = x

        if self.training or force_full:
            for layer in self.dynamic_layers:
                curr_x = layer(curr_x)
                layer_outs.append(curr_x)

            if self.training or CONFIG.get("eval_soft_select", True):
                stacked_outs = torch.stack(layer_outs, dim=1)
                out = torch.sum(select_probs.view(-1, num_dynamic, 1, 1) * stacked_outs, dim=1)
            else:
                out = curr_x

            return out, select_probs, base_feat, layer_outs if collect_intermediates else []

        best_idx = torch.argmax(select_logits, dim=-1)
        max_idx = int(best_idx.max().item())

        for i in range(max_idx + 1):
            curr_x = self.dynamic_layers[i](curr_x)
            layer_outs.append(curr_x)

        if x.shape[0] == 1:
            out = layer_outs[int(best_idx.item())]
        else:
            stacked_outs = torch.stack(layer_outs, dim=1)
            out = stacked_outs[torch.arange(x.shape[0], device=x.device), best_idx]

        return out, select_probs, base_feat, layer_outs if collect_intermediates else []

    def forward(self, x):
        """标准 PyTorch 入口，按序运行 Base 和 Dynamic"""
        x = self.run_base(x)
        return self.forward_dynamic(x)


############################################
# Dynamic fusion
############################################
class DynamicFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, g, a, v):
        g_cls = g.mean(1)
        a_cls = a.mean(1)
        v_cls = v.mean(1)

        feat = torch.cat([g_cls, a_cls, v_cls], dim=1)
        w = self.net(feat)

        fused = (
            w[:, 0:1] * g_cls +
            w[:, 1:2] * a_cls +
            w[:, 2:3] * v_cls
        )

        return fused, w
class CRCFFusion(nn.Module):
    """
    Cross-Modal Similarity-Guided Reliability-Consistency Fusion (CRCF)

    融合依据：
      1) 单模态分类置信度 confidence；
      2) 模态间一致性 consistency；
      3) dynamic-path stability from the dynamic-layer selector.

    该模块用于替换最终 AvgFusion，不改变前端三分支编码和动态路径逻辑。
    """
    def __init__(self, dim=128, num_classes=6, hidden=64, temperature=1.0):
        super().__init__()
        self.branch_head_g = nn.Linear(dim, num_classes)
        self.branch_head_a = nn.Linear(dim, num_classes)
        self.branch_head_v = nn.Linear(dim, num_classes)

        self.score_mlp = nn.Sequential(
            nn.Linear(dim + 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.temperature = temperature

    @staticmethod
    def selector_certainty(probs):
        if probs is None:
            return None
        eps = 1e-8
        k = probs.size(1)
        if k <= 1:
            return torch.ones(probs.size(0), 1, device=probs.device, dtype=probs.dtype)
        entropy = -(probs * (probs + eps).log()).sum(dim=1, keepdim=True)
        certainty = 1.0 - entropy / math.log(k)
        return certainty.clamp(0.0, 1.0)

    def forward(self, g, a, v, probs_tuple=None):
        g_cls = g.mean(dim=1)
        a_cls = a.mean(dim=1)
        v_cls = v.mean(dim=1)

        logit_g = self.branch_head_g(g_cls)
        logit_a = self.branch_head_a(a_cls)
        logit_v = self.branch_head_v(v_cls)

        conf_g = F.softmax(logit_g, dim=1).max(dim=1, keepdim=True)[0]
        conf_a = F.softmax(logit_a, dim=1).max(dim=1, keepdim=True)[0]
        conf_v = F.softmax(logit_v, dim=1).max(dim=1, keepdim=True)[0]

        sim_ga = F.cosine_similarity(g_cls, a_cls, dim=1, eps=1e-8).unsqueeze(1)
        sim_gv = F.cosine_similarity(g_cls, v_cls, dim=1, eps=1e-8).unsqueeze(1)
        sim_av = F.cosine_similarity(a_cls, v_cls, dim=1, eps=1e-8).unsqueeze(1)

        # [-1, 1] -> [0, 1]，避免负相似度直接主导 MLP 输入尺度
        cons_g = 0.5 * (0.5 * (sim_ga + 1.0) + 0.5 * (sim_gv + 1.0))
        cons_a = 0.5 * (0.5 * (sim_ga + 1.0) + 0.5 * (sim_av + 1.0))
        cons_v = 0.5 * (0.5 * (sim_gv + 1.0) + 0.5 * (sim_av + 1.0))

        if probs_tuple is not None:
            cert_g = self.selector_certainty(probs_tuple[0])
            cert_a = self.selector_certainty(probs_tuple[1])
            cert_v = self.selector_certainty(probs_tuple[2])
        else:
            cert_g = cert_a = cert_v = None

        if cert_g is None:
            cert_g = torch.ones_like(conf_g)
        if cert_a is None:
            cert_a = torch.ones_like(conf_a)
        if cert_v is None:
            cert_v = torch.ones_like(conf_v)

        if not CONFIG.get("crcf_use_reliability", True):
            conf_g = torch.ones_like(conf_g)
            conf_a = torch.ones_like(conf_a)
            conf_v = torch.ones_like(conf_v)
        if not CONFIG.get("crcf_use_consistency", True):
            cons_g = torch.ones_like(cons_g)
            cons_a = torch.ones_like(cons_a)
            cons_v = torch.ones_like(cons_v)
        if not CONFIG.get("crcf_use_certainty", True):
            cert_g = torch.ones_like(cert_g)
            cert_a = torch.ones_like(cert_a)
            cert_v = torch.ones_like(cert_v)

        q_g = torch.cat([conf_g, cons_g, cert_g], dim=1)
        q_a = torch.cat([conf_a, cons_a, cert_a], dim=1)
        q_v = torch.cat([conf_v, cons_v, cert_v], dim=1)

        score_g = self.score_mlp(torch.cat([g_cls, q_g], dim=1))
        score_a = self.score_mlp(torch.cat([a_cls, q_a], dim=1))
        score_v = self.score_mlp(torch.cat([v_cls, q_v], dim=1))
        scores = torch.cat([score_g, score_a, score_v], dim=1)
        alpha = F.softmax(scores / self.temperature, dim=1)

        fused = (
            alpha[:, 0:1] * g_cls +
            alpha[:, 1:2] * a_cls +
            alpha[:, 2:3] * v_cls
        )

        branch_logits = {"g": logit_g, "a": logit_a, "v": logit_v}
        crcf_info = {
            "confidence": torch.cat([conf_g, conf_a, conf_v], dim=1),
            "consistency": torch.cat([cons_g, cons_a, cons_v], dim=1),
            "certainty": torch.cat([cert_g, cert_a, cert_v], dim=1),
            "fusion_weights": alpha,
        }
        return fused, alpha, branch_logits, crcf_info


class TokenDenoiseBlock(nn.Module):
    """
    融合前特征去噪模块。
    输入:  [B, T, C]
    输出:  [B, T, C]
    作用:  在 token 维度上做轻量深度卷积平滑，同时用残差保留原始故障特征。
    """
    def __init__(self, dim, kernel_size=5, reduction=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False
        )
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(dim)

        hidden = max(dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        residual = x

        z = self.norm(x)
        z_conv = z.transpose(1, 2)          # [B, C, T]
        z_conv = self.dwconv(z_conv)
        z_conv = self.pwconv(z_conv)
        z_conv = self.bn(z_conv)
        z_conv = z_conv.transpose(1, 2)     # [B, T, C]

        # 样本级通道门控，控制去噪强度
        gate = self.gate(z.mean(dim=1)).unsqueeze(1)  # [B, 1, C]

        out = residual + self.dropout(gate * z_conv)
        return out


class SignalTransformerModel(nn.Module):
    def get_gate_weight(self, g, a, v):
        g_cls = g.mean(1)
        B = g_cls.size(0)

        if CONFIG.get("use_gate_dynamic_weight", True):
            _, w = self.gate_fusion(g, a, v)
        else:
            w = torch.ones(B, 3, device=g_cls.device) / 3.0

        return w

    def final_fuse(self, g, a, v, probs_tuple=None):
        g_cls = g.mean(1)
        a_cls = a.mean(1)
        v_cls = v.mean(1)

        if CONFIG.get("use_crcf", False):
            fused, w_final, branch_logits, crcf_info = self.crcf(
                g, a, v, probs_tuple=probs_tuple
            )
            aux = {
                "branch_logits": branch_logits,
                "fusion_weights": w_final,
                "crcf_info": crcf_info,
            }
        else:
            fused = (g_cls + a_cls + v_cls) / 3.0
            B = g_cls.size(0)
            w_final = torch.ones(B, 3, device=g_cls.device) / 3.0
            aux = None

        return fused, w_final, aux

    def apply_prefusion_denoise(self, g_c, a_c, v_c):
        if CONFIG.get("use_prefusion_denoise", False):
            g_c = self.denoise_g(g_c)
            a_c = self.denoise_a(a_c)
            v_c = self.denoise_v(v_c)
        return g_c, a_c, v_c
    def __init__(self, dim=128, num_classes=6, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = float(CONFIG.get("model_dropout", 0.2))
        transformer_depth = int(CONFIG.get("transformer_depth", 5))
        transformer_split_layer = int(CONFIG.get("transformer_split_layer", 3))
        if not 0 <= transformer_split_layer < transformer_depth:
            raise ValueError(
                "transformer_split_layer must satisfy "
                f"0 <= split < depth, got split={transformer_split_layer}, "
                f"depth={transformer_depth}."
            )
        # Modality-specific CNN-Transformer encoders (Section 3.1)
        self.encoder_g = DynamicDepthTransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )
        self.encoder_a = DynamicDepthTransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )
        self.encoder_v = DynamicDepthTransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )

        # Temporal CNN maps each 3-channel modality to 32 channels (Section 3.1.1)
        self.pre_g = TemporalCNN(3, 32)
        self.pre_a = TemporalCNN(3, 32)
        self.pre_v = TemporalCNN(3, 32)

        # Patch embedding: 16 for Gyro/Accel, 32 for Vel (Section 3.1.2)
        self.embed_g = PatchEmbedding(32, dim, patch=16)
        self.embed_a = PatchEmbedding(32, dim, patch=16)
        self.embed_v = PatchEmbedding(32, dim, patch=32)

        # Gated feature-level denoising (GTD, Section 3.2)
        if CONFIG.get("use_prefusion_denoise", False):
            self.denoise_g = TokenDenoiseBlock(dim)
            self.denoise_a = TokenDenoiseBlock(dim)
            self.denoise_v = TokenDenoiseBlock(dim)
        else:
            self.denoise_g = nn.Identity()
            self.denoise_a = nn.Identity()
            self.denoise_v = nn.Identity()

        # Sample-wise modal gate weights for the dynamic-depth selector
        self.gate_fusion = DynamicFusion(dim)

        # Cross-modal reliability-consistency fusion (CRCF, Section 3.3)
        self.crcf = CRCFFusion(
            dim=dim,
            num_classes=num_classes,
            hidden=64,
            temperature=1.0
        ) if CONFIG.get("use_crcf", False) else None

        # Classification head
        self.simple_head = nn.Linear(dim, num_classes)

    def stem_to_tokens(self, g, a, v):
        g, a, v = self.pre_g(g), self.pre_a(a), self.pre_v(v)
        g, a, v = self.embed_g(g), self.embed_a(a), self.embed_v(v)
        return g, a, v

    def run_base_triplet(self, g, a, v, start_layer=0, end_layer=None):
        g_base = self.encoder_g.run_base(g, start_layer=start_layer, end_layer=end_layer)
        a_base = self.encoder_a.run_base(a, start_layer=start_layer, end_layer=end_layer)
        v_base = self.encoder_v.run_base(v, start_layer=start_layer, end_layer=end_layer)
        return g_base, a_base, v_base

    def preprocess_to_base(self, g, a, v):
        g, a, v = self.stem_to_tokens(g, a, v)
        return self.run_base_triplet(g, a, v)

    def extract_fused_feature_from_base(
        self,
        g_base,
        a_base,
        v_base,
        gate_threshold=0.25,
        force_full=False,
        collect_intermediates=True,
    ):
        w = self.get_gate_weight(g_base, a_base, v_base)

        if CONFIG.get("use_outer_gate", True):
            gate_w = w
        else:
            gate_w = torch.ones_like(w)

        g_out, pg, bg, og = self._adaptive_step_eval_per_sample(
            self.encoder_g, g_base, gate_w[:, 0], gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )
        a_out, pa, ba, oa = self._adaptive_step_eval_per_sample(
            self.encoder_a, a_base, gate_w[:, 1], gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )
        v_out, pv, bv, ov = self._adaptive_step_eval_per_sample(
            self.encoder_v, v_base, gate_w[:, 2], gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )

        g_c, a_c, v_c = g_out, a_out, v_out
        g_c, a_c, v_c = self.apply_prefusion_denoise(g_c, a_c, v_c)

        probs_tuple = (pg, pa, pv)
        fused, final_w, fusion_aux = self.final_fuse(g_c, a_c, v_c, probs_tuple=probs_tuple)
        return fused, w, probs_tuple, (bg, ba, bv), (og, oa, ov), fusion_aux, final_w

    def extract_fused_feature(
        self,
        g,
        a,
        v,
        gate_threshold=0.25,
        force_full=False,
        collect_intermediates=True,
    ):
        g_base, a_base, v_base = self.preprocess_to_base(g, a, v)
        return self.extract_fused_feature_from_base(
            g_base, a_base, v_base,
            gate_threshold=gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )
    def forward_features(
        self,
        g,
        a,
        v,
        gate_threshold=0.25,
        force_full=False,
        collect_intermediates=True,
    ):
        fused, _, _, _, _, _, _ = self.extract_fused_feature(
            g, a, v,
            gate_threshold=gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )
        return fused

    def classify_fused(self, fused):
        return self.simple_head(fused)

    def _adaptive_step_eval_per_sample(
        self,
        encoder,
        base_feat,
        weight_vec,
        threshold,
        force_full=False,
        collect_intermediates=True,
    ):
        """
        Per-sample outer-gate decision for evaluation (Eq. (9)).

        Returns the modality feature, the selector probabilities (None when the
        modality bypasses the dynamic layers), the base-layer feature and the
        intermediate dynamic-layer outputs.
        """
        if self.training or force_full:
            return encoder.forward_dynamic(
                base_feat,
                force_full=force_full,
                collect_intermediates=collect_intermediates,
            )

        # weight_vec: [B]
        if weight_vec.dim() != 1:
            weight_vec = weight_vec.view(-1)

        B = base_feat.size(0)
        active_mask = weight_vec >= threshold   # True 表示该样本继续进入动态层

        # Every sample enters the dynamic layers.
        if active_mask.all():
            return encoder.forward_dynamic(
                base_feat,
                force_full=False,
                collect_intermediates=collect_intermediates,
            )

        # Every sample bypasses the dynamic layers and keeps the base features.
        if (~active_mask).all():
            return base_feat, None, base_feat, []

        # Mixed batch: fill in both groups.
        out = base_feat.clone()
        num_dynamic = len(encoder.dynamic_layers)
        probs_full = base_feat.new_zeros((B, num_dynamic))

        active_feat = base_feat[active_mask]  # [B_active, T, C]
        out_active, probs_active, _, outs_active = encoder.forward_dynamic(
            active_feat,
            force_full=False,
            collect_intermediates=collect_intermediates,
        )

        out[active_mask] = out_active

        # When the selector is disabled, probs_active is None.
        if probs_active is not None:
            probs_full[active_mask] = probs_active
        else:
            probs_full = None

        if not collect_intermediates:
            return out, probs_full, base_feat, []

        # Scatter the per-layer outputs back onto the full batch.
        full_outs = []
        for layer_out in outs_active:
            buf = base_feat.clone()
            buf[active_mask] = layer_out
            full_outs.append(buf)

        return out, probs_full, base_feat, full_outs

    def forward(
        self,
        g,
        a,
        v,
        gate_threshold=0.25,
        force_full=False,
        collect_intermediates=True,
    ):
        """
        Run the complete DSFN model.

        Returns the fused logits, the auxiliary fusion dictionary (or None),
        the modal gate weights, the dynamic-layer selection probabilities, the
        base-layer features and the intermediate dynamic-layer outputs.
        """
        (
            fused,
            gate_w,
            probs_tuple,
            bases_tuple,
            outs_tuple,
            fusion_aux,
            _final_w,
        ) = self.extract_fused_feature(
            g, a, v,
            gate_threshold=gate_threshold,
            force_full=force_full,
            collect_intermediates=collect_intermediates,
        )

        logits = self.classify_fused(fused)
        aux = dict(fusion_aux) if isinstance(fusion_aux, dict) else None

        # gate_w is returned as the third value so that the dynamic-layer usage
        # statistics stay unchanged; CRCF fusion weights live in
        # aux["fusion_weights"].
        return logits, aux, gate_w, probs_tuple, bases_tuple, outs_tuple


class HybridLoss(nn.Module):
    """Total training objective, Eq. (28)-(30)."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("main_class_weights", torch.ones(num_classes))

    def similarity_guided_loss(self, probs, base_feat, layer_outs):
        """Similarity-guided loss of Eqs. (11)-(15)."""
        if probs is None or len(layer_outs) < 1:
            return torch.tensor(0.0, device=base_feat.device)

        # [base, dynamic_1, ..., dynamic_K]
        all_layers = [base_feat] + list(layer_outs)
        adj_sims = torch.stack(
            [
                F.cosine_similarity(
                    all_layers[i - 1].mean(dim=1), all_layers[i].mean(dim=1), dim=1
                )
                for i in range(1, len(all_layers))
            ],
            dim=1,
        )

        # Eq. (12): similarity-derived target layer-selection distribution.
        target_probs = F.softmax(adj_sims / SIMILARITY_SCALE, dim=1)
        kl_loss = F.kl_div(
            probs.clamp(min=1e-7).log(), target_probs, reduction="batchmean"
        )

        # Eq. (14): similarity-stability constraint (lambda_s in Eq. (15)).
        sat_loss = F.mse_loss(adj_sims, torch.ones_like(adj_sims))

        return kl_loss + SIMILARITY_STABILITY_WEIGHT * sat_loss

    def forward(self, main_logits, aux_outputs, targets, probs_tuple, bases_tuple, outs_tuple):
        weight = self.main_class_weights if CONFIG.get("use_class_weight", True) else None
        label_smoothing = float(CONFIG.get("label_smoothing", 0.0))

        # 1. Main classification loss
        loss_main = F.cross_entropy(
            main_logits,
            targets,
            weight=weight,
            label_smoothing=label_smoothing,
        )

        # 2. Similarity-guided loss, averaged over the three modality branches
        loss_sim = torch.tensor(0.0, device=targets.device)
        if (
            CONFIG.get("use_dynamic_depth", True)
            and CONFIG.get("use_similarity_loss", True)
            and probs_tuple is not None
        ):
            for probs, base_feat, layer_outs in zip(probs_tuple, bases_tuple, outs_tuple):
                if probs is None or not layer_outs:
                    continue
                if probs.shape[1] == len(layer_outs):
                    loss_sim += self.similarity_guided_loss(probs, base_feat, layer_outs)

        # 3. Unimodal auxiliary classification loss and fusion-weight balancing
        loss_aux = torch.tensor(0.0, device=targets.device)
        loss_balance = torch.tensor(0.0, device=targets.device)
        if CONFIG.get("use_crcf", False) and aux_outputs is not None:
            branch_logits = aux_outputs.get("branch_logits")
            if branch_logits is not None:
                loss_aux = (
                    sum(
                        F.cross_entropy(
                            branch_logits[key],
                            targets,
                            weight=weight,
                            label_smoothing=label_smoothing,
                        )
                        for key in ("g", "a", "v")
                    )
                    / 3.0
                )
            fusion_weights = aux_outputs.get("fusion_weights")
            if fusion_weights is not None:
                mean_alpha = fusion_weights.mean(dim=0)
                target_alpha = torch.ones_like(mean_alpha) / mean_alpha.numel()
                loss_balance = F.mse_loss(mean_alpha, target_alpha)

        return (
            loss_main
            + SIMILARITY_LOSS_WEIGHT * (loss_sim / 3.0)
            + float(CONFIG.get("crcf_aux_weight", 0.05)) * loss_aux
            + float(CONFIG.get("crcf_balance_weight", 0.01)) * loss_balance
        )


def plot_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve');
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)')
    plt.legend();
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    print(f"📊 曲线图已保存至 {save_dir}")


def eval_forward_kwargs():
    if CONFIG.get("eval_force_full", True):
        return {"gate_threshold": -1.0, "force_full": True}
    return {
        "gate_threshold": CONFIG.get("gate_threshold", 0.25),
        "force_full": False,
    }


def eval_forward(model, g, a, v):
    return model(g, a, v, **eval_forward_kwargs())


def create_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay=0.995):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def load_state_dict_flexible(model, checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    # Checkpoints trained before the fusion module was renamed to CRCF store it
    # as "rcd_fusion.*"; accept both spellings so those runs stay loadable.
    checkpoint = {
        ("crcf." + key.split(".", 1)[1]) if key.startswith("rcd_fusion.") else key: value
        for key, value in checkpoint.items()
    }
    if any(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in checkpoint.items()
        }

    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    unexpected_keys = []

    for key, value in checkpoint.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible_state[key] = value

    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    model_state.update(compatible_state)
    model.load_state_dict(model_state, strict=True)
    return missing_keys, unexpected_keys, skipped_keys


def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for g, a, v, y in loader:
            g, a, v, y = g.to(device), a.to(device), v.to(device), y.to(device)
            # 这里的返回值必须和 forward 一一对应
            logits, aux, w, pt, bt, ot = eval_forward(model, g, a, v)
            loss = criterion(logits, aux, y, pt, bt, ot)

            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), 100 * correct / total


def visualize_tsne(model, dataloader, device, save_dir):
    print("\n🔍 开始提取特征绘制 t-SNE ...")
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)
            feat = model.forward_features(
                g, a, v,
                **eval_forward_kwargs()
            )
            all_feats.append(feat.cpu().numpy())
            all_labels.append(y.numpy())

    feats_2d = TSNE(
    n_components=2,
    perplexity=30,
    random_state=CONFIG["seed"],
    init="pca",
    learning_rate="auto"
).fit_transform(np.concatenate(all_feats))
    labels = np.concatenate(all_labels)

    plt.figure(figsize=(10, 8))
    classes = ['Normal', 'Accel', 'GPS', 'Gyro', 'Mag', 'Baro']
    sc = plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=sc.legend_elements()[0], labels=classes)
    plt.title('t-SNE')
    plt.savefig(os.path.join(save_dir, 'tsne_final.png'))
    print(f"✅ t-SNE 已保存！")

def plot_saturation_evolution(model, loader, device, save_dir):
    model.eval()
    g, a, v, y = next(iter(loader))
    g, a, v = g.to(device), a.to(device), v.to(device)

    with torch.no_grad():
        # 强制跑满所有层
        logits, aux, w, probs, bases, outs = model(g, a, v, gate_threshold=-1.0, force_full=True)

        def get_adjacent_sims(base, dynamic_outs):
            if not dynamic_outs: return []
            # 🚀 关键修改：对比相邻层
            all_steps = [base] + dynamic_outs
            sims = []
            for i in range(1, len(all_steps)):
                prev_vec = all_steps[i - 1].mean(dim=1)
                curr_vec = all_steps[i].mean(dim=1)
                s = F.cosine_similarity(prev_vec, curr_vec, dim=1).mean().item()
                sims.append(s)
            return sims

        sim_g = get_adjacent_sims(bases[0], outs[0])
        sim_a = get_adjacent_sims(bases[1], outs[1])
        sim_v = get_adjacent_sims(bases[2], outs[2])

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(sim_g) + 1), sim_g, 'o-', label='Gyro Branch (Adjacent)')
        plt.plot(range(1, len(sim_a) + 1), sim_a, 's-', label='Accel Branch (Adjacent)')
        plt.plot(range(1, len(sim_v) + 1), sim_v, '^-', label='Vel Branch (Adjacent)')



        plt.title('Adjacent Layer Feature Saturation', fontsize=14)
        plt.xlabel('Dynamic Layer Index', fontsize=12)
        plt.ylabel('Cosine Similarity (Layer N vs Layer N-1)', fontsize=12)
        plt.ylim(0, 1.05)  # 相似度范围 0-1
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()

        plt.savefig(os.path.join(save_dir, "saturation_evolution_adjacent.png"))
        plt.close()

def plot_confusion_matrix(model, dataloader, device, save_dir):
    print("\n🔍 开始计算并绘制混淆矩阵...")
    model.eval()
    all_preds =[]
    all_labels =[]

    with torch.no_grad():
        # 🔥 修复：你的 dataloader 返回的是 g, a, v, y
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)


            # 🔥 修复：传入三个模态，接收所有返回值
            logits, _, _, _, _, _ = eval_forward(model, g, a, v)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_normalized = cm.astype('float') / np.maximum(row_sum, 1)

    plt.figure(figsize=(10, 8))
    class_names =['Normal', 'Accel', 'GPS', 'Gyro', 'Mag', 'Baro']

    import seaborn as sns # 确保导入了 sns
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})

    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 混淆矩阵图已成功保存至: {save_path}")
############################################
# FPS 测速
############################################
import time
import numpy as np
import torch

def test_fps(model, device, val_loader=None, repeat=5, warmup=100, iters=500,
             gate_threshold=0.25, force_full=False):
    model.eval()
    torch.backends.cudnn.benchmark = True

    bench_batches = []
    fps_batch_size = max(1, int(CONFIG.get("fps_batch_size", 1)))
    fps_num_batches = max(1, int(CONFIG.get("fps_num_batches", 16)))

    # Average dynamic early-exit latency over several fixed samples instead of
    # timing one validation sample whose exit decision may be unrepresentative.
    if val_loader is not None:
        for batch_idx, (g, a, v, _) in enumerate(val_loader):
            if batch_idx >= fps_num_batches:
                break
            bench_batches.append((
                g[:fps_batch_size].to(device),
                a[:fps_batch_size].to(device),
                v[:fps_batch_size].to(device),
            ))
    else:
        bench_batches.append((
            torch.randn(fps_batch_size, 3, 1024, device=device),
            torch.randn(fps_batch_size, 3, 1024, device=device),
            torch.randn(fps_batch_size, 3, 1024, device=device),
        ))

    with torch.inference_mode():
        for i in range(warmup):
            g, a, v = bench_batches[i % len(bench_batches)]
            _ = model(
                g, a, v,
                gate_threshold=gate_threshold,
                force_full=force_full,
                collect_intermediates=False,
            )

    timings = []

    if "cuda" in str(device):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.inference_mode():
            for _ in range(repeat):
                processed_samples = 0
                torch.cuda.synchronize()
                starter.record()
                for i in range(iters):
                    g, a, v = bench_batches[i % len(bench_batches)]
                    _ = model(
                        g, a, v,
                        gate_threshold=gate_threshold,
                        force_full=force_full,
                        collect_intermediates=False,
                    )
                    processed_samples += int(g.size(0))
                ender.record()
                torch.cuda.synchronize()
                elapsed_s = starter.elapsed_time(ender) / 1000.0
                timings.append((elapsed_s, processed_samples))
    else:
        with torch.inference_mode():
            for _ in range(repeat):
                processed_samples = 0
                start = time.perf_counter()
                for i in range(iters):
                    g, a, v = bench_batches[i % len(bench_batches)]
                    _ = model(
                        g, a, v,
                        gate_threshold=gate_threshold,
                        force_full=force_full,
                        collect_intermediates=False,
                    )
                    processed_samples += int(g.size(0))
                end = time.perf_counter()
                timings.append((end - start, processed_samples))

    fps_list = [samples / t for t, samples in timings]
    latency_ms_list = [(t / samples) * 1000 for t, samples in timings]

    print("\n" + "=" * 50)
    print(f"gate_threshold = {gate_threshold}, force_full = {force_full}")
    print(f"FPS: {np.mean(fps_list):.2f} ± {np.std(fps_list):.2f} samples/s")
    print(f"Latency: {np.mean(latency_ms_list):.2f} ± {np.std(latency_ms_list):.2f} ms/sample")
    print("=" * 50)

    return {
        "fps_mean": float(np.mean(fps_list)),
        "fps_std": float(np.std(fps_list)),
        "latency_mean_ms": float(np.mean(latency_ms_list)),
        "latency_std_ms": float(np.std(latency_ms_list)),
        "benchmark_samples_per_repeat": int(timings[0][1]) if timings else 0,
    }


def analyze_dynamic_usage(model, dataloader, device, save_dir, threshold=0.25):
    model.eval()

    branch_names = ["Gyro", "Accel", "Vel"]
    total_samples = 0

    active_counts = np.zeros(3)
    selected_layer_sum = np.zeros(3)

    with torch.no_grad():
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)
            logits, aux, w, probs_tuple, bases_tuple, outs_tuple = model(
                g, a, v, gate_threshold=threshold
            )

            B = g.size(0)
            total_samples += B

            if CONFIG.get("use_outer_gate", True):
                gate_w_np = w.cpu().numpy()
            else:
                gate_w_np = np.ones_like(w.cpu().numpy())

            for b in range(3):
                active = gate_w_np[:, b] >= threshold
                active_counts[b] += active.sum()

                probs = probs_tuple[b]
                if probs is not None:
                    probs_np = probs.cpu().numpy()
                    selected = probs_np.argmax(axis=1) + 1
                    selected_layer_sum[b] += (selected * active).sum()

    rows = []
    for i, name in enumerate(branch_names):
        active_ratio = active_counts[i] / total_samples
        avg_layer = selected_layer_sum[i] / max(active_counts[i], 1)

        rows.append({
            "branch": name,
            "active_ratio": active_ratio,
            "avg_selected_dynamic_layer": avg_layer,
        })

    df = pd.DataFrame(rows)
    save_path = os.path.join(save_dir, "dynamic_usage.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n📊 Dynamic Usage:")
    print(df)
    print(f"动态层使用统计已保存至: {save_path}")

    return df
def visualize_modal_weights(model, dataloader, device, save_dir):
    """
    可视化不同故障类别下，模型对三个模态（Gyro, Accel, Vel）分配的平均权重
    """
    model.eval()
    # 类别名称
    class_names = ['Normal', 'Accel', 'GPS', 'Gyro', 'Mag', 'Baro']
    # 存储每个类别的权重累加值
    class_weights = {i: [] for i in range(len(class_names))}

    with torch.no_grad():
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)
            # 优先可视化 CRCF 的最终融合权重；若不是 CRCF，则退回 outer gate 权重。
            _, aux, w, _, _, _ = eval_forward(model, g, a, v)
            if aux is not None and isinstance(aux, dict) and "fusion_weights" in aux:
                w_plot = aux["fusion_weights"]
            else:
                w_plot = w

            w_np = w_plot.cpu().numpy()  # [Batch, 3]
            y_np = y.numpy()

            for i in range(len(y_np)):
                label = y_np[i]
                class_weights[label].append(w_np[i])

    # 计算每个类别的平均权重
    avg_weights = []
    for i in range(len(class_names)):
        if len(class_weights[i]) > 0:
            avg_weights.append(np.mean(class_weights[i], axis=0))
        else:
            avg_weights.append([0.33, 0.33, 0.33])  # 默认值

    avg_weights = np.array(avg_weights)  # [6, 3]

    # 绘图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25

    plt.bar(x - width, avg_weights[:, 0], width, label='Gyro Weight', color='#1f77b4')
    plt.bar(x, avg_weights[:, 1], width, label='Accel Weight', color='#ff7f0e')
    plt.bar(x + width, avg_weights[:, 2], width, label='Vel Weight', color='#2ca02c')

    plt.xlabel('Fault Types')
    plt.ylabel('Attention Weight')
    plt.title('Modal Attention Distribution per Fault Type')
    plt.xticks(x, class_names)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(save_dir, 'modal_weights.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 模态权重可视化已保存至: {save_path}")



def evaluate_metrics(model, dataloader, device, save_dir):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)
            logits, _, _, _, _, _ = eval_forward(model, g, a, v)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    result = {
        "exp_name": CONFIG["exp_name"],
        "seed": CONFIG["seed"],
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
    }

    df = pd.DataFrame([result])
    df.to_csv(os.path.join(save_dir, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    print("\n📊 Metrics Summary:")
    print(df)

    return result
def evaluate_detailed_performance(model, dataloader, device, save_dir):
    """
    计算并保存每一类故障的详细分类指标
    """
    print("\n🔍 正在进行全量类别性能评估...")
    model.eval()
    all_preds = []
    all_labels = []
    class_names = ['Normal', 'Accel', 'GPS', 'Gyro', 'Mag', 'Baro']

    with torch.no_grad():
        for g, a, v, y in dataloader:
            g, a, v = g.to(device), a.to(device), v.to(device)
            # 确保这里的解包数量（6个）与模型 forward 返回量一致
            logits, _, _, _, _, _ = eval_forward(model, g, a, v)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    # 1. 生成详细报告 (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n📊 详细分类报告 (Per-class Performance):")
    print(report)

    # 2. 计算每一类的准确率 (Per-class Accuracy)
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # 3. 将结果保存到文件
    report_path = os.path.join(save_dir, 'detailed_performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Detailed Classification Report ===\n")
        f.write(report)
        f.write("\n=== Per-Class Accuracy ===\n")
        for name, acc in zip(class_names, per_class_acc):
            line = f"{name}: {acc * 100:.2f}%\n"
            f.write(line)
            print(f"✅ {name} 准确率: {acc * 100:.2f}%")

    print(f"📊 详细分类报告已自动保存至: {report_path}")


try:
    from thop import profile
except ImportError:
    profile = None
import torch


def report_model_complexity(model, device):
    """
    全面评估模型的复杂度：参数量、FLOPs（计算量）、模型体积、内存占用
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)

    if profile is None:
        print("\n" + "=" * 50)
        print("📋 模型复杂度报告")
        print("-" * 50)
        print(f"📦 完整参数量 (Total Parameters): {total_params / 1e6:.3f} M")
        print(f"🛠️ 可训练参数量 (Trainable Parameters): {trainable_params / 1e6:.3f} M")
        print(f"💾 模型文件大小估计 (Float32): {model_size_mb:.2f} MB")
        print("⚠️ 当前环境未安装 thop，跳过旧 THOP FLOPs/MACs 统计。论文 FLOPs 请使用统一复核脚本。")
        print("=" * 50 + "\n")
        return total_params, None

    was_training = model.training
    model.eval()

    class FullForwardWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, g, a, v):
            logits, _, _, _, _, _ = self.base_model(
                g,
                a,
                v,
                gate_threshold=-1.0,
                force_full=True,
                collect_intermediates=False,
            )
            return logits

    # 模拟输入：三路信号 (Gyro, Accel, Vel)，每路 3通道, 长度 1024
    dummy_g = torch.randn(1, 3, 1024).to(device)
    dummy_a = torch.randn(1, 3, 1024).to(device)
    dummy_v = torch.randn(1, 3, 1024).to(device)

    # 1. 使用 thop 计算 FLOPs 和 Params
    # 注意：因为你的 forward 返回值比较复杂，我们需要包装一下
    macs, _ = profile(FullForwardWrapper(model), inputs=(dummy_g, dummy_a, dummy_v), verbose=False)
    if was_training:
        model.train()

    print("\n" + "=" * 50)
    print("📋 模型边缘计算适配性评估报告")
    print("-" * 50)
    print(f"📦 完整参数量 (Total Parameters): {total_params / 1e6:.3f} M")
    print(f"🛠️ 可训练参数量 (Trainable Parameters): {trainable_params / 1e6:.3f} M")
    print(f"🧮 理论计算量 (FLOPs/MACs): {macs / 1e6:.2f} M (每样本)")
    print(f"💾 模型文件大小 (Float32): {model_size_mb:.2f} MB")

    # 3. Dynamic-depth analysis (base layers versus the full encoder)
    print("-" * 50)
    print("Dynamic-depth computation profile:")
    print(f"   - 满载模式 (6层): {macs / 1e6:.2f} M FLOPs")
    # 粗略估算：Encoder 部分占据了约 80% 的计算量，跳过一半动态层约节省 30-40% 总计算量
    print(f"   - 节能模式 (3层): 约 {(macs / 1e6) * 0.6:.2f} M FLOPs (节省 ~40%)")
    print("=" * 50 + "\n")

    return total_params, macs
############################################
# 主流程
############################################
def main():
    if "DSFN_EXP_NAME" in os.environ:
        CONFIG["exp_name"] = os.environ["DSFN_EXP_NAME"]
    apply_ablation_config(CONFIG["exp_name"])
    apply_runtime_overrides()
    set_seed(CONFIG["seed"])
    print(f"Device: {device}")
    print(
        "Runtime config: "
        f"exp_name={CONFIG['exp_name']}, "
        f"epochs={CONFIG['epochs']}, batch_size={CONFIG['batch_size']}, "
        f"lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']}, "
        f"dropout={CONFIG['model_dropout']}, "
        f"transformer_depth={CONFIG['transformer_depth']}, "
        f"transformer_split_layer={CONFIG['transformer_split_layer']}, "
        f"gate_threshold={CONFIG['gate_threshold']}, "
        f"use_dynamic_depth={CONFIG['use_dynamic_depth']}, "
        f"use_similarity_loss={CONFIG['use_similarity_loss']}, "
        f"use_gate_dynamic_weight={CONFIG['use_gate_dynamic_weight']}, "
        f"use_outer_gate={CONFIG['use_outer_gate']}, "
        f"use_prefusion_denoise={CONFIG['use_prefusion_denoise']}, "
        f"use_crcf={CONFIG['use_crcf']}, "
        f"crcf_aux_weight={CONFIG['crcf_aux_weight']}, "
        f"crcf_balance_weight={CONFIG['crcf_balance_weight']}, "
        f"label_smoothing={CONFIG['label_smoothing']}, "
        f"use_ema={CONFIG['use_ema']}, patience={CONFIG['patience']}, "
        f"aug_noise_std={CONFIG['aug_noise_std']}, "
        f"aug_drift_prob={CONFIG['aug_drift_prob']}, "
        f"init_weights={CONFIG['init_weights'] or 'none'}, "
        f"data_root={CONFIG['data_root']}, split_dir={CONFIG['split_dir']}, "
        f"save_dir={CONFIG['save_dir']}"
    )

    # ============================================================
    # Data and the fixed stratified 80/10/10 split
    # ============================================================
    full_dataset = SensorDataset(CONFIG["data_root"], is_train=False)
    num_samples = len(full_dataset)
    os.makedirs(CONFIG["split_dir"], exist_ok=True)

    train_idx_path = os.path.join(CONFIG["split_dir"], "train_indices.npy")
    val_idx_path = os.path.join(CONFIG["split_dir"], "val_indices.npy")
    test_idx_path = os.path.join(CONFIG["split_dir"], "test_indices.npy")

    labels_all = full_dataset.labels.astype(int)

    if (
        os.path.exists(train_idx_path)
        and os.path.exists(val_idx_path)
        and os.path.exists(test_idx_path)
    ):
        train_indices = np.load(train_idx_path)
        val_indices = np.load(val_idx_path)
        test_indices = np.load(test_idx_path)
        print(f"Loaded fixed 80/10/10 split: {CONFIG['split_dir']}")
    else:
        rng = np.random.RandomState(CONFIG["split_seed"])
        train_list, val_list, test_list = [], [], []

        for cls in np.unique(labels_all):
            cls_indices = np.where(labels_all == cls)[0]
            rng.shuffle(cls_indices)

            n = len(cls_indices)
            n_train = int(round(0.80 * n))
            n_val = int(round(0.10 * n))

            if n_train + n_val >= n:
                n_train = max(n - 2, 1)
                n_val = 1

            train_list.append(cls_indices[:n_train])
            val_list.append(cls_indices[n_train:n_train + n_val])
            test_list.append(cls_indices[n_train + n_val:])

        train_indices = np.concatenate(train_list)
        val_indices = np.concatenate(val_list)
        test_indices = np.concatenate(test_list)

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)

        np.save(train_idx_path, train_indices)
        np.save(val_idx_path, val_indices)
        np.save(test_idx_path, test_indices)
        print(f"Created fixed stratified 80/10/10 split: {CONFIG['split_dir']}")

    assert len(set(train_indices).intersection(set(val_indices))) == 0, "train/val overlap"
    assert len(set(train_indices).intersection(set(test_indices))) == 0, "train/test overlap"
    assert len(set(val_indices).intersection(set(test_indices))) == 0, "val/test overlap"
    assert len(train_indices) + len(val_indices) + len(test_indices) == num_samples, "split size"

    def print_label_distribution(name, indices):
        y = labels_all[indices]
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"{name} label distribution: {dist}")

    print(f"Total samples: {num_samples}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    print_label_distribution("Train", train_indices)
    print_label_distribution("Val", val_indices)
    print_label_distribution("Test", test_indices)

    channel_mean, channel_std = None, None
    if CONFIG.get("use_channel_norm", True):
        channel_mean, channel_std = compute_channel_stats(full_dataset.data, train_indices)
        print("Channel mean:", np.round(channel_mean, 4).tolist())
        print("Channel std:", np.round(channel_std, 4).tolist())

    class_weights = compute_class_weights(labels_all, train_indices, CONFIG["num_classes"])
    print("Class weights:", np.round(class_weights, 4).tolist())

    # Only the training split uses augmentation.
    train_set = torch.utils.data.Subset(
        SensorDataset(
            CONFIG["data_root"], is_train=True,
            channel_mean=channel_mean, channel_std=channel_std,
        ),
        train_indices,
    )
    val_set = torch.utils.data.Subset(
        SensorDataset(
            CONFIG["data_root"], is_train=False,
            channel_mean=channel_mean, channel_std=channel_std,
        ),
        val_indices,
    )
    test_set = torch.utils.data.Subset(
        SensorDataset(
            CONFIG["data_root"], is_train=False,
            channel_mean=channel_mean, channel_std=channel_std,
        ),
        test_indices,
    )

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], shuffle=False)

    # ============================================================
    # Model, optimization and training
    # ============================================================
    model = SignalTransformerModel().to(device)
    init_weights = CONFIG.get("init_weights", "")
    if init_weights:
        state_dict = torch.load(init_weights, map_location=device)
        missing_keys, unexpected_keys, skipped_keys = load_state_dict_flexible(model, state_dict)
        print(f"Loaded initial weights from: {init_weights}")
        if missing_keys or unexpected_keys or skipped_keys:
            print(
                f"  flexible_load missing={len(missing_keys)}, "
                f"unexpected={len(unexpected_keys)}, skipped_shape={len(skipped_keys)}"
            )
            if skipped_keys:
                print(f"  skipped shape-mismatch keys (first 8): {skipped_keys[:8]}")

    ema_model = create_ema_model(model) if CONFIG.get("use_ema", True) else None
    was_training = model.training
    model.eval()
    with torch.no_grad():
        g = torch.randn(2, 3, 1024).to(device)
        a = torch.randn(2, 3, 1024).to(device)
        v = torch.randn(2, 3, 1024).to(device)
        logits, aux, w, pt, bt, ot = model(g, a, v)
        feat = model.forward_features(g, a, v)
        print("logits:", logits.shape)
        print("w:", w.shape)
        print("feat:", feat.shape)
    if was_training:
        model.train()

    criterion = HybridLoss(CONFIG["num_classes"]).to(device)
    if CONFIG.get("use_class_weight", True):
        criterion.main_class_weights.copy_(torch.tensor(class_weights, device=device))

    trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_acc = 0.0
    patience = int(CONFIG.get("patience", 20))
    counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if init_weights:
        init_val_model = ema_model if ema_model is not None else model
        init_val_loss, init_val_acc = validate(init_val_model, val_loader, criterion)
        best_val_acc = init_val_acc
        torch.save(init_val_model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
        print(
            f"Init checkpoint validation: loss={init_val_loss:.4f}, "
            f"Val Acc={init_val_acc:.2f}% (saved as initial best)"
        )

    print("Starting DSFN training ...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0

        for g, a, v, y in train_loader:
            g, a, v, y = g.to(device), a.to(device), v.to(device), y.to(device)
            optimizer.zero_grad()
            logits, aux, w, probs_t, bases_t, outs_t = model(g, a, v)
            loss = criterion(logits, aux, y, probs_t, bases_t, outs_t)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, decay=float(CONFIG.get("ema_decay", 0.995)))

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += y.size(0)

        train_loss = t_loss / len(train_loader)
        train_acc = 100 * t_correct / t_total

        val_model = ema_model if ema_model is not None else model
        val_loss, val_acc = validate(val_model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:03d} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% "
            f"| LR: {current_lr:.6f} "
            f"| weights[G:{w[:, 0].mean():.2f}, A:{w[:, 1].mean():.2f}, V:{w[:, 2].mean():.2f}]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(val_model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            counter = 0
            print(f"  New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break

    print(f"Training finished. Best validation accuracy: {best_val_acc:.2f}%")
    print("Generating final reports and figures ...")

    plot_history(history, CONFIG["save_dir"])
    model.load_state_dict(
        torch.load(os.path.join(CONFIG["save_dir"], "best_model.pth"), map_location=device)
    )

    if CONFIG.get("clean_test_only", False):
        test_eval_dir = os.path.join(CONFIG["save_dir"], "test_eval_full_clean")
        os.makedirs(test_eval_dir, exist_ok=True)
        print("Running the single full-load clean test evaluation.")
        evaluate_metrics(model, test_loader, device, test_eval_dir)
        return

    if not CONFIG.get("run_final_reports", True):
        print("Skipping final reports because DSFN_RUN_FINAL_REPORTS=0.")
        return

    report_model_complexity(model, device)

    # Validation evaluation
    val_eval_dir = os.path.join(CONFIG["save_dir"], "val_eval")
    os.makedirs(val_eval_dir, exist_ok=True)
    print("\n================ Validation Evaluation ================")
    evaluate_detailed_performance(model, val_loader, device, val_eval_dir)
    visualize_tsne(model, val_loader, device, val_eval_dir)
    plot_confusion_matrix(model, val_loader, device, val_eval_dir)
    visualize_modal_weights(model, val_loader, device, val_eval_dir)
    plot_saturation_evolution(model, val_loader, device, val_eval_dir)
    evaluate_metrics(model, val_loader, device, val_eval_dir)
    analyze_dynamic_usage(
        model, val_loader, device, val_eval_dir, threshold=CONFIG["gate_threshold"]
    )

    # Test evaluation (only once, with the best validation model)
    test_eval_dir = os.path.join(CONFIG["save_dir"], "test_eval")
    os.makedirs(test_eval_dir, exist_ok=True)
    print("\n================ Test Evaluation ================")
    evaluate_detailed_performance(model, test_loader, device, test_eval_dir)
    plot_confusion_matrix(model, test_loader, device, test_eval_dir)
    visualize_modal_weights(model, test_loader, device, test_eval_dir)
    evaluate_metrics(model, test_loader, device, test_eval_dir)
    analyze_dynamic_usage(
        model, test_loader, device, test_eval_dir, threshold=CONFIG["gate_threshold"]
    )

    test_fps(model, device, val_loader=val_loader, gate_threshold=CONFIG["gate_threshold"])
    test_fps(
        model,
        device,
        val_loader=val_loader,
        gate_threshold=CONFIG["gate_threshold"],
        force_full=True,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    from paper_training import main as train_paper
    train_paper()

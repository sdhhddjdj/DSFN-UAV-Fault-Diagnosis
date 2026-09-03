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
# 1. 配置参数
# ==========================================
CONFIG = {
    "batch_size": _env_int("SGLA_BATCH_SIZE", 64),
    "epochs": _env_int("SGLA_EPOCHS", 150),
    "patience": _env_int("SGLA_PATIENCE", 20),
    "lr": _env_float("SGLA_LR", 3e-4),
    "weight_decay": _env_float("SGLA_WEIGHT_DECAY", 1e-4),
    "num_classes": 6,
    "data_root": os.environ.get("SGLA_DATA_ROOT", "dataset"),

    # 实验名称
        "exp_name": "Denoise_RCDFusion",

    # 模型结构开关
    "use_temporal_cnn": True,
    "use_cross_attn": True,
    # 动态融合拆成两部分
    "use_gate_dynamic_weight": True,      # base阶段是否用动态融合生成门控权重
    "use_final_dynamic_fusion": True,     # 最终分类前是否用动态融合
    "use_rcd_fusion": False,             # RCD-Fusion: 可靠性-一致性动态融合
    "rcd_aux_weight": 0.20,              # 单模态辅助分类损失权重
    "rcd_balance_weight": 0.01,          # 融合权重批均衡正则权重
    "rcd_use_reliability": True,
    "rcd_use_consistency": True,
    "rcd_use_certainty": True,

    "use_outer_gate": True,
    "use_matl": False,

    # SGLA 相关
    "use_sgla": True,          # 是否使用动态层选择器
    "use_sgla_loss": True,     # 是否使用相邻层相似度约束
    "transformer_depth": _env_int("SGLA_TRANSFORMER_DEPTH", 5),
    "transformer_split_layer": _env_int("SGLA_TRANSFORMER_SPLIT_LAYER", 3),
    # Explicit temporal order information after patch embedding. Sinusoidal
    # encoding is parameter-free and supports both 16- and 32-point patches.
    "use_positional_encoding": _env_bool("SGLA_USE_POSITIONAL_ENCODING", True),
    "position_dropout": _env_float("SGLA_POSITION_DROPOUT", 0.0),
    "gate_threshold": 0.25,
    #降噪
    "use_prefusion_denoise": True,
    #AMCF
    "use_amcf": False,
    "amcf_filters": 4,
    "amcf_kernel": 31,
    # 训练策略开关
    "use_aug": True,
    "aug_noise_std": _env_float("SGLA_AUG_NOISE_STD", 0.005),
    "aug_drift_prob": _env_float("SGLA_AUG_DRIFT_PROB", 0.5),
    "aug_scale_min": _env_float("SGLA_AUG_SCALE_MIN", 0.98),
    "aug_scale_max": _env_float("SGLA_AUG_SCALE_MAX", 1.02),
    "aug_time_shift_max": _env_int("SGLA_AUG_TIME_SHIFT_MAX", 0),
    "use_mixup": _env_bool("SGLA_USE_MIXUP", False),
    "mixup_alpha": _env_float("SGLA_MIXUP_ALPHA", 0.2),
    "mixup_prob": _env_float("SGLA_MIXUP_PROB", 0.5),
    "use_class_weight": True,
    "label_smoothing": _env_float("SGLA_LABEL_SMOOTHING", 0.0),
    "use_channel_norm": True,
    "eval_force_full": True,
    "eval_soft_select": True,
    "sim_skip_threshold": None,
    "fast_gate_thresholds": _env_float_list("SGLA_FAST_GATE_THRESHOLDS", [0.25, 0.35, 0.45, 0.55]),
    # Runtime cosine-threshold skipping is kept for ablation only. The selector
    # path is the recommended fast inference mode after FPS/FLOPs evaluation.
    "sim_skip_thresholds": _env_float_list("SGLA_SIM_SKIP_THRESHOLDS", []),
    "use_cheap_exit": True,
    "cheap_aux_weight": _env_float("SGLA_CHEAP_AUX_WEIGHT", 0.30),
    "cheap_kd_weight": _env_float("SGLA_CHEAP_KD_WEIGHT", 0.0),
    "cheap_kd_temperature": _env_float("SGLA_CHEAP_KD_TEMPERATURE", 2.0),
    "cheap_conf_temperature": _env_float("SGLA_CHEAP_CONF_TEMPERATURE", 1.0),
    "cheap_head_width": _env_int("SGLA_CHEAP_HEAD_WIDTH", 48),
    "cheap_head_stats_width": _env_int("SGLA_CHEAP_HEAD_STATS_WIDTH", 96),
    "cheap_head_dropout": _env_float("SGLA_CHEAP_HEAD_DROPOUT", 0.15),
    "fast_exit_thresholds": _env_float_list("SGLA_FAST_EXIT_THRESHOLDS", [0.80, 0.85, 0.90, 0.95]),
    "train_cheap_only": _env_bool("SGLA_TRAIN_CHEAP_ONLY", False),
    "use_shallow_exit": _env_bool("SGLA_USE_SHALLOW_EXIT", False),
    "train_shallow_only": _env_bool("SGLA_TRAIN_SHALLOW_ONLY", False),
    "train_fusion_head_only": _env_bool("SGLA_TRAIN_FUSION_HEAD_ONLY", False),
    "shallow_aux_weight": _env_float("SGLA_SHALLOW_AUX_WEIGHT", 0.0),
    "shallow_kd_weight": _env_float("SGLA_SHALLOW_KD_WEIGHT", 0.0),
    "shallow_kd_temperature": _env_float("SGLA_SHALLOW_KD_TEMPERATURE", 2.0),
    "shallow_conf_temperature": _env_float("SGLA_SHALLOW_CONF_TEMPERATURE", 1.0),
    "shallow_head_dropout": _env_float("SGLA_SHALLOW_HEAD_DROPOUT", 0.1),
    "shallow_exit_base_layers": _env_int("SGLA_SHALLOW_EXIT_BASE_LAYERS", 3),
    "use_ema": _env_bool("SGLA_USE_EMA", True),
    "ema_decay": _env_float("SGLA_EMA_DECAY", 0.995),
    "model_dropout": _env_float("SGLA_MODEL_DROPOUT", 0.2),
    "use_global_branch": _env_bool("SGLA_USE_GLOBAL_BRANCH", False),
    "global_branch_width": _env_int("SGLA_GLOBAL_BRANCH_WIDTH", 64),
    "global_branch_init_alpha": _env_float("SGLA_GLOBAL_BRANCH_INIT_ALPHA", -1.5),
    "teacher_model": os.environ.get("SGLA_TEACHER_MODEL", "").strip().lower(),
    "teacher_path": os.environ.get("SGLA_TEACHER_PATH", ""),
    "teacher_kd_weight": _env_float("SGLA_TEACHER_KD_WEIGHT", 0.0),
    "teacher_kd_temperature": _env_float("SGLA_TEACHER_KD_TEMPERATURE", 3.0),
    "fast_eval_max_batches": _env_int("SGLA_FAST_EVAL_MAX_BATCHES", 0),
    "fps_num_batches": _env_int("SGLA_FPS_NUM_BATCHES", 16),
    "fps_batch_size": _env_int("SGLA_FPS_BATCH_SIZE", 1),

    # 固定划分
    "split_dir": os.environ.get("SGLA_SPLIT_DIR", os.path.join("dataset", "fixed_split_80_10_10")),
    "split_seed": 42,
    "seed": 3407,

    # 保存路径
    "save_root": "runs",
    "init_weights": os.environ.get("SGLA_INIT_WEIGHTS", ""),
}
CONFIG["save_root"] = os.environ.get("SGLA_SAVE_ROOT", "runs_sgla_rcd_fusion_80_10_10")
CONFIG["run_final_reports"] = _env_bool("SGLA_RUN_FINAL_REPORTS", True)
CONFIG["clean_test_only"] = _env_bool("SGLA_CLEAN_TEST_ONLY", False)
CONFIG["save_dir"] = os.path.join(
    CONFIG["save_root"],
    f"{CONFIG['exp_name']}_seed{CONFIG['seed']}"
)
os.makedirs(CONFIG["save_dir"], exist_ok=True)
ABLATION_CONFIGS = {
"Denoised_DynamicFusion": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_cross_residual": True,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,

    # 第一版独立三分支去噪
    "use_prefusion_denoise": True,

    # 关闭 AMCF
    "use_amcf": False,
},
"Denoised_GatedCross_ResidualDynamicFusion": {
    "use_temporal_cnn": True,
    "use_cross_attn": True,
    "use_cross_residual": True,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": True,
    "use_amcf": False,
},
"Full": {
    "use_temporal_cnn": True,
    "use_cross_attn": True,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": False,
    "use_cross_residual": True,
    "use_amcf": False,
    "rcd_use_reliability": True,
    "rcd_use_consistency": True,
    "rcd_use_certainty": True,
},
"Proposed": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": False,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": False,
    "use_amcf": False,
},

"AMCF_AvgFusion": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": False,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": False,
    "use_amcf": True,
    "amcf_filters": 4,
    "amcf_kernel": 31,
},

"AMCF_DynamicFusion": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": False,
    "use_amcf": True,
    "amcf_filters": 4,
    "amcf_kernel": 31,
},
"Denoise_AvgFusion": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": False,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": True,
},

"Denoise_RCDFusion": {
    # 在 Denoise_AvgFusion 基础上，仅将最终平均融合替换为 RCD-Fusion；
    # 其余结构与训练策略尽量保持一致，便于做单因素对比。
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": False,
    "use_rcd_fusion": True,
    "rcd_aux_weight": 0.05,
    "rcd_balance_weight": 0.01,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.0,
    "use_matl": False,
    "use_prefusion_denoise": True,
},

"DynamicFusion_NoDenoise": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": False,
},

"Denoised_DynamicFusion-shared": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": True,
    "use_outer_gate": True,
    "use_sgla": True,
    "use_sgla_loss": True,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    "use_prefusion_denoise": True,
},
    "w_o_TemporalCNN": {
        "use_temporal_cnn": False,
    },

    "w_o_CrossAttn": {
        "use_cross_attn": False,
    },

    "w_o_DynamicFusion": {
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
    },

    "w_o_ModalGate": {
        "use_outer_gate": False,
    },

    "w_o_SGLA_DynamicDepth": {
    "use_temporal_cnn": True,
    "use_cross_attn": False,
    "use_gate_dynamic_weight": True,
    "use_final_dynamic_fusion": False,
    "use_outer_gate": False,
    "use_sgla": False,
    "use_sgla_loss": False,
    "use_aug": True,
    "use_class_weight": True,
    "label_smoothing": 0.05,
    "use_matl": False,
    },

    "w_o_SGLA_Loss": {
        "use_sgla": True,
        "use_sgla_loss": False,
    },

    "w_o_DataAug": {
        "use_aug": False,
    },

    "w_o_ClassWeight": {
        "use_class_weight": False,
    },

    "w_o_LabelSmoothing": {
        "label_smoothing": 0.0,
    },
    "w_o_CrossAttn_DynamicFusion": {
        "use_cross_attn": False,
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
    },
}

PAPER_ABLATION_CONFIGS = {
    # Paper_* variants are kept orthogonal for thesis/journal tables.
    # They progressively add the proposed components on the same backbone.
    "Paper_A0_StaticAvg": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": False,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": False,
        "use_outer_gate": False,
        "use_sgla": False,
        "use_sgla_loss": False,
        "use_prefusion_denoise": False,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A1_DynamicDepth": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": False,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": False,
        "use_outer_gate": False,
        "use_sgla": True,
        "use_sgla_loss": False,
        "use_prefusion_denoise": False,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A2_DynamicDepthLoss": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": False,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": False,
        "use_outer_gate": False,
        "use_sgla": True,
        "use_sgla_loss": True,
        "use_prefusion_denoise": False,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A3_ModalGate": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": False,
        "use_outer_gate": True,
        "use_sgla": True,
        "use_sgla_loss": True,
        "use_prefusion_denoise": False,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A4_Denoise": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": False,
        "use_outer_gate": True,
        "use_sgla": True,
        "use_sgla_loss": True,
        "use_prefusion_denoise": True,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A5_RCD_Fusion": {
        "use_temporal_cnn": True,
        "use_cross_attn": False,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": True,
        "rcd_aux_weight": 0.05,
        "rcd_balance_weight": 0.01,
        "rcd_use_reliability": True,
        "rcd_use_consistency": True,
        "rcd_use_certainty": True,
        "use_outer_gate": True,
        "use_sgla": True,
        "use_sgla_loss": True,
        "use_prefusion_denoise": True,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
    "Paper_A6_CrossModal": {
        "use_temporal_cnn": True,
        "use_cross_attn": True,
        "use_cross_residual": True,
        "use_gate_dynamic_weight": True,
        "use_final_dynamic_fusion": False,
        "use_rcd_fusion": True,
        "rcd_aux_weight": 0.05,
        "rcd_balance_weight": 0.01,
        "rcd_use_reliability": True,
        "rcd_use_consistency": True,
        "rcd_use_certainty": True,
        "use_outer_gate": True,
        "use_sgla": True,
        "use_sgla_loss": True,
        "use_prefusion_denoise": True,
        "use_aug": True,
        "use_class_weight": True,
        "label_smoothing": 0.0,
        "use_amcf": False,
    },
}
REVISED_PAPER_CONFIGS = {
    "Paper_R_A3_CRCF_NoGTD": {
        **PAPER_ABLATION_CONFIGS["Paper_A5_RCD_Fusion"],
        "use_prefusion_denoise": False,
        "rcd_use_reliability": True,
        "rcd_use_consistency": True,
        "rcd_use_certainty": True,
    },
    "Paper_R_B1_Reliability": {
        **PAPER_ABLATION_CONFIGS["Paper_A5_RCD_Fusion"],
        "rcd_use_reliability": True,
        "rcd_use_consistency": False,
        "rcd_use_certainty": False,
    },
    "Paper_R_B2_ReliabilityConsistency": {
        **PAPER_ABLATION_CONFIGS["Paper_A5_RCD_Fusion"],
        "rcd_use_reliability": True,
        "rcd_use_consistency": True,
        "rcd_use_certainty": False,
    },
    "Paper_R_B3_ReliabilityCertainty": {
        **PAPER_ABLATION_CONFIGS["Paper_A5_RCD_Fusion"],
        "rcd_use_reliability": True,
        "rcd_use_consistency": False,
        "rcd_use_certainty": True,
    },
}
PAPER_ABLATION_CONFIGS.update(REVISED_PAPER_CONFIGS)
for _paper_name, _paper_extra in {
    "Paper_A7_GlobalBranch": {
        "use_global_branch": True,
    },
    "Paper_A8_TCN_KD": {
        "teacher_model": "tcn",
        "teacher_kd_weight": 0.5,
        "teacher_kd_temperature": 3.0,
    },
    "Paper_A9_ResNet_KD": {
        "teacher_model": "resnet1d",
        "teacher_kd_weight": 0.5,
        "teacher_kd_temperature": 3.0,
    },
    "Paper_A10_Global_TCN_KD": {
        "use_global_branch": True,
        "teacher_model": "tcn",
        "teacher_kd_weight": 0.5,
        "teacher_kd_temperature": 3.0,
    },
}.items():
    PAPER_ABLATION_CONFIGS[_paper_name] = {
        **PAPER_ABLATION_CONFIGS["Paper_A5_RCD_Fusion"],
        **_paper_extra,
    }
ABLATION_CONFIGS.update(PAPER_ABLATION_CONFIGS)
def apply_ablation_config(exp_name):
    # 先加载 Full 配置
    CONFIG.update(ABLATION_CONFIGS["Full"])
    CONFIG.update(ABLATION_CONFIGS[exp_name])

    # 再覆盖当前实验配置
    CONFIG.update(ABLATION_CONFIGS[exp_name])

    CONFIG["exp_name"] = exp_name
    CONFIG["save_dir"] = os.path.join(
        CONFIG["save_root"],
        f"{CONFIG['exp_name']}_seed{CONFIG['seed']}"
    )
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

def apply_runtime_overrides():
    env_map = {
        "SGLA_EXP_NAME": ("exp_name", str),
        "SGLA_DATA_ROOT": ("data_root", str),
        "SGLA_SPLIT_DIR": ("split_dir", str),
        "SGLA_SAVE_ROOT": ("save_root", str),
        "SGLA_INIT_WEIGHTS": ("init_weights", str),
        "SGLA_SEED": ("seed", int),
        "SGLA_SPLIT_SEED": ("split_seed", int),
        "SGLA_EPOCHS": ("epochs", int),
        "SGLA_PATIENCE": ("patience", int),
        "SGLA_BATCH_SIZE": ("batch_size", int),
        "SGLA_LR": ("lr", float),
        "SGLA_WEIGHT_DECAY": ("weight_decay", float),
        "SGLA_FAST_EVAL_MAX_BATCHES": ("fast_eval_max_batches", int),
        "SGLA_FPS_NUM_BATCHES": ("fps_num_batches", int),
        "SGLA_FPS_BATCH_SIZE": ("fps_batch_size", int),
        "SGLA_LABEL_SMOOTHING": ("label_smoothing", float),
        "SGLA_RCD_AUX_WEIGHT": ("rcd_aux_weight", float),
        "SGLA_RCD_BALANCE_WEIGHT": ("rcd_balance_weight", float),
        "SGLA_RCD_USE_RELIABILITY": ("rcd_use_reliability", _parse_bool),
        "SGLA_RCD_USE_CONSISTENCY": ("rcd_use_consistency", _parse_bool),
        "SGLA_RCD_USE_CERTAINTY": ("rcd_use_certainty", _parse_bool),
        "SGLA_CHEAP_AUX_WEIGHT": ("cheap_aux_weight", float),
        "SGLA_CHEAP_KD_WEIGHT": ("cheap_kd_weight", float),
        "SGLA_CHEAP_KD_TEMPERATURE": ("cheap_kd_temperature", float),
        "SGLA_CHEAP_CONF_TEMPERATURE": ("cheap_conf_temperature", float),
        "SGLA_CHEAP_HEAD_WIDTH": ("cheap_head_width", int),
        "SGLA_CHEAP_HEAD_STATS_WIDTH": ("cheap_head_stats_width", int),
        "SGLA_CHEAP_HEAD_DROPOUT": ("cheap_head_dropout", float),
        "SGLA_SHALLOW_AUX_WEIGHT": ("shallow_aux_weight", float),
        "SGLA_SHALLOW_KD_WEIGHT": ("shallow_kd_weight", float),
        "SGLA_SHALLOW_KD_TEMPERATURE": ("shallow_kd_temperature", float),
        "SGLA_SHALLOW_CONF_TEMPERATURE": ("shallow_conf_temperature", float),
        "SGLA_SHALLOW_HEAD_DROPOUT": ("shallow_head_dropout", float),
        "SGLA_SHALLOW_EXIT_BASE_LAYERS": ("shallow_exit_base_layers", int),
        "SGLA_EMA_DECAY": ("ema_decay", float),
        "SGLA_MODEL_DROPOUT": ("model_dropout", float),
        "SGLA_TRANSFORMER_DEPTH": ("transformer_depth", int),
        "SGLA_TRANSFORMER_SPLIT_LAYER": ("transformer_split_layer", int),
        "SGLA_USE_POSITIONAL_ENCODING": ("use_positional_encoding", _parse_bool),
        "SGLA_POSITION_DROPOUT": ("position_dropout", float),
        "SGLA_AUG_NOISE_STD": ("aug_noise_std", float),
        "SGLA_AUG_DRIFT_PROB": ("aug_drift_prob", float),
        "SGLA_AUG_SCALE_MIN": ("aug_scale_min", float),
        "SGLA_AUG_SCALE_MAX": ("aug_scale_max", float),
        "SGLA_AUG_TIME_SHIFT_MAX": ("aug_time_shift_max", int),
        "SGLA_MIXUP_ALPHA": ("mixup_alpha", float),
        "SGLA_MIXUP_PROB": ("mixup_prob", float),
        "SGLA_GLOBAL_BRANCH_WIDTH": ("global_branch_width", int),
        "SGLA_GLOBAL_BRANCH_INIT_ALPHA": ("global_branch_init_alpha", float),
        "SGLA_TEACHER_MODEL": ("teacher_model", str),
        "SGLA_TEACHER_PATH": ("teacher_path", str),
        "SGLA_TEACHER_KD_WEIGHT": ("teacher_kd_weight", float),
        "SGLA_TEACHER_KD_TEMPERATURE": ("teacher_kd_temperature", float),
    }

    for env_name, (config_key, caster) in env_map.items():
        if env_name in os.environ:
            CONFIG[config_key] = caster(os.environ[env_name])

    CONFIG["run_final_reports"] = _env_bool(
        "SGLA_RUN_FINAL_REPORTS",
        CONFIG.get("run_final_reports", True),
    )
    CONFIG["clean_test_only"] = _env_bool(
        "SGLA_CLEAN_TEST_ONLY",
        CONFIG.get("clean_test_only", False),
    )
    if "SGLA_FAST_GATE_THRESHOLDS" in os.environ:
        CONFIG["fast_gate_thresholds"] = _env_float_list(
            "SGLA_FAST_GATE_THRESHOLDS",
            CONFIG.get("fast_gate_thresholds", [CONFIG.get("gate_threshold", 0.25)]),
        )
    if "SGLA_SIM_SKIP_THRESHOLDS" in os.environ:
        CONFIG["sim_skip_thresholds"] = _env_float_list(
            "SGLA_SIM_SKIP_THRESHOLDS",
            CONFIG.get("sim_skip_thresholds", []),
        )
    if "SGLA_SIM_SKIP_THRESHOLD" in os.environ:
        CONFIG["sim_skip_threshold"] = _env_float("SGLA_SIM_SKIP_THRESHOLD", None)
    if "SGLA_FAST_EXIT_THRESHOLDS" in os.environ:
        CONFIG["fast_exit_thresholds"] = _env_float_list(
            "SGLA_FAST_EXIT_THRESHOLDS",
            CONFIG.get("fast_exit_thresholds", []),
        )
    CONFIG["use_cheap_exit"] = _env_bool(
        "SGLA_USE_CHEAP_EXIT",
        CONFIG.get("use_cheap_exit", True),
    )
    CONFIG["use_ema"] = _env_bool(
        "SGLA_USE_EMA",
        CONFIG.get("use_ema", True),
    )
    CONFIG["train_cheap_only"] = _env_bool(
        "SGLA_TRAIN_CHEAP_ONLY",
        CONFIG.get("train_cheap_only", False),
    )
    CONFIG["use_shallow_exit"] = _env_bool(
        "SGLA_USE_SHALLOW_EXIT",
        CONFIG.get("use_shallow_exit", False),
    )
    CONFIG["train_shallow_only"] = _env_bool(
        "SGLA_TRAIN_SHALLOW_ONLY",
        CONFIG.get("train_shallow_only", False),
    )
    CONFIG["train_fusion_head_only"] = _env_bool(
        "SGLA_TRAIN_FUSION_HEAD_ONLY",
        CONFIG.get("train_fusion_head_only", False),
    )
    CONFIG["use_aug"] = _env_bool(
        "SGLA_USE_AUG",
        CONFIG.get("use_aug", True),
    )
    CONFIG["use_mixup"] = _env_bool(
        "SGLA_USE_MIXUP",
        CONFIG.get("use_mixup", False),
    )
    CONFIG["use_class_weight"] = _env_bool(
        "SGLA_USE_CLASS_WEIGHT",
        CONFIG.get("use_class_weight", True),
    )
    CONFIG["use_global_branch"] = _env_bool(
        "SGLA_USE_GLOBAL_BRANCH",
        CONFIG.get("use_global_branch", False),
    )
    CONFIG["teacher_model"] = str(CONFIG.get("teacher_model", "")).strip().lower()
    CONFIG["save_dir"] = os.path.join(
        CONFIG["save_root"],
        f"{CONFIG['exp_name']}_seed{CONFIG['seed']}"
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

class GatedResidualCrossAttention(nn.Module):
    """
    残差门控式跨模态注意力：
    out = LayerNorm(x + alpha * CrossAttn(x, context, context))

    alpha 初始很小，让模型一开始接近原始特征，
    训练中如果跨模态信息有用，再逐渐增大贡献。
    """
    def __init__(self, dim, num_heads=4, dropout=0.1, init_alpha=-2.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # sigmoid(-2) ≈ 0.12，初始只引入少量跨模态信息
        self.alpha_logit = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x, context):
        q = self.norm_q(x)
        kv = self.norm_kv(context)

        cross, _ = self.attn(q, kv, kv, need_weights=False)

        alpha = torch.sigmoid(self.alpha_logit)
        out = self.out_norm(x + alpha * self.dropout(cross))
        return out
############################################
# Patch Embedding
############################################
class SinusoidalPositionEncoding1D(nn.Module):
    def __init__(self, embed_dim, max_length=1024, dropout=0.0):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(max_length, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if embed_dim > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        # Non-persistent keeps old checkpoints loadable because no new state key
        # is introduced by this deterministic buffer.
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"Token length {x.size(1)} exceeds positional encoding limit "
                f"{self.encoding.size(1)}."
            )
        return self.dropout(x + self.encoding[:, : x.size(1)].to(dtype=x.dtype))


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_ch=3,
        embed_dim=128,
        patch=16,
        use_positional_encoding=True,
        position_dropout=0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.position = (
            SinusoidalPositionEncoding1D(embed_dim, max_length=1024, dropout=position_dropout)
            if use_positional_encoding
            else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return self.position(x)
############################################
# 局部特征增强
############################################
class AMCF1D(nn.Module):
    """
    AMCF-like adaptive multi-channel filter.
    输入:  [B, C, L]
    输出:  [B, C*K, L]

    说明：
    这是参考 MATL 中 AMCF 思想的可学习带通滤波器组，
    不是原文代码的完全复现。
    """
    def __init__(self, in_channels=3, num_filters=4, kernel_size=31):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # 每个通道、每个滤波器一组可学习频率参数
        self.low_raw = nn.Parameter(torch.randn(in_channels, num_filters) * 0.1)
        self.band_raw = nn.Parameter(torch.randn(in_channels, num_filters) * 0.1)

        n = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        self.register_buffer("n", n)

        # Hanning window
        window = 0.5 - 0.5 * torch.cos(
            2 * torch.pi * torch.arange(kernel_size).float() / (kernel_size - 1)
        )
        self.register_buffer("window", window)

    def _build_filters(self):
        """
        构造 sinc band-pass filters.
        返回: [C*K, 1, kernel_size]
        """
        # 归一化频率范围控制在 0~0.5 内
        low = 0.02 + 0.20 * torch.sigmoid(self.low_raw)
        band = 0.02 + 0.25 * torch.sigmoid(self.band_raw)
        high = torch.clamp(low + band, max=0.49)

        n = self.n.to(low.device).view(1, 1, -1)
        window = self.window.to(low.device).view(1, 1, -1)

        low = low.unsqueeze(-1)
        high = high.unsqueeze(-1)

        eps = 1e-8
        two_pi_n = 2 * torch.pi * n

        sinc_high = torch.where(
            torch.abs(n) < eps,
            2 * high,
            torch.sin(two_pi_n * high) / (torch.pi * n)
        )

        sinc_low = torch.where(
            torch.abs(n) < eps,
            2 * low,
            torch.sin(two_pi_n * low) / (torch.pi * n)
        )

        band_pass = (sinc_high - sinc_low) * window

        # 幅值归一化，避免输出尺度过大
        band_pass = band_pass / (band_pass.abs().sum(dim=-1, keepdim=True) + 1e-8)

        # [C, K, kernel] -> [C*K, 1, kernel]
        filters = band_pass.view(self.in_channels * self.num_filters, 1, self.kernel_size)
        return filters

    def forward(self, x):
        # x: [B, C, L]
        filters = self._build_filters()
        y = F.conv1d(
            x,
            filters,
            padding=self.kernel_size // 2,
            groups=self.in_channels
        )
        return y
class AMCFBlock(nn.Module):
    """
    AMCF + BN + activation
    输入:  [B, C, L]
    输出:  [B, C*K, L]
    """
    def __init__(self, in_channels=3, num_filters=4, kernel_size=31):
        super().__init__()
        self.amcf = AMCF1D(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size
        )
        out_channels = in_channels * num_filters
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.amcf(x)
        x = self.norm(x)
        x = self.act(x)
        return x
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
# 🔥 SGLA Transformer Encoder
############################################
class SGLATransformerEncoder(nn.Module):
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

        # 3. SGLA 选择器
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

    def forward_dynamic(self, x, force_full=False, sim_skip_threshold=None, collect_intermediates=True):
        """
        Dynamic depth path.
        Training keeps the differentiable soft-merge path. Evaluation can either
        follow the selector depth or stop early when adjacent-layer cosine
        similarity is already high enough.
        """
        base_feat = x
        num_dynamic = len(self.dynamic_layers)

        if not CONFIG.get("use_sgla", True):
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

        if sim_skip_threshold is not None:
            out = base_feat.clone()
            active_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            prev_active = base_feat

            for layer in self.dynamic_layers:
                if not active_mask.any():
                    break

                curr_active = layer(prev_active)
                if collect_intermediates:
                    buf = out.clone()
                    buf[active_mask] = curr_active
                    layer_outs.append(buf)

                out[active_mask] = curr_active
                sim = F.cosine_similarity(
                    prev_active.mean(dim=1),
                    curr_active.mean(dim=1),
                    dim=1,
                    eps=1e-8,
                )
                stop_active = sim >= sim_skip_threshold
                old_active = active_mask.clone()
                active_mask[old_active] = ~stop_active
                prev_active = curr_active[~stop_active]

            return out, select_probs, base_feat, layer_outs

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
# Cross Modal Attention & Dynamic Fusion
############################################
class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v)
        return out

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
class ResidualDynamicFusion(nn.Module):
    """
    残差式动态融合：
    avg = (g + a + v) / 3
    dynamic = wg*g + wa*a + wv*v
    fused = avg + rho * (dynamic - avg)

    rho 初始较小，避免动态融合一开始破坏稳定的平均融合。
    """
    def __init__(self, dim, hidden=128, temperature=1.5, init_rho=-2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 3)
        )

        self.temperature = temperature
        self.rho_logit = nn.Parameter(torch.tensor(init_rho, dtype=torch.float32))

    def forward(self, g, a, v):
        g_cls = g.mean(1)
        a_cls = a.mean(1)
        v_cls = v.mean(1)

        feat = torch.cat([g_cls, a_cls, v_cls], dim=1)

        logits = self.net(feat)
        w = torch.softmax(logits / self.temperature, dim=1)

        avg = (g_cls + a_cls + v_cls) / 3.0
        dynamic = (
            w[:, 0:1] * g_cls +
            w[:, 1:2] * a_cls +
            w[:, 2:3] * v_cls
        )

        rho = torch.sigmoid(self.rho_logit)
        fused = avg + rho * (dynamic - avg)

        return fused, w





class RCDynamicFusion(nn.Module):
    """
    Reliability-Consistency Dynamic Fusion (RCD-Fusion)

    融合依据：
      1) 单模态分类置信度 confidence；
      2) 模态间一致性 consistency；
      3) SGLA selector 动态深度选择确定性 certainty。

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

        if not CONFIG.get("rcd_use_reliability", True):
            conf_g = torch.ones_like(conf_g)
            conf_a = torch.ones_like(conf_a)
            conf_v = torch.ones_like(conf_v)
        if not CONFIG.get("rcd_use_consistency", True):
            cons_g = torch.ones_like(cons_g)
            cons_a = torch.ones_like(cons_a)
            cons_v = torch.ones_like(cons_v)
        if not CONFIG.get("rcd_use_certainty", True):
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
        rcd_info = {
            "confidence": torch.cat([conf_g, conf_a, conf_v], dim=1),
            "consistency": torch.cat([cons_g, cons_a, cons_v], dim=1),
            "certainty": torch.cat([cert_g, cert_a, cert_v], dim=1),
            "fusion_weights": alpha,
        }
        return fused, alpha, branch_logits, rcd_info


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


class CheapDSConvBlock(nn.Module):
    """Tiny depthwise-separable residual block used by the early-exit head."""
    def __init__(self, channels, dropout=0.1, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class CheapSignalEncoder(nn.Module):
    def __init__(self, in_channels, width=48, dropout=0.15):
        super().__init__()
        width = max(int(width), 8)
        out_channels = width * 2
        self.out_dim = out_channels * 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            CheapDSConvBlock(width, dropout=dropout, kernel_size=7),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3, groups=width, bias=False),
            nn.Conv1d(width, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            CheapDSConvBlock(out_channels, dropout=dropout, kernel_size=5),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, groups=out_channels, bias=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            CheapDSConvBlock(out_channels, dropout=dropout, kernel_size=3),
        )

    def forward(self, x):
        z = self.net(x)
        avg_pool = z.mean(dim=-1)
        max_pool = z.amax(dim=-1)
        return torch.cat([avg_pool, max_pool], dim=1)


class CheapSignalHead(nn.Module):
    """Stronger low-cost raw-signal classifier for confidence-guided early exit."""
    def __init__(self, num_classes=6, width=None):
        super().__init__()
        if width is None:
            width = int(CONFIG.get("cheap_head_width", 48))
        stats_width = int(CONFIG.get("cheap_head_stats_width", 96))
        dropout = float(CONFIG.get("cheap_head_dropout", 0.15))
        modal_width = max(width // 2, 16)

        self.global_encoder = CheapSignalEncoder(9, width=width, dropout=dropout)
        self.modal_encoders = nn.ModuleList([
            CheapSignalEncoder(3, width=modal_width, dropout=dropout),
            CheapSignalEncoder(3, width=modal_width, dropout=dropout),
            CheapSignalEncoder(3, width=modal_width, dropout=dropout),
        ])
        self.stats_mlp = nn.Sequential(
            nn.Linear(9 * 8, stats_width),
            nn.BatchNorm1d(stats_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        feature_dim = self.global_encoder.out_dim
        feature_dim += sum(encoder.out_dim for encoder in self.modal_encoders)
        feature_dim += stats_width
        hidden_dim = max(width * 4, 128)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    @staticmethod
    def _stats(x):
        dx = x[..., 1:] - x[..., :-1]
        return torch.cat(
            [
                x.mean(dim=-1),
                x.std(dim=-1, unbiased=False),
                x.amax(dim=-1),
                x.amin(dim=-1),
                torch.sqrt((x * x).mean(dim=-1).clamp_min(1e-8)),
                dx.mean(dim=-1),
                dx.std(dim=-1, unbiased=False),
                torch.sqrt((dx * dx).mean(dim=-1).clamp_min(1e-8)),
            ],
            dim=1,
        )

    def forward(self, g, a, v):
        x = torch.cat([g, a, v], dim=1)
        features = [self.global_encoder(x)]
        for encoder, signal in zip(self.modal_encoders, (g, a, v)):
            features.append(encoder(signal))
        features.append(self.stats_mlp(self._stats(x)))
        return self.head(torch.cat(features, dim=1))


class FeatureExitHead(nn.Module):
    """Classifier on base Transformer features for a more accurate early exit."""
    def __init__(self, dim=128, num_classes=6, dropout=None):
        super().__init__()
        if dropout is None:
            dropout = float(CONFIG.get("shallow_head_dropout", 0.1))
        feature_dim = dim * 3 * 2
        hidden_dim = dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    @staticmethod
    def _pool(x):
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)

    def forward(self, g, a, v):
        return self.head(torch.cat([self._pool(g), self._pool(a), self._pool(v)], dim=1))


class GlobalResidualBlock1D(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        padding = 2 * dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class GlobalSignalBranch(nn.Module):
    """A compact 9-channel residual TCN branch that preserves cross-modal local patterns."""
    def __init__(self, in_ch=9, dim=128, width=64, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            GlobalResidualBlock1D(width, dilation=1, dropout=dropout),
            GlobalResidualBlock1D(width, dilation=2, dropout=dropout),
            GlobalResidualBlock1D(width, dilation=4, dropout=dropout),
            GlobalResidualBlock1D(width, dilation=8, dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(width * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x):
        z = self.blocks(self.stem(x))
        pooled = torch.cat([z.mean(dim=-1), z.amax(dim=-1)], dim=1)
        return self.proj(pooled)


class SignalTransformerModel(nn.Module):
    def get_gate_weight(self, g, a, v):
        g_cls = g.mean(1)
        B = g_cls.size(0)

        if CONFIG.get("use_gate_dynamic_weight", True):
            _, w = self.gate_fusion(g, a, v)
        else:
            w = torch.ones(B, 3, device=g_cls.device) / 3.0

        return w

    def apply_cross_modal(self, g_c, a_c, v_c):
        if CONFIG.get("use_cross_attn", True):
            if CONFIG.get("use_cross_residual", True):
                g_new = self.cross_g(g_c, a_c)
                a_new = self.cross_a(a_c, v_c)
                v_new = self.cross_v(v_c, g_c)
                g_c, a_c, v_c = g_new, a_new, v_new
            else:
                g_new = self.cross1(g_c, a_c, a_c)
                a_new = self.cross2(a_c, v_c, v_c)
                v_new = self.cross3(v_c, g_c, g_c)
                g_c, a_c, v_c = g_new, a_new, v_new

        return g_c, a_c, v_c
    def final_fuse(self, g, a, v, probs_tuple=None):
        g_cls = g.mean(1)
        a_cls = a.mean(1)
        v_cls = v.mean(1)

        if CONFIG.get("use_rcd_fusion", False):
            fused, w_final, branch_logits, rcd_info = self.rcd_fusion(
                g, a, v, probs_tuple=probs_tuple
            )
            aux = {
                "branch_logits": branch_logits,
                "fusion_weights": w_final,
                "rcd_info": rcd_info,
            }
        elif CONFIG.get("use_final_dynamic_fusion", True):
            fused, w_final = self.final_dynamic_fusion(g, a, v)
            aux = None
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
        self.use_cheap_exit = CONFIG.get("use_cheap_exit", True)
        self.cheap_head = CheapSignalHead(num_classes=num_classes) if self.use_cheap_exit else None
        self.use_shallow_exit = CONFIG.get("use_shallow_exit", False)
        self.shallow_head = FeatureExitHead(dim=dim, num_classes=num_classes) if self.use_shallow_exit else None
        # 分支编码器
        self.encoder_g = SGLATransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )
        self.encoder_a = SGLATransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )
        self.encoder_v = SGLATransformerEncoder(
            dim, depth=transformer_depth, split_layer=transformer_split_layer, dropout=dropout
        )

        # 0. AMCF-like 原始信号滤波
        self.use_amcf = CONFIG.get("use_amcf", False)

        if self.use_amcf:
            k = CONFIG.get("amcf_filters", 4)
            kernel = CONFIG.get("amcf_kernel", 31)

            self.amcf_g = AMCFBlock(in_channels=3, num_filters=k, kernel_size=kernel)
            self.amcf_a = AMCFBlock(in_channels=3, num_filters=k, kernel_size=kernel)
            self.amcf_v = AMCFBlock(in_channels=3, num_filters=k, kernel_size=kernel)

            amcf_out_ch = 3 * k
        else:
            self.amcf_g = nn.Identity()
            self.amcf_a = nn.Identity()
            self.amcf_v = nn.Identity()

            amcf_out_ch = 3

        # 1. TemporalCNN
        if CONFIG.get("use_temporal_cnn", True):
            self.pre_g = TemporalCNN(amcf_out_ch, 32)
            self.pre_a = TemporalCNN(amcf_out_ch, 32)
            self.pre_v = TemporalCNN(amcf_out_ch, 32)
            embed_in_ch = 32
        else:
            self.pre_g = nn.Identity()
            self.pre_a = nn.Identity()
            self.pre_v = nn.Identity()
            embed_in_ch = amcf_out_ch

        use_position = CONFIG.get("use_positional_encoding", True)
        position_dropout = float(CONFIG.get("position_dropout", 0.0))
        self.embed_g = PatchEmbedding(
            embed_in_ch, dim, patch=16,
            use_positional_encoding=use_position,
            position_dropout=position_dropout,
        )
        self.embed_a = PatchEmbedding(
            embed_in_ch, dim, patch=16,
            use_positional_encoding=use_position,
            position_dropout=position_dropout,
        )
        self.embed_v = PatchEmbedding(
            embed_in_ch, dim, patch=32,
            use_positional_encoding=use_position,
            position_dropout=position_dropout,
        )

        # ===============================
        # 融合前独立三分支去噪模块
        # ===============================
        if CONFIG.get("use_prefusion_denoise", False):
            self.denoise_g = TokenDenoiseBlock(dim)
            self.denoise_a = TokenDenoiseBlock(dim)
            self.denoise_v = TokenDenoiseBlock(dim)
        else:
            self.denoise_g = nn.Identity()
            self.denoise_a = nn.Identity()
            self.denoise_v = nn.Identity()


        # ===============================
        # 跨模态注意力模块
        # ===============================
        if CONFIG.get("use_cross_residual", True):
            self.cross_g = GatedResidualCrossAttention(dim)
            self.cross_a = GatedResidualCrossAttention(dim)
            self.cross_v = GatedResidualCrossAttention(dim)

            self.cross1 = None
            self.cross2 = None
            self.cross3 = None
        else:
            self.cross_g = None
            self.cross_a = None
            self.cross_v = None

            self.cross1 = CrossModalAttention(dim)
            self.cross2 = CrossModalAttention(dim)
            self.cross3 = CrossModalAttention(dim)

        # ===============================
        # 门控融合与最终融合解耦
        # ===============================
        self.gate_fusion = DynamicFusion(dim)
        self.final_dynamic_fusion = ResidualDynamicFusion(dim)
        self.rcd_fusion = RCDynamicFusion(
            dim=dim,
            num_classes=num_classes,
            hidden=64,
            temperature=1.0
        ) if CONFIG.get("use_rcd_fusion", False) else None

        self.use_global_branch = CONFIG.get("use_global_branch", False)
        if self.use_global_branch:
            global_width = int(CONFIG.get("global_branch_width", 64))
            self.global_branch = GlobalSignalBranch(
                in_ch=9,
                dim=dim,
                width=global_width,
                dropout=max(dropout * 0.5, 0.05),
            )
            self.global_fusion = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(dim, dim),
            )
            self.global_alpha_logit = nn.Parameter(
                torch.tensor(float(CONFIG.get("global_branch_init_alpha", -1.5)), dtype=torch.float32)
            )
        else:
            self.global_branch = None
            self.global_fusion = None
            self.global_alpha_logit = None

        # 分类头
        self.simple_head = nn.Linear(dim, num_classes)

    def shallow_exit_layers(self):
        max_layers = len(self.encoder_g.base_layers)
        return max(0, min(int(CONFIG.get("shallow_exit_base_layers", max_layers)), max_layers))

    def stem_to_tokens(self, g, a, v):
        g = self.amcf_g(g)
        a = self.amcf_a(a)
        v = self.amcf_v(v)

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

    def preprocess_to_shallow_exit(self, g, a, v):
        g, a, v = self.stem_to_tokens(g, a, v)
        return self.run_base_triplet(g, a, v, end_layer=self.shallow_exit_layers())

    def complete_base_from_shallow(self, g, a, v):
        return self.run_base_triplet(g, a, v, start_layer=self.shallow_exit_layers())

    def extract_fused_feature_from_base(
        self,
        g_base,
        a_base,
        v_base,
        gate_threshold=0.25,
        force_full=False,
        sim_skip_threshold=None,
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
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )
        a_out, pa, ba, oa = self._adaptive_step_eval_per_sample(
            self.encoder_a, a_base, gate_w[:, 1], gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )
        v_out, pv, bv, ov = self._adaptive_step_eval_per_sample(
            self.encoder_v, v_base, gate_w[:, 2], gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )

        g_c, a_c, v_c = g_out, a_out, v_out
        g_c, a_c, v_c = self.apply_prefusion_denoise(g_c, a_c, v_c)
        g_c, a_c, v_c = self.apply_cross_modal(g_c, a_c, v_c)

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
        sim_skip_threshold=None,
        collect_intermediates=True,
    ):
        g_base, a_base, v_base = self.preprocess_to_base(g, a, v)
        return self.extract_fused_feature_from_base(
            g_base, a_base, v_base,
            gate_threshold=gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )

        # 1. AMCF
        g = self.amcf_g(g)
        a = self.amcf_a(a)
        v = self.amcf_v(v)

        # 2. CNN
        g, a, v = self.pre_g(g), self.pre_a(a), self.pre_v(v)

        # 3. Embedding
        g, a, v = self.embed_g(g), self.embed_a(a), self.embed_v(v)

        # 4. Base layers
        g_base = self.encoder_g.run_base(g)
        a_base = self.encoder_a.run_base(a)
        v_base = self.encoder_v.run_base(v)

        # 5. Gate weights
        w = self.get_gate_weight(g_base, a_base, v_base)

        if CONFIG.get("use_outer_gate", True):
            gate_w = w
        else:
            gate_w = torch.ones_like(w)

        # 6. Dynamic layers
        g_out, pg, bg, og = self._adaptive_step_eval_per_sample(
            self.encoder_g, g_base, gate_w[:, 0], gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )
        a_out, pa, ba, oa = self._adaptive_step_eval_per_sample(
            self.encoder_a, a_base, gate_w[:, 1], gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )
        v_out, pv, bv, ov = self._adaptive_step_eval_per_sample(
            self.encoder_v, v_base, gate_w[:, 2], gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )

        # 7. 初始三模态特征
        g_c, a_c, v_c = g_out, a_out, v_out

        # 8. 融合前独立去噪
        g_c, a_c, v_c = self.apply_prefusion_denoise(g_c, a_c, v_c)

        # 9. 残差跨模态注意力
        g_c, a_c, v_c = self.apply_cross_modal(g_c, a_c, v_c)

        # 10. 最终融合：AvgFusion / DynamicFusion / RCD-Fusion 三选一
        probs_tuple = (pg, pa, pv)
        fused, final_w, fusion_aux = self.final_fuse(g_c, a_c, v_c, probs_tuple=probs_tuple)

        return fused, w, probs_tuple, (bg, ba, bv), (og, oa, ov), fusion_aux, final_w
    def forward_features(
        self,
        g,
        a,
        v,
        gate_threshold=0.25,
        force_full=False,
        sim_skip_threshold=None,
        collect_intermediates=True,
    ):

        fused, _, _, _, _, _, _ = self.extract_fused_feature(
            g, a, v,
            gate_threshold=gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )
        return fused

    def classify_fused(self, fused, g, a, v):
        if not self.use_global_branch or self.global_branch is None:
            return self.simple_head(fused)

        raw = torch.cat([g, a, v], dim=1)
        global_feat = self.global_branch(raw)
        delta = self.global_fusion(torch.cat([fused, global_feat], dim=1))
        alpha = torch.sigmoid(self.global_alpha_logit)
        return self.simple_head(fused + alpha * delta)

    def _adaptive_step_eval_per_sample(
        self,
        encoder,
        base_feat,
        weight_vec,
        threshold,
        force_full=False,
        sim_skip_threshold=None,
        collect_intermediates=True,
    ):
        """
        只在 eval 且非 force_full 时启用真正逐样本门控。
        返回:
            out      : [B, T, C]
            probs    : [B, num_dynamic]，若全跳过则为 None
            base_feat: [B, T, C]
            outs     : list[tensor]，若全跳过则 []
        """
        # 训练阶段 or 强制满载：保持原逻辑
        if self.training or force_full:
            return encoder.forward_dynamic(
                base_feat,
                force_full=force_full,
                sim_skip_threshold=None,
                collect_intermediates=collect_intermediates,
            )

        # weight_vec: [B]
        if weight_vec.dim() != 1:
            weight_vec = weight_vec.view(-1)

        B = base_feat.size(0)
        active_mask = weight_vec >= threshold   # True 表示该样本继续进入动态层

        # 全部继续
        if active_mask.all():
            return encoder.forward_dynamic(
                base_feat,
                force_full=False,
                sim_skip_threshold=sim_skip_threshold,
                collect_intermediates=collect_intermediates,
            )

        # 全部跳过
        if (~active_mask).all():
            return base_feat, None, base_feat, []

        # 混合情况：一部分样本跳过，一部分样本进入动态层
        out = base_feat.clone()
        num_dynamic = len(encoder.dynamic_layers)
        probs_full = base_feat.new_zeros((B, num_dynamic))

        active_feat = base_feat[active_mask]  # [B_active, T, C]
        out_active, probs_active, _, outs_active = encoder.forward_dynamic(
            active_feat,
            force_full=False,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )

        # 把 active 样本的结果回填
        out[active_mask] = out_active

        # 如果 use_sgla=False，则 probs_active 为 None，不需要回填 selector 概率
        if probs_active is not None:
            probs_full[active_mask] = probs_active
        else:
            probs_full = None

        if not collect_intermediates:
            return out, probs_full, base_feat, []

        # 为了兼容你现有的返回格式，把每层输出也回填成 full batch 形式
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
        sim_skip_threshold=None,
        collect_intermediates=True,
        early_exit_threshold=None,
    ):
        cheap_logits = self.cheap_head(g, a, v) if self.cheap_head is not None else None

        if (
            cheap_logits is not None
            and early_exit_threshold is not None
            and not self.training
            and not force_full
        ):
            confidence_temperature = max(float(CONFIG.get("cheap_conf_temperature", 1.0)), 1e-6)
            cheap_probs = F.softmax(cheap_logits / confidence_temperature, dim=1)
            cheap_conf = cheap_probs.max(dim=1).values
            exit_mask = cheap_conf >= early_exit_threshold
            B = g.size(0)

            if exit_mask.all():
                gate_w = torch.zeros(B, 3, device=g.device, dtype=cheap_logits.dtype)
                aux = {
                    "cheap_logits": cheap_logits,
                    "cheap_confidence": cheap_conf,
                    "exit_mask": exit_mask,
                }
                return cheap_logits, aux, gate_w, (None, None, None), (None, None, None), ([], [], [])

            if exit_mask.any():
                logits = cheap_logits.clone()
                gate_w = torch.zeros(B, 3, device=g.device, dtype=cheap_logits.dtype)

                full_idx = ~exit_mask
                fused, gate_w_full, probs_tuple, bases_tuple, outs_tuple, fusion_aux, final_w = self.extract_fused_feature(
                    g[full_idx], a[full_idx], v[full_idx],
                    gate_threshold=gate_threshold,
                    force_full=force_full,
                    sim_skip_threshold=sim_skip_threshold,
                    collect_intermediates=False,
                )
                logits[full_idx] = self.classify_fused(
                    fused,
                    g[full_idx],
                    a[full_idx],
                    v[full_idx],
                )
                gate_w[full_idx] = gate_w_full
                aux = dict(fusion_aux) if isinstance(fusion_aux, dict) else {}
                aux.update({
                    "cheap_logits": cheap_logits,
                    "cheap_confidence": cheap_conf,
                    "exit_mask": exit_mask,
                    "early_exit_rate": exit_mask.float().mean(),
                })
                return logits, aux, gate_w, (None, None, None), (None, None, None), ([], [], [])

        if self.shallow_head is not None:
            g_exit, a_exit, v_exit = self.preprocess_to_shallow_exit(g, a, v)
            shallow_logits = self.shallow_head(g_exit, a_exit, v_exit)

            if early_exit_threshold is not None and not self.training and not force_full:
                confidence_temperature = max(float(CONFIG.get("shallow_conf_temperature", 1.0)), 1e-6)
                shallow_probs = F.softmax(shallow_logits / confidence_temperature, dim=1)
                shallow_conf = shallow_probs.max(dim=1).values
                exit_mask = shallow_conf >= early_exit_threshold
                B = g.size(0)

                if exit_mask.all():
                    gate_w = torch.zeros(B, 3, device=g.device, dtype=shallow_logits.dtype)
                    aux = {
                        "shallow_logits": shallow_logits,
                        "shallow_confidence": shallow_conf,
                        "exit_mask": exit_mask,
                        "early_exit_rate": exit_mask.float().mean(),
                    }
                    if cheap_logits is not None:
                        aux["cheap_logits"] = cheap_logits
                    return shallow_logits, aux, gate_w, (None, None, None), (None, None, None), ([], [], [])

                if exit_mask.any():
                    logits = shallow_logits.clone()
                    gate_w = torch.zeros(B, 3, device=g.device, dtype=shallow_logits.dtype)
                    full_idx = ~exit_mask
                    g_base, a_base, v_base = self.complete_base_from_shallow(
                        g_exit[full_idx], a_exit[full_idx], v_exit[full_idx]
                    )
                    fused, gate_w_full, probs_tuple, bases_tuple, outs_tuple, fusion_aux, final_w = self.extract_fused_feature_from_base(
                        g_base, a_base, v_base,
                        gate_threshold=gate_threshold,
                        force_full=force_full,
                        sim_skip_threshold=sim_skip_threshold,
                        collect_intermediates=False,
                    )
                    logits[full_idx] = self.classify_fused(
                        fused,
                        g[full_idx],
                        a[full_idx],
                        v[full_idx],
                    )
                    gate_w[full_idx] = gate_w_full
                    aux = dict(fusion_aux) if isinstance(fusion_aux, dict) else {}
                    aux.update({
                        "shallow_logits": shallow_logits,
                        "shallow_confidence": shallow_conf,
                        "exit_mask": exit_mask,
                        "early_exit_rate": exit_mask.float().mean(),
                    })
                    if cheap_logits is not None:
                        aux["cheap_logits"] = cheap_logits
                    return logits, aux, gate_w, (None, None, None), (None, None, None), ([], [], [])

            g_base, a_base, v_base = self.complete_base_from_shallow(g_exit, a_exit, v_exit)
            fused, gate_w, probs_tuple, bases_tuple, outs_tuple, fusion_aux, final_w = self.extract_fused_feature_from_base(
                g_base, a_base, v_base,
                gate_threshold=gate_threshold,
                force_full=force_full,
                sim_skip_threshold=sim_skip_threshold,
                collect_intermediates=collect_intermediates,
            )

            logits = self.classify_fused(fused, g, a, v)
            aux = dict(fusion_aux) if isinstance(fusion_aux, dict) else {}
            aux["shallow_logits"] = shallow_logits
            if cheap_logits is not None:
                aux["cheap_logits"] = cheap_logits
            return logits, aux, gate_w, probs_tuple, bases_tuple, outs_tuple

        fused, gate_w, probs_tuple, bases_tuple, outs_tuple, fusion_aux, final_w = self.extract_fused_feature(
            g, a, v,
            gate_threshold=gate_threshold,
            force_full=force_full,
            sim_skip_threshold=sim_skip_threshold,
            collect_intermediates=collect_intermediates,
        )

        logits = self.classify_fused(fused, g, a, v)
        aux = dict(fusion_aux) if isinstance(fusion_aux, dict) else {}
        if cheap_logits is not None:
            aux["cheap_logits"] = cheap_logits
        if not aux:
            aux = None

        # 第三个返回值保持为 outer gate 权重，保证动态层使用统计逻辑不变。
        # RCD 的最终融合权重保存在 aux["fusion_weights"] 中。
        return logits, aux, gate_w, probs_tuple, bases_tuple, outs_tuple

# ==========================================
# 4. 🔥 强化版 Hybrid Loss (SGLA相似度 )
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
        # 🚀 修正点1：定义类别权重 (0-Normal, 1-Accel, 2-GPS, 3-Gyro, 4-Mag, 5-Baro)
        # 给 GPS (2) 和 Mag (4) 设置更高的权重 (例如 2.0)
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        # 使用 register_buffer，这样 weights 会自动随模型移动到 GPU/CPU
        self.register_buffer('main_class_weights', weights)



    def sgla_loss(self, probs, base_feat, layer_outs):
        if probs is None or len(layer_outs) < 1:
            return torch.tensor(0.0, device=base_feat.device)

        # 🚀 关键修改：构建包含基础层和动态层的完整列表
        # 列表内容：[Base, Dynamic_1, Dynamic_2, ..., Dynamic_N]
        all_layers = [base_feat] + layer_outs
        adj_sims = []

        # 计算相邻两层之间的相似度
        for i in range(1, len(all_layers)):
            prev_feat = all_layers[i - 1].mean(dim=1)  # 前一层
            curr_feat = all_layers[i].mean(dim=1)  # 当前层
            sim = F.cosine_similarity(prev_feat, curr_feat, dim=1)
            adj_sims.append(sim)

        adj_sims = torch.stack(adj_sims, dim=1)  # [Batch, Num_Dynamic_Layers]

        # 1. 引导选择器：相似度越高，被选择退出的概率应该越大
        # 我们使用 softmax 归一化相似度作为目标分布
        target_probs = F.softmax(adj_sims / 0.1, dim=1)
        kl_loss = F.kl_div(probs.clamp(min=1e-7).log(), target_probs, reduction='batchmean')

        # 2. 🚀 强制饱和引导 (关键项)：强迫相邻层长得越来越像
        # 让相似度往 1.0 靠拢
        sat_loss = F.mse_loss(adj_sims, torch.ones_like(adj_sims))

        return kl_loss + 0.5 * sat_loss  # 0.5 是引导强度

    def forward(self, main_logits, aux_outputs, targets, probs_tuple, bases_tuple, outs_tuple, teacher_logits=None):
        # 1. 🚀 主分类 Loss：显式传入 GPU 上的权重
        weight = self.main_class_weights if CONFIG.get("use_class_weight", True) else None

        loss_main = F.cross_entropy(
            main_logits,
            targets,
            weight=weight,
            label_smoothing=CONFIG.get("label_smoothing", 0.05)
        )




        # 2. SGLA 相似度 Loss
        loss_sim = torch.tensor(0.0, device=targets.device)
        if CONFIG.get("use_sgla", True) and CONFIG.get("use_sgla_loss", True) and probs_tuple is not None:
            for i in range(3):
                p, b, o = probs_tuple[i], bases_tuple[i], outs_tuple[i]
                if p is not None and o is not None and len(o) > 0:
                    if p.shape[1] == len(o):
                        loss_sim += self.sgla_loss(p, b, o)

        # 3. RCD-Fusion 单模态辅助分类损失与融合权重批均衡正则
        loss_aux = torch.tensor(0.0, device=targets.device)
        loss_balance = torch.tensor(0.0, device=targets.device)
        loss_cheap = torch.tensor(0.0, device=targets.device)
        loss_cheap_kd = torch.tensor(0.0, device=targets.device)
        loss_shallow = torch.tensor(0.0, device=targets.device)
        loss_shallow_kd = torch.tensor(0.0, device=targets.device)
        loss_teacher_kd = torch.tensor(0.0, device=targets.device)
        if teacher_logits is not None and float(CONFIG.get("teacher_kd_weight", 0.0)) > 0:
            temperature = max(float(CONFIG.get("teacher_kd_temperature", 3.0)), 1e-6)
            teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
            student_log_probs = F.log_softmax(main_logits / temperature, dim=1)
            loss_teacher_kd = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (temperature ** 2)

        if aux_outputs is not None and "shallow_logits" in aux_outputs:
            shallow_logits = aux_outputs["shallow_logits"]
            loss_shallow = F.cross_entropy(
                shallow_logits,
                targets,
                weight=weight,
                label_smoothing=CONFIG.get("label_smoothing", 0.05),
            )
            if float(CONFIG.get("shallow_kd_weight", 0.0)) > 0:
                temperature = max(float(CONFIG.get("shallow_kd_temperature", 2.0)), 1e-6)
                teacher_probs = F.softmax(main_logits.detach() / temperature, dim=1)
                student_log_probs = F.log_softmax(shallow_logits / temperature, dim=1)
                loss_shallow_kd = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature ** 2)
        if aux_outputs is not None and "cheap_logits" in aux_outputs:
            cheap_logits = aux_outputs["cheap_logits"]
            loss_cheap = F.cross_entropy(
                cheap_logits,
                targets,
                weight=weight,
                label_smoothing=CONFIG.get("label_smoothing", 0.05),
            )
            if float(CONFIG.get("cheap_kd_weight", 0.0)) > 0:
                temperature = max(float(CONFIG.get("cheap_kd_temperature", 2.0)), 1e-6)
                teacher_probs = F.softmax(main_logits.detach() / temperature, dim=1)
                student_log_probs = F.log_softmax(cheap_logits / temperature, dim=1)
                loss_cheap_kd = F.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction="batchmean",
                ) * (temperature ** 2)

        if CONFIG.get("use_rcd_fusion", False) and aux_outputs is not None:
            branch_logits = aux_outputs.get("branch_logits", None)
            if branch_logits is not None:
                loss_aux = (
                    F.cross_entropy(
                        branch_logits["g"], targets, weight=weight,
                        label_smoothing=CONFIG.get("label_smoothing", 0.05)
                    ) +
                    F.cross_entropy(
                        branch_logits["a"], targets, weight=weight,
                        label_smoothing=CONFIG.get("label_smoothing", 0.05)
                    ) +
                    F.cross_entropy(
                        branch_logits["v"], targets, weight=weight,
                        label_smoothing=CONFIG.get("label_smoothing", 0.05)
                    )
                ) / 3.0

            fusion_weights = aux_outputs.get("fusion_weights", None)
            if fusion_weights is not None:
                mean_alpha = fusion_weights.mean(dim=0)
                target_alpha = torch.ones_like(mean_alpha) / mean_alpha.numel()
                loss_balance = F.mse_loss(mean_alpha, target_alpha)

        return (
            loss_main
            + 0.01 * (loss_sim / 3.0)
            + float(CONFIG.get("rcd_aux_weight", 0.20)) * loss_aux
            + float(CONFIG.get("rcd_balance_weight", 0.01)) * loss_balance
            + float(CONFIG.get("cheap_aux_weight", 0.0)) * loss_cheap
            + float(CONFIG.get("cheap_kd_weight", 0.0)) * loss_cheap_kd
            + float(CONFIG.get("shallow_aux_weight", 0.0)) * loss_shallow
            + float(CONFIG.get("shallow_kd_weight", 0.0)) * loss_shallow_kd
            + float(CONFIG.get("teacher_kd_weight", 0.0)) * loss_teacher_kd
        )

############################################
# 训练曲线、验证与 t-SNE
############################################
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


def build_external_teacher(device):
    teacher_model_name = str(CONFIG.get("teacher_model", "")).strip().lower()
    teacher_path = str(CONFIG.get("teacher_path", "")).strip()
    kd_weight = float(CONFIG.get("teacher_kd_weight", 0.0))
    if not teacher_model_name or not teacher_path or kd_weight <= 0:
        return None

    repo_root = os.path.dirname(os.path.abspath(__file__))
    paper_dir = os.path.join(repo_root, "paper_experiments")
    for path in (repo_root, paper_dir):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        from baseline_models import build_baseline
    except Exception as exc:
        raise RuntimeError(f"Failed to import teacher baseline model: {exc}") from exc

    teacher = build_baseline(teacher_model_name, num_classes=CONFIG["num_classes"]).to(device)
    checkpoint = torch.load(teacher_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    teacher.load_state_dict(checkpoint, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    print(
        f"Loaded teacher model for KD: model={teacher_model_name}, "
        f"path={teacher_path}, weight={kd_weight}, "
        f"T={CONFIG.get('teacher_kd_temperature', 3.0)}"
    )
    return teacher


def apply_mixup_batch(g, a, v, y, alpha=0.2):
    if alpha <= 0 or g.size(0) < 2:
        return g, a, v, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(g.size(0), device=g.device)
    mixed_g = lam * g + (1.0 - lam) * g[index]
    mixed_a = lam * a + (1.0 - lam) * a[index]
    mixed_v = lam * v + (1.0 - lam) * v[index]
    return mixed_g, mixed_a, mixed_v, y, y[index], lam


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


def evaluate_cheap_head(model, loader, device):
    if not hasattr(model, "cheap_head") or model.cheap_head is None:
        return None
    model.eval()
    correct, total = 0, 0
    confidences = []
    with torch.no_grad():
        for g, a, v, y in loader:
            g, a, v, y = g.to(device), a.to(device), v.to(device), y.to(device)
            logits = model.cheap_head(g, a, v)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            confidences.append(conf.detach().cpu())

    if total == 0:
        return {"accuracy": 0.0, "mean_confidence": 0.0}
    confidence = torch.cat(confidences).mean().item() if confidences else 0.0
    return {
        "accuracy": 100.0 * correct / total,
        "mean_confidence": 100.0 * confidence,
    }


def evaluate_shallow_head(model, loader, device):
    if not hasattr(model, "shallow_head") or model.shallow_head is None:
        return None
    model.eval()
    correct, total = 0, 0
    confidences = []
    with torch.no_grad():
        for g, a, v, y in loader:
            g, a, v, y = g.to(device), a.to(device), v.to(device), y.to(device)
            g_exit, a_exit, v_exit = model.preprocess_to_shallow_exit(g, a, v)
            logits = model.shallow_head(g_exit, a_exit, v_exit)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            confidences.append(conf.detach().cpu())

    if total == 0:
        return {"accuracy": 0.0, "mean_confidence": 0.0}
    confidence = torch.cat(confidences).mean().item() if confidences else 0.0
    return {
        "accuracy": 100.0 * correct / total,
        "mean_confidence": 100.0 * confidence,
    }


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
                    gate_threshold=0.25, force_full=False, sim_skip_threshold=None,
                    early_exit_threshold=None):
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
                sim_skip_threshold=sim_skip_threshold,
                collect_intermediates=False,
                early_exit_threshold=early_exit_threshold,
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
                        sim_skip_threshold=sim_skip_threshold,
                        collect_intermediates=False,
                        early_exit_threshold=early_exit_threshold,
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
                        sim_skip_threshold=sim_skip_threshold,
                        collect_intermediates=False,
                        early_exit_threshold=early_exit_threshold,
                    )
                    processed_samples += int(g.size(0))
                end = time.perf_counter()
                timings.append((end - start, processed_samples))

    fps_list = [samples / t for t, samples in timings]
    latency_ms_list = [(t / samples) * 1000 for t, samples in timings]

    print("\n" + "=" * 50)
    print(
        f"gate_threshold = {gate_threshold}, force_full = {force_full}, "
        f"sim_skip_threshold = {sim_skip_threshold}, early_exit_threshold = {early_exit_threshold}"
    )
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


def evaluate_fast_mode(
    model,
    dataloader,
    device,
    gate_threshold,
    sim_skip_threshold=None,
    early_exit_threshold=None,
    max_batches=0,
):
    model.eval()
    all_preds = []
    all_labels = []
    exit_count = 0
    sample_count = 0

    with torch.inference_mode():
        for batch_idx, (g, a, v, y) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            g, a, v = g.to(device), a.to(device), v.to(device)
            logits, aux, _, _, _, _ = model(
                g, a, v,
                gate_threshold=gate_threshold,
                force_full=False,
                sim_skip_threshold=sim_skip_threshold,
                early_exit_threshold=early_exit_threshold,
                collect_intermediates=False,
            )
            if aux is not None and isinstance(aux, dict) and "exit_mask" in aux:
                exit_count += int(aux["exit_mask"].sum().item())
            sample_count += int(y.size(0))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )[2]

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "exit_rate": float(exit_count / max(sample_count, 1)),
    }


def run_fast_mode_sweep(model, val_loader, test_loader, device, save_dir):
    gate_thresholds = CONFIG.get("fast_gate_thresholds", [CONFIG.get("gate_threshold", 0.25)])
    sim_thresholds = CONFIG.get("sim_skip_thresholds", [])
    exit_thresholds = CONFIG.get("fast_exit_thresholds", [])
    max_batches = int(CONFIG.get("fast_eval_max_batches", 0) or 0)

    rows = []
    candidates = []
    for gate_threshold in gate_thresholds:
        candidates.append((gate_threshold, None, None))
        for sim_threshold in sim_thresholds:
            candidates.append((gate_threshold, sim_threshold, None))
        for exit_threshold in exit_thresholds:
            candidates.append((gate_threshold, None, exit_threshold))

    print("\n" + "=" * 50)
    print("Fast inference sweep")
    print(f"gate_thresholds = {gate_thresholds}")
    print(f"sim_skip_thresholds = {sim_thresholds}")
    print(f"early_exit_thresholds = {exit_thresholds}")
    print(f"fast_eval_max_batches = {max_batches if max_batches else 'full'}")
    print("=" * 50)

    for gate_threshold, sim_threshold, exit_threshold in candidates:
        val_metrics = evaluate_fast_mode(
            model,
            val_loader,
            device,
            gate_threshold=gate_threshold,
            sim_skip_threshold=sim_threshold,
            early_exit_threshold=exit_threshold,
            max_batches=max_batches,
        )
        test_metrics = evaluate_fast_mode(
            model,
            test_loader,
            device,
            gate_threshold=gate_threshold,
            sim_skip_threshold=sim_threshold,
            early_exit_threshold=exit_threshold,
            max_batches=max_batches,
        )
        fps_stats = test_fps(
            model,
            device,
            val_loader=val_loader,
            repeat=3,
            warmup=50,
            iters=300,
            gate_threshold=gate_threshold,
            force_full=False,
            sim_skip_threshold=sim_threshold,
            early_exit_threshold=exit_threshold,
        )

        row = {
            "gate_threshold": gate_threshold,
            "sim_skip_threshold": sim_threshold if sim_threshold is not None else "",
            "early_exit_threshold": exit_threshold if exit_threshold is not None else "",
            "val_exit_rate": val_metrics["exit_rate"],
            "test_exit_rate": test_metrics["exit_rate"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "fps_mean": fps_stats["fps_mean"],
            "fps_std": fps_stats["fps_std"],
            "latency_mean_ms": fps_stats["latency_mean_ms"],
            "latency_std_ms": fps_stats["latency_std_ms"],
            "benchmark_samples_per_repeat": fps_stats.get("benchmark_samples_per_repeat", 0),
        }
        rows.append(row)
        print(
            "Fast mode "
            f"gate={gate_threshold}, sim={sim_threshold}, exit={exit_threshold}: "
            f"val_acc={row['val_accuracy'] * 100:.2f}%, "
            f"test_acc={row['test_accuracy'] * 100:.2f}%, "
            f"test_exit={row['test_exit_rate'] * 100:.1f}%, "
            f"fps={row['fps_mean']:.2f}"
        )

    df = pd.DataFrame(rows)
    save_path = os.path.join(save_dir, "fast_mode_sweep.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Fast mode sweep saved to: {save_path}")
    print(df)
    return df
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
            # 优先可视化 RCD-Fusion 的最终融合权重；若不是 RCD，则退回 outer gate 权重。
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

    # 3. 针对 SGLA 的深度分析（创新点体现）
    # 假设 SGLA 运行 3 层和运行 6 层的差异
    print("-" * 50)
    print("💡 SGLA 边缘计算优化潜力:")
    print(f"   - 满载模式 (6层): {macs / 1e6:.2f} M FLOPs")
    # 粗略估算：Encoder 部分占据了约 80% 的计算量，跳过一半动态层约节省 30-40% 总计算量
    print(f"   - 节能模式 (3层): 约 {(macs / 1e6) * 0.6:.2f} M FLOPs (节省 ~40%)")
    print("=" * 50 + "\n")

    return total_params, macs
############################################
# 主流程
############################################
def main():
    if "SGLA_EXP_NAME" in os.environ:
        CONFIG["exp_name"] = os.environ["SGLA_EXP_NAME"]
    apply_ablation_config(CONFIG["exp_name"])
    apply_runtime_overrides()
    set_seed(CONFIG["seed"])
    print(f"Device: {device}")
    print(
        "Runtime config: "
        f"epochs={CONFIG['epochs']}, batch_size={CONFIG['batch_size']}, "
        f"lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']}, "
        f"dropout={CONFIG.get('model_dropout', 0.2)}, "
        f"transformer_depth={CONFIG.get('transformer_depth', 5)}, "
        f"transformer_split_layer={CONFIG.get('transformer_split_layer', 3)}, "
        f"positional_encoding={CONFIG.get('use_positional_encoding', True)}, "
        f"position_dropout={CONFIG.get('position_dropout', 0.0)}, "
        f"label_smoothing={CONFIG.get('label_smoothing', 0.05)}, "
        f"use_ema={CONFIG.get('use_ema', True)}, "
        f"use_cheap_exit={CONFIG.get('use_cheap_exit', True)}, "
        f"cheap_aux_weight={CONFIG.get('cheap_aux_weight', 0.0)}, "
        f"cheap_kd_weight={CONFIG.get('cheap_kd_weight', 0.0)}, "
        f"cheap_kd_temperature={CONFIG.get('cheap_kd_temperature', 2.0)}, "
        f"cheap_conf_temperature={CONFIG.get('cheap_conf_temperature', 1.0)}, "
        f"cheap_head_width={CONFIG.get('cheap_head_width', 48)}, "
        f"train_cheap_only={CONFIG.get('train_cheap_only', False)}, "
        f"use_shallow_exit={CONFIG.get('use_shallow_exit', False)}, "
        f"train_shallow_only={CONFIG.get('train_shallow_only', False)}, "
        f"train_fusion_head_only={CONFIG.get('train_fusion_head_only', False)}, "
        f"shallow_exit_base_layers={CONFIG.get('shallow_exit_base_layers', 3)}, "
        f"shallow_aux_weight={CONFIG.get('shallow_aux_weight', 0.0)}, "
        f"shallow_kd_weight={CONFIG.get('shallow_kd_weight', 0.0)}, "
        f"use_mixup={CONFIG.get('use_mixup', False)}, "
        f"mixup_alpha={CONFIG.get('mixup_alpha', 0.0)}, "
        f"use_global_branch={CONFIG.get('use_global_branch', False)}, "
        f"global_branch_width={CONFIG.get('global_branch_width', 64)}, "
        f"teacher_model={CONFIG.get('teacher_model', '') or 'none'}, "
        f"teacher_kd_weight={CONFIG.get('teacher_kd_weight', 0.0)}, "
        f"aug_time_shift_max={CONFIG.get('aug_time_shift_max', 0)}, "
        f"patience={CONFIG.get('patience', 20)}, "
        f"clean_test_only={CONFIG.get('clean_test_only', False)}, "
        f"gate_thresholds={CONFIG.get('fast_gate_thresholds', [])}, "
        f"sim_skip_thresholds={CONFIG.get('sim_skip_thresholds', [])}, "
        f"fast_exit_thresholds={CONFIG.get('fast_exit_thresholds', [])}, "
        f"init_weights={CONFIG.get('init_weights', '') or 'none'}, "
        f"data_root={CONFIG['data_root']}, save_dir={CONFIG['save_dir']}"
    )

    # ============================================================
    # 先创建完整数据集，用于获取样本总数和建立固定划分
    # 这里只读取数据，不启用训练增强
    # ============================================================
    full_dataset = SensorDataset(CONFIG["data_root"], is_train=False)
    num_samples = len(full_dataset)

    # ============================================================
    # 固定 80/10/10 分层划分：Train / Validation / Test
    # train：训练模型；val：早停与调参；test：最终独立评估，只在训练结束后使用。
    # 如果 split_dir 已有 train_indices.npy / val_indices.npy / test_indices.npy，则直接复用。
    # ============================================================
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
        print(f"✅ 已加载固定 80/10/10 数据划分: {CONFIG['split_dir']}")
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

        print(f"✅ 首次生成并保存固定 80/10/10 分层数据划分: {CONFIG['split_dir']}")

    assert len(set(train_indices).intersection(set(val_indices))) == 0, "训练集和验证集索引有重叠！"
    assert len(set(train_indices).intersection(set(test_indices))) == 0, "训练集和测试集索引有重叠！"
    assert len(set(val_indices).intersection(set(test_indices))) == 0, "验证集和测试集索引有重叠！"
    assert len(train_indices) + len(val_indices) + len(test_indices) == num_samples, "三划分样本总数不等于数据集总数！"

    def print_label_distribution(name, indices):
        y = labels_all[indices]
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"{name} label distribution: {dist}")

    print(f"数据总数: {num_samples}")
    print(f"训练集数量: {len(train_indices)}")
    print(f"验证集数量: {len(val_indices)}")
    print(f"测试集数量: {len(test_indices)}")
    print(f"训练集比例: {len(train_indices) / num_samples * 100:.2f}%")
    print(f"验证集比例: {len(val_indices) / num_samples * 100:.2f}%")
    print(f"测试集比例: {len(test_indices) / num_samples * 100:.2f}%")
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

    # ============================================================
    # 根据固定索引建立 Train / Validation 数据集
    # 只有训练集开启数据增强
    # ============================================================
    train_set = torch.utils.data.Subset(
        SensorDataset(CONFIG["data_root"], is_train=True, channel_mean=channel_mean, channel_std=channel_std),
        train_indices
    )
    val_set = torch.utils.data.Subset(
        SensorDataset(CONFIG["data_root"], is_train=False, channel_mean=channel_mean, channel_std=channel_std),
        val_indices
    )
    test_set = torch.utils.data.Subset(
        SensorDataset(CONFIG["data_root"], is_train=False, channel_mean=channel_mean, channel_std=channel_std),
        test_indices
    )

    train_loader = DataLoader(
        train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )
    test_loader = DataLoader(
        test_set,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

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
    teacher_model = build_external_teacher(device)
    if CONFIG.get("train_cheap_only", False) and CONFIG.get("train_shallow_only", False):
        raise ValueError("Choose only one of SGLA_TRAIN_CHEAP_ONLY=1 or SGLA_TRAIN_SHALLOW_ONLY=1.")
    if CONFIG.get("train_cheap_only", False):
        if model.cheap_head is None:
            raise ValueError("SGLA_TRAIN_CHEAP_ONLY=1 requires SGLA_USE_CHEAP_EXIT=1.")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("cheap_head."))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"Training cheap_head only: trainable={trainable_count:,} / total={total_count:,} parameters")
    elif CONFIG.get("train_shallow_only", False):
        if model.shallow_head is None:
            raise ValueError("SGLA_TRAIN_SHALLOW_ONLY=1 requires SGLA_USE_SHALLOW_EXIT=1.")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("shallow_head."))
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(f"Training shallow_head only: trainable={trainable_count:,} / total={total_count:,} parameters")
    elif CONFIG.get("train_fusion_head_only", False):
        trainable_prefixes = (
            "simple_head.",
            "rcd_fusion.",
            "gate_fusion.",
            "final_dynamic_fusion.",
            "global_branch.",
            "global_fusion.",
        )
        trainable_names = {"global_alpha_logit"}
        for name, param in model.named_parameters():
            should_train = name in trainable_names or name.startswith(trainable_prefixes)
            param.requires_grad_(should_train)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        print(
            "Training fusion/head only: "
            f"trainable={trainable_count:,} / total={total_count:,} parameters"
        )
    else:
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    selection_metric = "val_loss" if (CONFIG.get("train_cheap_only", False) or CONFIG.get("train_shallow_only", False)) else "val_acc"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min' if selection_metric == "val_loss" else 'max',
        factor=0.5,
        patience=10
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience = int(CONFIG.get("patience", 20))  # 容忍若干个 Epoch 验证集不提升
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    if init_weights:
        init_val_model = ema_model if ema_model is not None else model
        init_val_loss, init_val_acc = validate(init_val_model, val_loader, criterion)
        best_val_acc = init_val_acc
        best_val_loss = init_val_loss
        torch.save(init_val_model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
        print(
            f"Init checkpoint validation: loss={init_val_loss:.4f}, "
            f"Val Acc={init_val_acc:.2f}% (saved as initial best)"
        )

    print("🚀 开始 SGLA 多模态动态网络训练 ...")

    # 🔥 融合后的单一训练循环
    for epoch in range(CONFIG["epochs"]):
        if (
            CONFIG.get("train_cheap_only", False)
            or CONFIG.get("train_shallow_only", False)
            or CONFIG.get("train_fusion_head_only", False)
        ):
            model.eval()
            if CONFIG.get("train_cheap_only", False) and model.cheap_head is not None:
                model.cheap_head.train()
            if CONFIG.get("train_shallow_only", False) and model.shallow_head is not None:
                model.shallow_head.train()
            if CONFIG.get("train_fusion_head_only", False):
                for module_name in (
                    "simple_head",
                    "rcd_fusion",
                    "gate_fusion",
                    "final_dynamic_fusion",
                    "global_branch",
                    "global_fusion",
                ):
                    module = getattr(model, module_name, None)
                    if module is not None:
                        module.train()
        else:
            model.train()
        t_loss, t_correct, t_total = 0, 0, 0

        # 1. 训练阶段
        for g, a, v, y in train_loader:
            g, a, v, y = g.to(device), a.to(device), v.to(device), y.to(device)
            optimizer.zero_grad()

            use_mixup_batch = (
                CONFIG.get("use_mixup", False)
                and np.random.random() < float(CONFIG.get("mixup_prob", 0.5))
            )
            if use_mixup_batch:
                g_in, a_in, v_in, y_a, y_b, lam = apply_mixup_batch(
                    g, a, v, y,
                    alpha=float(CONFIG.get("mixup_alpha", 0.2)),
                )
            else:
                g_in, a_in, v_in, y_a, y_b, lam = g, a, v, y, y, 1.0

            logits, aux, w, probs_t, bases_t, outs_t = model(g_in, a_in, v_in)
            teacher_logits = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(torch.cat([g_in, a_in, v_in], dim=1))
            if use_mixup_batch:
                loss = (
                    lam * criterion(logits, aux, y_a, probs_t, bases_t, outs_t, teacher_logits=teacher_logits)
                    + (1.0 - lam) * criterion(logits, aux, y_b, probs_t, bases_t, outs_t, teacher_logits=teacher_logits)
                )
            else:
                loss = criterion(logits, aux, y, probs_t, bases_t, outs_t, teacher_logits=teacher_logits)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema_model(
                    ema_model,
                    model,
                    decay=float(CONFIG.get("ema_decay", 0.995)),
                )

            t_loss += loss.item()
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += y.size(0)

        train_loss = t_loss / len(train_loader)
        train_acc = 100 * t_correct / t_total

        # 2. 验证阶段
        val_model = ema_model if ema_model is not None else model
        val_loss, val_acc = validate(val_model, val_loader, criterion)
        cheap_val_metrics = (
            evaluate_cheap_head(val_model, val_loader, device)
            if CONFIG.get("train_cheap_only", False)
            else None
        )
        shallow_val_metrics = (
            evaluate_shallow_head(val_model, val_loader, device)
            if CONFIG.get("train_shallow_only", False)
            else None
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 3. 学习率步进更新
        scheduler.step(val_loss if selection_metric == "val_loss" else val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:03d} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | LR: {current_lr:.6f} | 权重[G:{w[:, 0].mean():.2f}, A:{w[:, 1].mean():.2f}, V:{w[:, 2].mean():.2f}]")
        if cheap_val_metrics is not None:
            print(
                f"  Cheap head | Val Acc {cheap_val_metrics['accuracy']:.2f}% | "
                f"Mean Conf {cheap_val_metrics['mean_confidence']:.2f}%"
            )
        if shallow_val_metrics is not None:
            print(
                f"  Shallow head | Val Acc {shallow_val_metrics['accuracy']:.2f}% | "
                f"Mean Conf {shallow_val_metrics['mean_confidence']:.2f}%"
            )

        # 4. 早停与最佳模型保存逻辑
        if selection_metric == "val_loss":
            improved = val_loss < best_val_loss - 1e-4
        else:
            improved = val_acc > best_val_acc

        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, val_loss)
            torch.save(val_model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
            counter = 0
            if selection_metric == "val_loss":
                print(f"  🌟 exit_head 验证损失下降，已保存！(Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
            else:
                print(f"  🌟 发现更好模型，已保存！(Val Acc: {val_acc:.2f}%)")
        else:
            counter += 1
            if counter >= patience:
                print(f"\n🛑 早停机制触发！模型在连续 {patience} 个 Epoch 中验证集未提升。")
                print(f"最佳验证集准确率锁定在: {best_val_acc:.2f}%")
                if selection_metric == "val_loss":
                    print(f"最佳 exit_head 验证损失锁定在: {best_val_loss:.4f}")
                break

    # ==========================================
    # 训练结束，加载最佳模型进行评估和绘图
    # ==========================================
    if selection_metric == "val_loss":
        print(f"\n✅ 训练完成! 最佳准确率: {best_val_acc:.2f}%, 最佳 exit_head 验证损失: {best_val_loss:.4f}")
    else:
        print(f"\n✅ 训练完成! 最佳准确率: {best_val_acc:.2f}%")
    print("开始生成最终报告和可视化图表...")

    # 绘图和评估函数调用
    plot_history(history, CONFIG["save_dir"])
    model.load_state_dict(
        torch.load(
            os.path.join(CONFIG["save_dir"], "best_model.pth"),
            map_location=device
        )
    )

    if CONFIG.get("clean_test_only", False):
        test_eval_dir = os.path.join(CONFIG["save_dir"], "test_eval_full_clean")
        os.makedirs(test_eval_dir, exist_ok=True)
        print("Running the single full-load clean test evaluation.")
        evaluate_metrics(model, test_loader, device, test_eval_dir)
        return

    if not CONFIG.get("run_final_reports", True):
        print("Skipping final reports because SGLA_RUN_FINAL_REPORTS=0.")
        return

    report_model_complexity(model, device)

    # ================= Validation Evaluation =================
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
        model,
        val_loader,
        device,
        val_eval_dir,
        threshold=CONFIG["gate_threshold"]
    )

    # ================= Test Evaluation =================
    # 测试集只在训练结束、最佳验证模型确定后评估一次。
    test_eval_dir = os.path.join(CONFIG["save_dir"], "test_eval")
    os.makedirs(test_eval_dir, exist_ok=True)
    print("\n================ Test Evaluation ================")
    evaluate_detailed_performance(model, test_loader, device, test_eval_dir)
    plot_confusion_matrix(model, test_loader, device, test_eval_dir)
    visualize_modal_weights(model, test_loader, device, test_eval_dir)
    evaluate_metrics(model, test_loader, device, test_eval_dir)
    analyze_dynamic_usage(
        model,
        test_loader,
        device,
        test_eval_dir,
        threshold=CONFIG["gate_threshold"]
    )

    fast_eval_dir = os.path.join(CONFIG["save_dir"], "fast_eval")
    os.makedirs(fast_eval_dir, exist_ok=True)
    run_fast_mode_sweep(model, val_loader, test_loader, device, fast_eval_dir)

    test_fps(
        model,
        device,
        val_loader=val_loader,
        gate_threshold=CONFIG["gate_threshold"],
        force_full=False
    )
    test_fps(
        model,
        device,
        val_loader=val_loader,
        gate_threshold=CONFIG["gate_threshold"],
        force_full=True
    )
if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    main()

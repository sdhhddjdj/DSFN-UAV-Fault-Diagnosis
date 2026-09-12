"""Run the manuscript training implementation using the reported row names.

The implementation preserves parameter-initialization order, the initial diagnostic
forward pass, EMA, and checkpoint selection. The compact model is used for
checkpoint evaluation; removing unused modules would change same-seed training.
"""
import copy
import json
import os

from paper_protocol import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR, validate_paper_data

REFERENCE_CONFIGS = {
    "Paper_A0_StaticAvg": "Paper_A0_StaticAvg",
    "Paper_A1_DynamicDepth": "Paper_A1_DynamicDepth",
    "Paper_A2_DynamicDepthLoss": "Paper_A2_DynamicDepthLoss",
    "Paper_A3_GTD": "Paper_A4_Denoise",
    "Paper_A4_DSFN": "Paper_A5_RCD_Fusion",
    "Paper_B0_AvgFusion": "Paper_A4_Denoise",
    "Paper_B1_Reliability": "Paper_R_B1_Reliability",
    "Paper_B2_ReliabilityConsistency": "Paper_R_B2_ReliabilityConsistency",
    "Paper_B3_ReliabilityDynamicPath": "Paper_R_B3_ReliabilityCertainty",
    "Paper_B4_CRCF": "Paper_A5_RCD_Fusion",
}


def main():
    data_root = os.environ.get("DSFN_DATA_ROOT", str(DEFAULT_DATA_ROOT))
    split_dir = os.environ.get("DSFN_SPLIT_DIR", str(DEFAULT_SPLIT_DIR))
    print(json.dumps(validate_paper_data(data_root, split_dir)), flush=True)
    experiment = os.environ.get("DSFN_EXP_NAME", "Paper_A4_DSFN")
    if experiment not in REFERENCE_CONFIGS:
        raise ValueError(f"Unknown paper experiment: {experiment}")
    for key, value in list(os.environ.items()):
        if key.startswith("DSFN_"):
            original = ("SGLA_" + key[5:]).replace("_CRCF_", "_RCD_")
            os.environ[original] = value
    os.environ.update(
        SGLA_DATA_ROOT=data_root, SGLA_SPLIT_DIR=split_dir,
        SGLA_SAVE_ROOT=os.environ.get("DSFN_SAVE_ROOT", "runs"),
        SGLA_EXP_NAME=experiment, SGLA_USE_CHEAP_EXIT="0",
        SGLA_USE_SHALLOW_EXIT="0", SGLA_TRAIN_CHEAP_ONLY="0",
        SGLA_TRAIN_SHALLOW_ONLY="0", SGLA_USE_GLOBAL_BRANCH="0",
        SGLA_TEACHER_MODEL="", SGLA_TEACHER_PATH="", SGLA_TEACHER_KD_WEIGHT="0",
        SGLA_EVAL_FORCE_FULL="1", SGLA_EVAL_SOFT_SELECT="1", SGLA_CLEAN_TEST_ONLY="1",
    )
    from paper_experiments import reference_trainer as reference
    if "DSFN_DEVICE" in os.environ:
        reference.device = reference.torch.device(os.environ["DSFN_DEVICE"])
    reference.ABLATION_CONFIGS[experiment] = copy.deepcopy(
        reference.ABLATION_CONFIGS[REFERENCE_CONFIGS[experiment]]
    )
    reference.main()


if __name__ == "__main__":
    main()

"""Run paper training or checkpoint evaluation with one fixed data protocol."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from paper_protocol import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR, validate_paper_data

ROOT = Path(__file__).resolve().parent
ABLATIONS = [
    "Paper_A0_StaticAvg", "Paper_A1_DynamicDepth", "Paper_A2_DynamicDepthLoss",
    "Paper_A3_GTD", "Paper_A4_DSFN", "Paper_B1_Reliability",
    "Paper_B2_ReliabilityConsistency", "Paper_B3_ReliabilityDynamicPath",
]
BASELINES = "resnet1d,tcn,inceptiontime,lstm_fcn,drsn,mra_cnn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--suite", choices=["dsfn", "ablations", "baselines", "all"], default="dsfn")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--checkpoint-root", default="runs")
    parser.add_argument("--output-dir", default="results/paper")
    parser.add_argument("--seeds", default="42,2025,3407")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data_check = validate_paper_data(args.data_root, args.split_dir)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import torch
    import numpy as np
    metadata = dict(vars(args), python=platform.python_version(), pytorch=torch.__version__,
                    cuda=torch.version.cuda, numpy=np.__version__, data_validation=data_check)
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    common = ["--data-root", str(Path(args.data_root).resolve()), "--split-dir", str(Path(args.split_dir).resolve()),
              "--seeds", args.seeds, "--device", args.device]
    runs = str(Path(args.checkpoint_root).resolve())
    def run(script, extra, env=None):
        subprocess.run([sys.executable, str(ROOT / script)] + extra, cwd=ROOT, env=env, check=True)
    if args.action == "train":
        experiments = ABLATIONS if args.suite in {"all", "ablations"} else ["Paper_A4_DSFN"]
        if args.suite != "baselines":
            for experiment in experiments:
                for seed in args.seeds.split(","):
                    env = os.environ.copy()
                    env.update(DSFN_DATA_ROOT=common[1], DSFN_SPLIT_DIR=common[3], DSFN_SAVE_ROOT=runs,
                               DSFN_EXP_NAME=experiment, DSFN_SEED=seed, DSFN_DEVICE=args.device)
                    run("paper_training.py", [], env)
        if args.suite in {"all", "baselines"}:
            run("paper_experiments/run_baseline_suite.py", common + ["--models", BASELINES, "--save-root", runs])
    else:
        models = "dsfn"
        if args.suite == "baselines": models = BASELINES
        elif args.suite == "all": models += "," + BASELINES
        if args.suite == "ablations": models = ",".join(ABLATIONS)
        evaluation = common + ["--checkpoint-root", runs]
        run("paper_experiments/run_clean_suite.py", evaluation + ["--models", models, "--output-dir", str(output / "clean")])
        if args.suite == "all":
            run("paper_experiments/run_clean_suite.py", evaluation + ["--models", ",".join(ABLATIONS), "--output-dir", str(output / "ablations")])
        if args.suite != "ablations":
            for kind in ["noise", "mask"]:
                run(f"paper_experiments/run_{kind}_robustness_suite.py", evaluation + ["--models", models, "--output-dir", str(output / kind)])


if __name__ == "__main__":
    main()

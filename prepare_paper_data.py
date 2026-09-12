"""Import or verify the processed arrays used for the manuscript experiments."""
import argparse
import json
import shutil
from pathlib import Path

from paper_protocol import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR, validate_paper_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory containing the processed train_data.npy and train_labels.npy.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source = args.source_dir or args.output_dir
    result = validate_paper_data(source, args.split_dir)
    if not args.verify_only and source.resolve() != args.output_dir.resolve():
        # Check every destination before copying, and preserve existing arrays.
        for name in ("train_data.npy", "train_labels.npy"):
            if (args.output_dir / name).exists():
                raise FileExistsError(f"{args.output_dir / name} already exists. Use --verify-only to check it.")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train_data.npy", "train_labels.npy"):
            shutil.copy2(source / name, args.output_dir / name)
    print(json.dumps(result, indent=2))
    print("Paper data and fixed split verified.")


if __name__ == "__main__":
    main()

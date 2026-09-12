"""Validate the processed arrays and fixed protocol files used by the paper."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "data" / "paper"
DEFAULT_SPLIT_DIR = ROOT / "data" / "paper_split"
MANIFEST_PATH = ROOT / "data" / "paper_manifest.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_paper_data(data_root=DEFAULT_DATA_ROOT, split_dir=DEFAULT_SPLIT_DIR):
    """Reject a different data version, sample order, or split before evaluation."""
    data_root, split_dir = Path(data_root), Path(split_dir)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paths = {
        "train_data.npy": data_root / "train_data.npy",
        "train_labels.npy": data_root / "train_labels.npy",
        **{f"{s}_indices.npy": split_dir / f"{s}_indices.npy" for s in ("train", "val", "test")},
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Paper reproduction requires {path}. Run prepare_paper_data.py "
                "with the processed manuscript arrays; see data/README.md."
            )
        if file_sha256(path) != manifest["sha256"][name]:
            raise ValueError(
                f"{path} does not match the manuscript file (data, labels and sample "
                "order must all match). Do not pair regenerated windows with paper indices."
            )
    data = np.load(paths["train_data.npy"], mmap_mode="r", allow_pickle=False)
    labels = np.load(paths["train_labels.npy"], allow_pickle=False)
    if list(data.shape) != manifest["shape"] or str(data.dtype) != "float32":
        raise ValueError("Unexpected paper data shape or dtype.")
    if labels.shape != (len(data),) or str(labels.dtype) != "int64":
        raise ValueError("Unexpected label shape or dtype.")
    if len(labels) and (int(labels.min()) < 0 or int(labels.max()) >= len(manifest["class_names"])):
        raise ValueError("Unexpected class id in the label array.")
    indices = {s: np.load(paths[f"{s}_indices.npy"], allow_pickle=False) for s in ("train", "val", "test")}
    if not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(len(data))):
        raise ValueError("Split indices must cover every sample exactly once.")
    return {"samples": len(data), "shape": list(data.shape), "files_verified": 5}

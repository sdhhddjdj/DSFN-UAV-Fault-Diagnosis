from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["Normal", "Accel", "GPS", "Gyro", "Mag", "Baro"]
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
FAULT_SUFFIX_TO_LABEL = {
    "acce": "Accel",
    "gps": "GPS",
    "gyro": "Gyro",
    "mag": "Mag",
    "baro": "Baro",
}
STATUS_NAMES = {
    "hover": "Hover",
    "waypoint": "Waypoint",
    "velocity": "Velocity",
    "circling": "Circling",
    "acce": "Acceleration",
    "dece": "Deceleration",
}


@dataclass(frozen=True)
class FlightCase:
    archive: str
    flight_id: str
    root_name: str
    case_name: str
    label: str
    label_id: int
    flight_status: str
    combined_csv: str
    local_position_csv: str
    control_csv: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the RflyMAD 9-channel window dataset used in the DSFN manuscript."
    )
    parser.add_argument("--sensor-zip", type=Path, required=True)
    parser.add_argument("--normal-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--split-mode",
        choices=("window", "flight"),
        default="window",
        help="Use the manuscript's window-level split by default; 'flight' keeps logs disjoint.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _find_unique(names: list[str], suffix: str, flight_id: str) -> str:
    matches = [name for name in names if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {suffix} under {flight_id}, found {len(matches)}: {matches[:3]}"
        )
    return matches[0]


def discover_cases(archive_path: Path, normal_archive: bool) -> list[FlightCase]:
    cases: list[FlightCase] = []
    with zipfile.ZipFile(archive_path) as archive:
        grouped: dict[str, list[str]] = {}
        for raw_name in archive.namelist():
            name = raw_name.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if len(parts) < 3:
                continue
            flight_id = "/".join(parts[:2])
            grouped.setdefault(flight_id, []).append(name)

        for flight_id, names in sorted(grouped.items()):
            root_name, case_name = flight_id.split("/", 1)
            if normal_archive:
                label = "Normal"
                status_key = root_name.lower()
            else:
                root_parts = root_name.lower().rsplit("-", 1)
                if len(root_parts) != 2 or root_parts[1] not in FAULT_SUFFIX_TO_LABEL:
                    continue
                status_key, fault_key = root_parts
                label = FAULT_SUFFIX_TO_LABEL[fault_key]

            try:
                combined = _find_unique(names, "sensor_combined_0.csv", flight_id)
                local_position = _find_unique(names, "vehicle_local_position_0.csv", flight_id)
                control = _find_unique(names, "rfly_ctrl_lxl_0.csv", flight_id)
            except RuntimeError as error:
                print(f"[skip] {error}", flush=True)
                continue

            cases.append(
                FlightCase(
                    archive=str(archive_path.resolve()),
                    flight_id=f"{archive_path.stem}:{flight_id}",
                    root_name=root_name,
                    case_name=case_name,
                    label=label,
                    label_id=LABEL_TO_ID[label],
                    flight_status=STATUS_NAMES.get(status_key, status_key.title()),
                    combined_csv=combined,
                    local_position_csv=local_position,
                    control_csv=control,
                )
            )
    return cases


def split_flights(
    cases: list[FlightCase], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, str]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one.")

    frame = pd.DataFrame([asdict(case) for case in cases])
    train, remainder = train_test_split(
        frame,
        test_size=1.0 - train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=frame["label_id"],
    )
    relative_test = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
    val, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=remainder["label_id"],
    )
    split_map: dict[str, str] = {}
    for split_name, subset in (("train", train), ("val", val), ("test", test)):
        split_map.update({flight_id: split_name for flight_id in subset["flight_id"]})
    if len(split_map) != len(cases):
        raise RuntimeError("Flight split did not assign every case exactly once.")
    return split_map


def split_windows(
    labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int
) -> dict[str, np.ndarray]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than one.")
    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == label_id)
        rng.shuffle(class_indices)
        class_size = len(class_indices)
        train_size = int(round(train_ratio * class_size))
        val_size = int(round(val_ratio * class_size))
        if train_size + val_size >= class_size:
            train_size = max(class_size - 2, 1)
            val_size = 1
        train_parts.append(class_indices[:train_size])
        val_parts.append(class_indices[train_size : train_size + val_size])
        test_parts.append(class_indices[train_size + val_size :])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    test_indices = np.concatenate(test_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _read_csv(archive: zipfile.ZipFile, member: str, usecols: list[str]) -> pd.DataFrame:
    with archive.open(member) as stream:
        frame = pd.read_csv(stream, usecols=usecols)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _fault_interval(archive: zipfile.ZipFile, case: FlightCase) -> tuple[float, float]:
    control = _read_csv(archive, case.control_csv, ["timestamp", "id", "mode"])
    control = control.dropna(subset=["timestamp"])
    if case.label == "Normal":
        active = control
    else:
        ids = pd.to_numeric(control["id"], errors="coerce")
        modes = pd.to_numeric(control["mode"], errors="coerce")
        active = control[(ids != 1500) & (modes != 1500)]
    if active.empty:
        raise RuntimeError(f"No active interval found for {case.flight_id}")
    return float(active["timestamp"].min()), float(active["timestamp"].max())


def _interpolate_velocity(local_position: pd.DataFrame, target_time: np.ndarray) -> np.ndarray:
    local_position = local_position.dropna(subset=["timestamp", "vx", "vy", "vz"])
    local_position = local_position.sort_values("timestamp").drop_duplicates("timestamp")
    if len(local_position) < 2:
        raise RuntimeError("Fewer than two valid local-position rows are available.")
    source_time = local_position["timestamp"].to_numpy(dtype=np.float64)
    velocity = np.column_stack(
        [
            np.interp(target_time, source_time, local_position[name].to_numpy(dtype=np.float64))
            for name in ("vx", "vy", "vz")
        ]
    )
    return velocity.astype(np.float32)


def load_flight_signal(
    archive: zipfile.ZipFile, case: FlightCase
) -> tuple[np.ndarray, float, float]:
    start_time, end_time = _fault_interval(archive, case)
    combined_columns = [
        "timestamp",
        "gyro_rad[0]",
        "gyro_rad[1]",
        "gyro_rad[2]",
        "accelerometer_m_s2[0]",
        "accelerometer_m_s2[1]",
        "accelerometer_m_s2[2]",
    ]
    combined = _read_csv(archive, case.combined_csv, combined_columns)
    combined = combined.dropna(subset=combined_columns).sort_values("timestamp")
    combined = combined.drop_duplicates("timestamp")
    combined = combined[
        (combined["timestamp"] >= start_time) & (combined["timestamp"] <= end_time)
    ]
    if len(combined) < 2:
        raise RuntimeError(f"No sensor_combined samples in active interval for {case.flight_id}")

    target_time = combined["timestamp"].to_numpy(dtype=np.float64)
    gyro_accel = combined[combined_columns[1:]].to_numpy(dtype=np.float32)
    local_position = _read_csv(
        archive, case.local_position_csv, ["timestamp", "vx", "vy", "vz"]
    )
    velocity = _interpolate_velocity(local_position, target_time)
    signal = np.concatenate([gyro_accel, velocity], axis=1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    duration_seconds = (target_time[-1] - target_time[0]) / 1_000_000.0
    sample_rate = (len(target_time) - 1) / max(duration_seconds, 1e-6)
    return signal, float(duration_seconds), float(sample_rate)


def make_windows(signal: np.ndarray, window_size: int, stride: int) -> tuple[np.ndarray, list[int]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")
    length = len(signal)
    if length < 2:
        return np.empty((0, window_size, signal.shape[1]), dtype=np.float32), []
    if length < window_size:
        old_axis = np.linspace(0.0, 1.0, num=length)
        new_axis = np.linspace(0.0, 1.0, num=window_size)
        resized = np.column_stack(
            [np.interp(new_axis, old_axis, signal[:, channel]) for channel in range(signal.shape[1])]
        ).astype(np.float32)
        return resized[None, ...], [0]

    starts = list(range(0, length - window_size + 1, stride))
    tail_start = length - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    windows = np.stack([signal[start : start + window_size] for start in starts]).astype(np.float32)
    return windows, starts


def _write_manifests(
    output_dir: Path,
    cases: list[FlightCase],
    split_map: dict[str, str],
    sample_rows: list[dict[str, object]],
    split_mode: str,
) -> None:
    case_frame = pd.DataFrame([asdict(case) for case in cases])
    case_frame["split"] = case_frame["flight_id"].map(split_map)
    case_frame.to_csv(output_dir / "case_manifest.csv", index=False, encoding="utf-8-sig")
    sample_frame = pd.DataFrame(sample_rows)
    sample_frame.to_csv(output_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    if sample_frame.empty:
        summary = (
            case_frame.groupby(["split", "label"], observed=False)
            .agg(flights=("flight_id", "nunique"))
            .reset_index()
        )
        summary["windows"] = 0
    else:
        summary = (
            sample_frame.groupby(["split", "label"], observed=False)
            .agg(flights=("flight_id", "nunique"), windows=("sample_index", "size"))
            .reset_index()
        )
    summary.to_csv(output_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    if sample_frame.empty:
        split_groups = {name: set() for name in ("train", "val", "test")}
    else:
        split_groups = {
            name: set(sample_frame.loc[sample_frame["split"] == name, "flight_id"])
            for name in ("train", "val", "test")
        }
    leakage = {
        "train_val_overlap": sorted(split_groups["train"] & split_groups["val"]),
        "train_test_overlap": sorted(split_groups["train"] & split_groups["test"]),
        "val_test_overlap": sorted(split_groups["val"] & split_groups["test"]),
        "flight_disjoint": all(
            not overlap
            for overlap in (
                split_groups["train"] & split_groups["val"],
                split_groups["train"] & split_groups["test"],
                split_groups["val"] & split_groups["test"],
            )
        ),
        "partition_unit": split_mode,
    }
    (output_dir / "leakage_check.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if split_mode == "flight" and not leakage["flight_disjoint"]:
        raise RuntimeError("Flight-level leakage verification failed.")


def main() -> None:
    args = parse_args()
    for archive_path in (args.sensor_zip, args.normal_zip):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.sensor_zip, normal_archive=False)
    cases.extend(discover_cases(args.normal_zip, normal_archive=True))
    if not cases:
        raise RuntimeError("No valid flight cases were discovered.")
    if args.split_mode == "flight":
        split_map = split_flights(cases, args.train_ratio, args.val_ratio, args.split_seed)
    else:
        split_map = {case.flight_id: "window_level" for case in cases}
    print(f"Discovered {len(cases)} complete flight logs.", flush=True)

    if args.dry_run:
        _write_manifests(output_dir, cases, split_map, [], args.split_mode)
        return

    archives = {
        str(path.resolve()): zipfile.ZipFile(path)
        for path in (args.sensor_zip, args.normal_zip)
    }
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    sample_rows: list[dict[str, object]] = []
    sample_offset = 0
    try:
        for case_number, case in enumerate(cases, start=1):
            archive = archives[case.archive]
            signal, duration_seconds, sample_rate = load_flight_signal(archive, case)
            windows, starts = make_windows(signal, args.window_size, args.stride)
            if len(windows) == 0:
                print(f"[skip] no windows for {case.flight_id}", flush=True)
                continue
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), case.label_id, dtype=np.int64))
            for local_index, start in enumerate(starts):
                sample_rows.append(
                    {
                        "sample_index": sample_offset + local_index,
                        "flight_id": case.flight_id,
                        "split": split_map[case.flight_id] if args.split_mode == "flight" else "",
                        "label": case.label,
                        "label_id": case.label_id,
                        "flight_status": case.flight_status,
                        "window_start_row": start,
                        "active_duration_s": duration_seconds,
                        "estimated_sample_rate_hz": sample_rate,
                    }
                )
            sample_offset += len(windows)
            print(
                f"[{case_number:03d}/{len(cases)}] {case.flight_id}: "
                f"{len(signal)} rows -> {len(windows)} windows",
                flush=True,
            )
    finally:
        for archive in archives.values():
            archive.close()

    data = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.save(output_dir / "train_data.npy", data)
    np.save(output_dir / "train_labels.npy", labels)

    sample_frame = pd.DataFrame(sample_rows)
    if args.split_mode == "window":
        split_indices = split_windows(labels, args.train_ratio, args.val_ratio, args.split_seed)
        for split_name, indices in split_indices.items():
            sample_frame.loc[indices, "split"] = split_name
    else:
        split_indices = {
            split_name: sample_frame.loc[
                sample_frame["split"] == split_name, "sample_index"
            ].to_numpy(dtype=np.int64)
            for split_name in ("train", "val", "test")
        }
    split_prefix = "fixed_split" if args.split_mode == "window" else "fixed_group_split"
    split_dir = output_dir / f"{split_prefix}_80_10_10_seed{args.split_seed}"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, indices in split_indices.items():
        np.save(split_dir / f"{split_name}_indices.npy", indices)

    config = {
        "channels": ["gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z", "vel_x", "vel_y", "vel_z"],
        "class_names": CLASS_NAMES,
        "label_to_id": LABEL_TO_ID,
        "window_size": args.window_size,
        "stride": args.stride,
        "split_seed": args.split_seed,
        "split_mode": args.split_mode,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": 1.0 - args.train_ratio - args.val_ratio},
        "fault_interval_rule": "rfly_ctrl_lxl rows where id != 1500 and mode != 1500",
        "normal_interval_rule": "complete logged interval",
        "reference_timeline": "sensor_combined timestamps",
        "velocity_alignment": "linear interpolation of vehicle_local_position vx/vy/vz",
        "short_interval_rule": "linear resize to one 1024-point window",
        "data_shape": list(data.shape),
    }
    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "label_map.json").write_text(
        json.dumps(LABEL_TO_ID, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_manifests(
        output_dir,
        cases,
        split_map,
        sample_frame.to_dict(orient="records"),
        args.split_mode,
    )
    print(f"Saved dataset: {data.shape}, labels: {labels.shape}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
